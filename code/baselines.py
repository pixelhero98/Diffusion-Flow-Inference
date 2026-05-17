from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from conditioning import ConditioningCache, SharedConditioningBackbone
from config import LOBConfig
from modules import MLP, TransformerFUNet, build_mlp


class RectifiedFlowLOB(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.sample_state_dim
        hidden_dim = cfg.model.hidden_dim
        cond_dim = hidden_dim if cfg.model.cond_dim > 0 else 0
        self.backbone = SharedConditioningBackbone(cfg)
        self.fu_net_type = str(cfg.model.fu_net_type).lower()
        base_cond_dim = 3 * hidden_dim

        if self.fu_net_type == "transformer":
            cond_in_dim = base_cond_dim + cond_dim
            self.v_cond_proj = build_mlp(
                cond_in_dim,
                hidden_dim,
                hidden_dim,
                dropout=cfg.model.dropout,
                use_res=cfg.model.use_res_mlp,
            )
            self.v_net = TransformerFUNet(cfg)
        elif self.fu_net_type in {"mlp", "resmlp"}:
            use_res = cfg.model.use_res_mlp if self.fu_net_type == "resmlp" else False
            in_dim = state_dim + base_cond_dim + cond_dim
            self.v_cond_proj = None
            self.v_net = build_mlp(
                in_dim,
                hidden_dim,
                state_dim,
                dropout=cfg.model.dropout,
                use_res=use_res,
            )
        else:
            raise ValueError(f"Unknown fu_net_type={cfg.model.fu_net_type}")

    def _conditioning_parts(self, cond_state) -> List[torch.Tensor]:
        parts = [cond_state.ctx, cond_state.ctx_summary, cond_state.t_emb]
        if cond_state.cond_emb is not None:
            parts.append(cond_state.cond_emb)
        return parts

    def _field_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
    ) -> torch.Tensor:
        cond_state = self.backbone.build_conditioning(
            hist=hist,
            x_ref=x_t,
            t=t,
            cond=cond,
            cache=conditioning_cache,
        )
        if self.fu_net_type == "transformer":
            adaln_cond = self.v_cond_proj(torch.cat(self._conditioning_parts(cond_state), dim=-1))
            return self.v_net(x_t, cond_state.ctx_tokens, adaln_cond)

        parts = [x_t, *self._conditioning_parts(cond_state)]
        return self.v_net(torch.cat(parts, dim=-1))

    def v_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
    ) -> torch.Tensor:
        return self._field_forward(x_t, t, hist, cond=cond, conditioning_cache=conditioning_cache)

    def fm_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        z = torch.randn_like(x)
        t = torch.rand(batch_size, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x
        v_target = x - z
        v_hat = self.v_forward(x_t, t, hist, cond=cond)
        return F.mse_loss(v_hat, v_target)

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        state_dim = self.cfg.sample_state_dim
        batch_size = hist.shape[0]
        x = torch.randn(batch_size, state_dim, device=hist.device)
        n_steps = int(max(1, self.cfg.sample.steps if steps is None else steps))
        dt = 1.0 / float(n_steps)
        for i in range(n_steps):
            t = torch.full((batch_size, 1), float(i) / float(n_steps), device=hist.device)
            x = x + dt * self._field_forward(x, t, hist, cond=cond)
        return x


class InvertiblePermutation(nn.Module):
    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        perm = torch.randperm(dim, generator=generator)
        inv = torch.argsort(perm)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inv]


class CouplingBackbone(nn.Module):
    def __init__(self, cfg: LOBConfig, x_a_dim: int):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        cond_dim = hidden_dim if cfg.model.cond_dim > 0 else 0
        self.net = nn.Sequential(
            nn.Linear(x_a_dim + hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x_a: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [x_a, ctx, t_emb]
        if cond_emb is not None:
            parts.append(cond_emb)
        return self.net(torch.cat(parts, dim=-1))


class AffineCoupling(nn.Module):
    def __init__(self, cfg: LOBConfig, dim: int, mask: torch.Tensor, backbone: Optional[CouplingBackbone] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.register_buffer("mask", mask.to(dtype=torch.bool), persistent=False)
        a_dim = int(self.mask.sum().item())
        b_dim = dim - a_dim
        self.backbone = backbone if backbone is not None else CouplingBackbone(cfg, x_a_dim=a_dim)
        self.to_s = nn.Linear(cfg.model.hidden_dim, b_dim)
        self.to_t = nn.Linear(cfg.model.hidden_dim, b_dim)

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.mask], x[:, ~self.mask]

    def _merge(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        x = torch.empty(x_a.shape[0], self.dim, device=x_a.device, dtype=x_a.dtype)
        x[:, self.mask] = x_a
        x[:, ~self.mask] = x_b
        return x

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_a, x_b = self._split(x)
        hidden = self.backbone(x_a, ctx, t_emb, cond_emb)
        scale = self.to_s(hidden).clamp(-self.cfg.nf.flow_scale_clip, self.cfg.nf.flow_scale_clip)
        shift = self.to_t(hidden)
        y_b = x_b * torch.exp(scale) + shift
        logdet = scale.sum(dim=-1)
        return self._merge(x_a, y_b), logdet, hidden

    def inverse(self, y: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_a, y_b = self._split(y)
        hidden = self.backbone(y_a, ctx, t_emb, cond_emb)
        scale = self.to_s(hidden).clamp(-self.cfg.nf.flow_scale_clip, self.cfg.nf.flow_scale_clip)
        shift = self.to_t(hidden)
        x_b = (y_b - shift) * torch.exp(-scale)
        logdet = (-scale).sum(dim=-1)
        return self._merge(y_a, x_b), logdet, hidden


class ConditionalRealNVP(nn.Module):
    def __init__(self, cfg: LOBConfig, dim: int, seed: int = 0, shared_backbones: Optional[List[CouplingBackbone]] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.perms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer_idx in range(cfg.nf.flow_layers):
            self.perms.append(InvertiblePermutation(dim, seed=seed + layer_idx))
            mask = torch.zeros(dim, dtype=torch.bool)
            idx = torch.randperm(dim, generator=generator)[: dim // 2]
            mask[idx] = True
            backbone = None if shared_backbones is None else shared_backbones[layer_idx]
            self.couplings.append(AffineCoupling(cfg, dim, mask=mask, backbone=backbone))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hidden_states: List[torch.Tensor] = []
        states: List[torch.Tensor] = [x]
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for perm, coupling in zip(self.perms, self.couplings):
            z = perm(z)
            z, layer_logdet, hidden = coupling(z, ctx, t_emb, cond_emb)
            logdet = logdet + layer_logdet
            hidden_states.append(hidden)
            states.append(z)
        return z, logdet, states, hidden_states

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hidden_states: List[torch.Tensor] = []
        states: List[torch.Tensor] = [z]
        logdet = torch.zeros(z.shape[0], device=z.device)
        x = z
        for perm, coupling in reversed(list(zip(self.perms, self.couplings))):
            x, layer_logdet, hidden = coupling.inverse(x, ctx, t_emb, cond_emb)
            x = perm.inverse(x)
            logdet = logdet + layer_logdet
            hidden_states.append(hidden)
            states.append(x)
        return x, logdet, states, hidden_states


class BiFlowNFBaseline(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SharedConditioningBackbone(cfg)
        state_dim = cfg.sample_state_dim
        shared_backbones: Optional[List[CouplingBackbone]] = None
        if cfg.nf.share_coupling_backbone:
            shared_backbones = [CouplingBackbone(cfg, x_a_dim=state_dim // 2) for _ in range(cfg.nf.flow_layers)]
        self.forward_flow = ConditionalRealNVP(cfg, dim=state_dim, seed=0, shared_backbones=shared_backbones)
        self.reverse_flow = ConditionalRealNVP(cfg, dim=state_dim, seed=1337, shared_backbones=shared_backbones)
        hidden_dim = cfg.model.hidden_dim
        self.align_heads = nn.ModuleList([MLP(hidden_dim, hidden_dim, state_dim, dropout=0.0) for _ in range(cfg.nf.flow_layers)])
        self._forward_frozen = False

    def freeze_forward(self) -> None:
        for param in self.forward_flow.parameters():
            param.requires_grad_(False)
        self._forward_frozen = True

    def conditioning(self, hist: torch.Tensor, x_ref: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cond_state = self.backbone.build_conditioning(hist=hist, x_ref=x_ref, t=None, cond=cond, force_zero_t=True)
        return cond_state.ctx, cond_state.t_emb, cond_state.cond_emb

    def log_prob(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx, t_emb, cond_emb = self.conditioning(hist, x, cond)
        z, logdet, _, _ = self.forward_flow(x, ctx, t_emb, cond_emb)
        base = -0.5 * (z**2).sum(dim=-1) - 0.5 * z.shape[-1] * math.log(2.0 * math.pi)
        return base + logdet

    def nll_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (-self.log_prob(x, hist, cond)).mean()

    def reverse_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._forward_frozen:
            raise RuntimeError("Call freeze_forward() before training reverse with reverse_loss().")
        ctx, t_emb, cond_emb = self.conditioning(hist, x, cond)
        with torch.no_grad():
            z, _, forward_states, _ = self.forward_flow(x, ctx, t_emb, cond_emb)
        x_hat, _, _, reverse_hidden = self.reverse_flow.inverse(z, ctx, t_emb, cond_emb)
        rec = F.mse_loss(x_hat, x)
        align = 0.0
        for layer_idx in range(self.cfg.nf.flow_layers):
            align = align + F.mse_loss(self.align_heads[layer_idx](reverse_hidden[layer_idx]), forward_states[layer_idx + 1])
        align = align / float(self.cfg.nf.flow_layers)
        loss = rec + 0.5 * align
        stats = {"rec": float(rec.detach().cpu()), "align": float(align.detach().cpu())}
        return loss, stats

    def biflow_loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.reverse_loss(x, hist, cond)

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        state_dim = self.cfg.sample_state_dim
        batch_size = hist.shape[0]
        z = torch.randn(batch_size, state_dim, device=hist.device)
        ctx, t_emb, cond_emb = self.conditioning(hist, z, cond)
        x, _, _, _ = self.reverse_flow.inverse(z, ctx, t_emb, cond_emb)
        return x


BiFlowLOB = RectifiedFlowLOB
BiFlowNFLOB = BiFlowNFBaseline


__all__ = [
    "RectifiedFlowLOB",
    "BiFlowLOB",
    "InvertiblePermutation",
    "CouplingBackbone",
    "AffineCoupling",
    "ConditionalRealNVP",
    "BiFlowNFBaseline",
    "BiFlowNFLOB",
]
