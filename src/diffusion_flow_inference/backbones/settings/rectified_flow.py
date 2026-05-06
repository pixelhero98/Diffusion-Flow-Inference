from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_flow_inference.backbones.settings.conditioning import ConditioningCache, SharedConditioningBackbone
from diffusion_flow_inference.backbones.settings.config import LOBConfig
from diffusion_flow_inference.backbones.settings.modules import TransformerFUNet, build_mlp


class RectifiedFlowLOB(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.sample_state_dim
        hidden_dim = cfg.model.hidden_dim
        cond_dim = hidden_dim if cfg.model.cond_dim > 0 else 0
        self.field_parameterization = str(cfg.model.field_parameterization).lower()
        if self.field_parameterization not in {"instantaneous", "average"}:
            raise ValueError(f"Unknown field_parameterization={cfg.model.field_parameterization}")
        self.uses_step_conditioning = self.field_parameterization == "average"
        self.backbone = SharedConditioningBackbone(cfg)
        self.fu_net_type = str(cfg.model.fu_net_type).lower()
        step_cond_dim = hidden_dim if self.uses_step_conditioning else 0
        base_cond_dim = 3 * hidden_dim

        if self.fu_net_type == "transformer":
            cond_in_dim = base_cond_dim + step_cond_dim + cond_dim
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
            in_dim = state_dim + base_cond_dim + step_cond_dim + cond_dim
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
        if cond_state.h_emb is not None:
            parts.append(cond_state.h_emb)
        if cond_state.cond_emb is not None:
            parts.append(cond_state.cond_emb)
        return parts

    def _field_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
    ) -> torch.Tensor:
        step_h = None
        if self.uses_step_conditioning:
            step_h = torch.zeros_like(t) if h is None else h
        cond_state = self.backbone.build_conditioning(
            hist=hist,
            x_ref=x_t,
            t=t,
            h=step_h,
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
        return self._field_forward(x_t, t, hist, cond=cond, h=None, conditioning_cache=conditioning_cache)

    def u_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        h: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
    ) -> torch.Tensor:
        return self._field_forward(x_t, t, hist, cond=cond, h=h, conditioning_cache=conditioning_cache)

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
            h = torch.full((batch_size, 1), dt, device=hist.device) if self.uses_step_conditioning else None
            x = x + dt * self._field_forward(x, t, hist, cond=cond, h=h)
        return x


__all__ = ["RectifiedFlowLOB"]
