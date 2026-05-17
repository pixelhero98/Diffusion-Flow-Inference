from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOBConfig
from modules import build_mlp


class _HistoryLSTMEncoder(nn.Module):
    def __init__(self, cfg: LOBConfig):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.input_proj = nn.Linear(cfg.state_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(hist)
        _, (h_n, _) = self.rnn(x)
        return self.out(h_n[-1])


class DeepMarketCGANBaseline(nn.Module):
    """PyTorch adaptation of the official DeepMarket CGAN baseline."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.state_dim
        hidden_dim = cfg.model.hidden_dim
        noise_dim = int(cfg.model.gan_noise_dim)
        self.recon_weight = float(cfg.model.cgan_recon_weight)
        self.noise_dim = noise_dim

        self.generator_hist = _HistoryLSTMEncoder(cfg)
        self.discriminator_hist = _HistoryLSTMEncoder(cfg)
        self.generator = build_mlp(
            hidden_dim + noise_dim,
            hidden_dim,
            state_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.discriminator = build_mlp(
            hidden_dim + state_dim,
            hidden_dim,
            1,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )

    def _noise(self, hist: torch.Tensor) -> torch.Tensor:
        return torch.randn(hist.shape[0], self.noise_dim, device=hist.device, dtype=hist.dtype)

    def generate(self, hist: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = self.generator_hist(hist)
        z = self._noise(hist) if noise is None else noise
        return self.generator(torch.cat([ctx, z], dim=-1))

    def discriminate(self, hist: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx = self.discriminator_hist(hist)
        return self.discriminator(torch.cat([ctx, x], dim=-1))

    def adversarial_step(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        *,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        real_targets = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        fake_targets = torch.zeros_like(real_targets)

        # Update discriminator.
        with torch.no_grad():
            fake_detached = self.generate(hist)
        real_logits = self.discriminate(hist, x)
        fake_logits = self.discriminate(hist, fake_detached)
        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_logits, real_targets)
            + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        )
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(self.discriminator_hist.parameters(), grad_clip)
        opt_d.step()

        # Update generator.
        fake = self.generate(hist)
        fake_logits = self.discriminate(hist, fake)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)
        recon_loss = F.mse_loss(fake, x)
        g_loss = adv_loss + self.recon_weight * recon_loss
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(self.generator_hist.parameters(), grad_clip)
        opt_g.step()

        return {
            "loss": float(g_loss.detach().cpu()),
            "gen_total": float(g_loss.detach().cpu()),
            "gen_adv": float(adv_loss.detach().cpu()),
            "gen_recon": float(recon_loss.detach().cpu()),
            "disc": float(d_loss.detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        del cond, steps
        return self.generate(hist)


class DeepMarketTRADESBaseline(nn.Module):
    """PyTorch adaptation of the official DeepMarket TRADES diffusion baseline."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        self.diffusion_steps = int(cfg.model.diffusion_steps)
        hidden_dim = cfg.model.hidden_dim
        state_dim = cfg.state_dim

        self.history_encoder = _HistoryLSTMEncoder(cfg)
        self.time_embed = nn.Embedding(self.diffusion_steps, hidden_dim)
        self.eps_net = build_mlp(
            state_dim + hidden_dim + hidden_dim,
            hidden_dim,
            state_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )

        betas = torch.linspace(1e-4, 2e-2, self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar), persistent=False)

    def _predict_eps(self, x_t: torch.Tensor, hist: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        hist_ctx = self.history_encoder(hist)
        t_emb = self.time_embed(t_idx)
        return self.eps_net(torch.cat([x_t, hist_ctx, t_emb], dim=-1))

    def loss(self, x: torch.Tensor, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, **_: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
        del cond
        batch_size = x.shape[0]
        t_idx = torch.randint(0, self.diffusion_steps, (batch_size,), device=x.device)
        eps = torch.randn_like(x)
        x_t = self.sqrt_alpha_bar[t_idx][:, None] * x + self.sqrt_one_minus_alpha_bar[t_idx][:, None] * eps
        eps_hat = self._predict_eps(x_t, hist, t_idx)
        loss = F.mse_loss(eps_hat, eps)
        return loss, {
            "loss": float(loss.detach().cpu()),
            "eps_mse": float(loss.detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        del cond
        batch_size = hist.shape[0]
        state_dim = self.cfg.state_dim
        x = torch.randn(batch_size, state_dim, device=hist.device, dtype=hist.dtype)
        n_steps = int(max(1, self.cfg.sample.steps if steps is None else steps))
        indices = torch.linspace(self.diffusion_steps - 1, 0, steps=n_steps, device=hist.device)
        indices = torch.unique(indices.round().to(dtype=torch.long), sorted=False)
        indices = torch.flip(indices, dims=[0])
        prev_indices = list(indices[1:].tolist()) + [-1]

        for t_idx_value, prev_t_value in zip(indices.tolist(), prev_indices):
            t_idx = torch.full((batch_size,), int(t_idx_value), device=hist.device, dtype=torch.long)
            eps_hat = self._predict_eps(x, hist, t_idx)
            alpha_bar_t = self.alpha_bar[t_idx][:, None]
            x0_hat = (x - self.sqrt_one_minus_alpha_bar[t_idx][:, None] * eps_hat) / self.sqrt_alpha_bar[t_idx][:, None]
            if prev_t_value < 0:
                x = x0_hat
                continue
            prev_t = torch.full((batch_size,), int(prev_t_value), device=hist.device, dtype=torch.long)
            x = self.sqrt_alpha_bar[prev_t][:, None] * x0_hat + self.sqrt_one_minus_alpha_bar[prev_t][:, None] * eps_hat
        return x


__all__ = [
    "DeepMarketCGANBaseline",
    "DeepMarketTRADESBaseline",
]
