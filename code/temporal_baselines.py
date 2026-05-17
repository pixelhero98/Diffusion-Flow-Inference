from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOBConfig
from modules import build_mlp


def _gaussian_kl(
    q_mean: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mean: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    return 0.5 * torch.mean(
        p_logvar - q_logvar + (q_var + (q_mean - p_mean) ** 2) / (p_var + 1e-8) - 1.0
    )


def _sample_diag_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


class _HistoryGRUEncoder(nn.Module):
    def __init__(self, cfg: LOBConfig, *, bidirectional: bool = False):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(cfg.state_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.seq_proj = nn.Sequential(
            nn.Linear(rnn_out_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(hist)
        seq_out, _ = self.rnn(x)
        seq_ctx = self.seq_proj(seq_out)
        summary = self.summary_proj(seq_ctx[:, -1, :])
        return seq_ctx, summary


class TimeCausalVAEBaseline(nn.Module):
    """History-conditioned recurrent CVAE inspired by TimeCausalVAE."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.model.hidden_dim
        state_dim = cfg.state_dim
        latent_dim = int(cfg.model.baseline_latent_dim)
        self.beta = float(cfg.model.vae_kl_weight)

        self.history_encoder = _HistoryGRUEncoder(cfg, bidirectional=False)
        self.posterior = build_mlp(
            hidden_dim + state_dim,
            hidden_dim,
            2 * latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.prior = build_mlp(
            hidden_dim,
            hidden_dim,
            2 * latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.decoder = build_mlp(
            hidden_dim + latent_dim,
            hidden_dim,
            state_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )

    def _posterior_stats(self, x: torch.Tensor, summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.posterior(torch.cat([x, summary], dim=-1))
        return torch.chunk(stats, 2, dim=-1)

    def _prior_stats(self, summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.prior(summary)
        return torch.chunk(stats, 2, dim=-1)

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **_: Dict[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del cond
        _, summary = self.history_encoder(hist)
        q_mean, q_logvar = self._posterior_stats(x, summary)
        p_mean, p_logvar = self._prior_stats(summary)
        z = _sample_diag_gaussian(q_mean, q_logvar)
        x_hat = self.decoder(torch.cat([summary, z], dim=-1))
        recon = F.mse_loss(x_hat, x)
        kl = _gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        loss = recon + self.beta * kl
        return loss, {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        del cond, steps
        _, summary = self.history_encoder(hist)
        p_mean, p_logvar = self._prior_stats(summary)
        z = _sample_diag_gaussian(p_mean, p_logvar)
        return self.decoder(torch.cat([summary, z], dim=-1))


class TimeGANBaseline(nn.Module):
    """History-conditioned TimeGAN-style latent adversarial generator."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.model.hidden_dim
        state_dim = cfg.state_dim
        latent_dim = int(cfg.model.baseline_latent_dim)
        self.latent_dim = latent_dim
        self.supervision_weight = float(cfg.model.timegan_supervision_weight)
        self.moment_weight = float(cfg.model.timegan_moment_weight)

        self.history_encoder = _HistoryGRUEncoder(cfg, bidirectional=False)
        self.embedder = build_mlp(
            state_dim,
            hidden_dim,
            latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.recovery = build_mlp(
            latent_dim,
            hidden_dim,
            state_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.generator = build_mlp(
            hidden_dim + latent_dim,
            hidden_dim,
            latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.supervisor = build_mlp(
            hidden_dim + latent_dim,
            hidden_dim,
            latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.discriminator = build_mlp(
            hidden_dim + latent_dim,
            hidden_dim,
            1,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )

    def _noise(self, summary: torch.Tensor) -> torch.Tensor:
        return torch.randn(summary.shape[0], self.latent_dim, device=summary.device, dtype=summary.dtype)

    def _generate_latent(self, summary: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self._noise(summary) if noise is None else noise
        e_hat = self.generator(torch.cat([summary, z], dim=-1))
        return self.supervisor(torch.cat([summary, e_hat], dim=-1))

    @staticmethod
    def _moment_loss(fake_x: torch.Tensor, real_x: torch.Tensor) -> torch.Tensor:
        fake_std, fake_mean = torch.std_mean(fake_x, dim=0, unbiased=False)
        real_std, real_mean = torch.std_mean(real_x, dim=0, unbiased=False)
        return torch.mean(torch.abs(fake_mean - real_mean)) + torch.mean(torch.abs(fake_std - real_std))

    def adversarial_step(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        *,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        summary = self.history_encoder(hist)[1]
        real_latent = self.embedder(x)

        real_targets = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        fake_targets = torch.zeros_like(real_targets)

        # Embedder / recovery update.
        recon_x = self.recovery(real_latent)
        supervised_real = self.supervisor(torch.cat([summary, real_latent], dim=-1))
        ae_loss = F.mse_loss(recon_x, x)
        sup_loss = F.mse_loss(supervised_real, real_latent.detach())
        pre_loss = ae_loss + self.supervision_weight * sup_loss
        opt_g.zero_grad(set_to_none=True)
        pre_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.history_encoder.parameters()) + list(self.embedder.parameters()) + list(self.recovery.parameters()) + list(self.supervisor.parameters()),
            grad_clip,
        )
        opt_g.step()

        # Discriminator update.
        with torch.no_grad():
            fake_latent_detached = self._generate_latent(summary)
        real_logits = self.discriminator(torch.cat([summary.detach(), real_latent.detach()], dim=-1))
        fake_logits = self.discriminator(torch.cat([summary.detach(), fake_latent_detached], dim=-1))
        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_logits, real_targets)
            + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        )
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), grad_clip)
        opt_d.step()

        # Generator update.
        summary = self.history_encoder(hist)[1]
        real_latent = self.embedder(x).detach()
        fake_latent = self._generate_latent(summary)
        fake_logits = self.discriminator(torch.cat([summary, fake_latent], dim=-1))
        fake_x = self.recovery(fake_latent)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)
        sup_gen = F.mse_loss(fake_latent, real_latent)
        moment = self._moment_loss(fake_x, x)
        g_loss = adv_loss + self.supervision_weight * sup_gen + self.moment_weight * moment
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.history_encoder.parameters())
            + list(self.embedder.parameters())
            + list(self.recovery.parameters())
            + list(self.generator.parameters())
            + list(self.supervisor.parameters()),
            grad_clip,
        )
        opt_g.step()

        return {
            "loss": float(g_loss.detach().cpu()),
            "gen_total": float(g_loss.detach().cpu()),
            "ae": float(ae_loss.detach().cpu()),
            "sup": float(sup_gen.detach().cpu()),
            "adv": float(adv_loss.detach().cpu()),
            "moment": float(moment.detach().cpu()),
            "disc": float(d_loss.detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        del cond, steps
        summary = self.history_encoder(hist)[1]
        fake_latent = self._generate_latent(summary)
        return self.recovery(fake_latent)


class KoVAEBaseline(nn.Module):
    """History-conditioned Koopman VAE baseline."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.model.hidden_dim
        state_dim = cfg.state_dim
        latent_dim = int(cfg.model.baseline_latent_dim)
        self.latent_dim = latent_dim
        self.beta = float(cfg.model.vae_kl_weight)
        self.pred_weight = float(cfg.model.kovae_pred_weight)
        self.ridge = float(cfg.model.kovae_ridge)

        self.history_encoder = _HistoryGRUEncoder(cfg, bidirectional=True)
        self.hist_latent = nn.Linear(hidden_dim, latent_dim)
        self.posterior = build_mlp(
            hidden_dim + state_dim,
            hidden_dim,
            2 * latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.prior_logvar = build_mlp(
            hidden_dim,
            hidden_dim,
            latent_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )
        self.decoder = build_mlp(
            hidden_dim + latent_dim,
            hidden_dim,
            state_dim,
            dropout=cfg.model.dropout,
            use_res=cfg.model.use_res_mlp,
        )

    def _compute_operator(self, z_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        past = z_hist[:, :-1, :].reshape(-1, self.latent_dim)
        future = z_hist[:, 1:, :].reshape(-1, self.latent_dim)
        eye = torch.eye(self.latent_dim, device=z_hist.device, dtype=z_hist.dtype)
        gram = past.T @ past + self.ridge * eye
        cross = past.T @ future
        operator = torch.linalg.solve(gram, cross)
        pred = z_hist[:, :-1, :] @ operator
        pred_err = F.mse_loss(pred, z_hist[:, 1:, :])
        return operator, pred_err

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **_: Dict[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del cond
        seq_ctx, summary = self.history_encoder(hist)
        z_hist = self.hist_latent(seq_ctx)
        operator, pred_err = self._compute_operator(z_hist)
        prior_mean = z_hist[:, -1, :] @ operator
        prior_logvar = self.prior_logvar(summary)

        q_stats = self.posterior(torch.cat([x, summary], dim=-1))
        q_mean, q_logvar = torch.chunk(q_stats, 2, dim=-1)
        z = _sample_diag_gaussian(q_mean, q_logvar)
        x_hat = self.decoder(torch.cat([summary, z], dim=-1))

        recon = F.mse_loss(x_hat, x)
        kl = _gaussian_kl(q_mean, q_logvar, prior_mean, prior_logvar)
        loss = recon + self.beta * kl + self.pred_weight * pred_err
        return loss, {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "pred": float(pred_err.detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, cond: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        del cond, steps
        seq_ctx, summary = self.history_encoder(hist)
        z_hist = self.hist_latent(seq_ctx)
        operator, _ = self._compute_operator(z_hist)
        prior_mean = z_hist[:, -1, :] @ operator
        prior_logvar = self.prior_logvar(summary)
        z = _sample_diag_gaussian(prior_mean, prior_logvar)
        return self.decoder(torch.cat([summary, z], dim=-1))


__all__ = [
    "TimeCausalVAEBaseline",
    "TimeGANBaseline",
    "KoVAEBaseline",
]
