from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusion_flow_inference.models.conditioning import ConditioningCache
from diffusion_flow_inference.models.rectified_flow import RectifiedFlow

SUPPORTED_SOLVERS = ("euler", "heun", "midpoint_rk2", "dpmpp2m")


def _solve_linear_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Solve a finite square linear assignment problem with the Hungarian algorithm."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"Expected a square cost matrix, got shape={tuple(cost.shape)}")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("Linear-assignment costs must all be finite.")

    matrix = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    n = len(matrix)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = matrix[i0 - 1]
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = torch.empty(n, dtype=torch.long)
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment.to(device=cost.device)


class OTFlow(RectifiedFlow):
    @torch.no_grad()
    def _match_minibatch_ot(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if x.shape[0] <= 1 or not bool(self.cfg.fm.use_minibatch_ot):
            identity = torch.arange(x.shape[0], device=x.device)
            zero_cost = x.new_tensor(0.0)
            return x, hist, cond, zero_cost, identity

        cost = torch.cdist(z, x, p=2).pow(2)
        perm = _solve_linear_assignment(cost)
        matched_x = x.index_select(0, perm)
        matched_hist = hist.index_select(0, perm)
        matched_cond = None if cond is None else cond.index_select(0, perm)
        matched_cost = cost[torch.arange(cost.shape[0], device=cost.device), perm].mean()
        return matched_x, matched_hist, matched_cond, matched_cost, perm

    def _sample_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        *,
        conditioning_cache: ConditioningCache,
    ) -> torch.Tensor:
        return self.v_forward(x, t, hist, conditioning_cache=conditioning_cache)

    def _prediction_horizon(self) -> int:
        return int(self.cfg.prediction_horizon)

    def _sample_state_dim(self) -> int:
        return int(self.cfg.sample_state_dim)

    def _snapshot_dim(self) -> int:
        return int(self.cfg.snapshot_dim)

    def _is_non_autoregressive(self) -> bool:
        return self._prediction_horizon() > 1

    def _future_training_target(
        self,
        tgt: torch.Tensor,
        fut: Optional[torch.Tensor],
    ) -> torch.Tensor:
        horizon = self._prediction_horizon()
        if horizon <= 1:
            return tgt
        if fut is None:
            raise ValueError(
                "Non-autoregressive OTFlow requires dataset batches with future trajectories."
            )
        required_future = horizon - 1
        if fut.ndim != 3 or fut.shape[0] != tgt.shape[0] or fut.shape[2] != self._snapshot_dim():
            raise ValueError(
                "Future trajectories must have shape [batch, future_steps, snapshot_dim], "
                f"got {tuple(fut.shape)}."
            )
        if int(fut.shape[1]) < required_future:
            raise ValueError(
                f"Non-autoregressive OTFlow requires at least {required_future} future steps, "
                f"but got fut.shape[1]={int(fut.shape[1])}."
            )
        block = torch.cat([tgt[:, None, :], fut[:, :required_future, :]], dim=1)
        return block.reshape(tgt.shape[0], -1)

    def _reshape_sample_block(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self._prediction_horizon(), self._snapshot_dim())

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self._is_non_autoregressive():
            x = self._future_training_target(x, fut)

        batch_size = x.shape[0]
        z = torch.randn_like(x)
        x_target, hist_target, cond_target, ot_cost, _ = self._match_minibatch_ot(
            x=x,
            hist=hist,
            cond=cond,
            z=z,
        )
        t = torch.rand(batch_size, 1, device=x.device, dtype=x.dtype)
        x_t = (1.0 - t) * z + t * x_target
        v_target = x_target - z

        v_hat = self.v_forward(x_t, t, hist_target, cond=cond_target)
        loss = F.mse_loss(v_hat, v_target)
        loss_value = float(loss.detach().cpu())
        logs = {
            "mean": loss_value,
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
            "loss": loss_value,
        }
        return loss, logs

    def _resolve_solver_name(self, solver: Optional[str]) -> str:
        solver_name = str(self.cfg.sample.solver if solver is None else solver).lower().strip()
        if solver_name not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unknown sample solver={solver_name!r}; expected one of {SUPPORTED_SOLVERS}."
            )
        return solver_name

    def _resolved_time_grid(self, n_steps: int) -> Tuple[float, ...]:
        raw_grid = tuple(float(x) for x in self.cfg.sample.time_grid)
        if len(raw_grid) == 0:
            return tuple(float(i) / float(n_steps) for i in range(int(n_steps) + 1))
        if len(raw_grid) != int(n_steps) + 1:
            raise ValueError(
                f"sample.time_grid must have length n_steps + 1 ({int(n_steps) + 1}), got {len(raw_grid)}."
            )
        if not all(math.isfinite(node) for node in raw_grid):
            raise ValueError("sample.time_grid must contain only finite values.")
        if abs(float(raw_grid[0])) > 1e-8 or abs(float(raw_grid[-1]) - 1.0) > 1e-8:
            raise ValueError("sample.time_grid must start at 0.0 and end at 1.0.")
        if any(float(right) <= float(left) for left, right in zip(raw_grid, raw_grid[1:])):
            raise ValueError("sample.time_grid must be strictly increasing.")
        return raw_grid

    def _top_of_book_feature_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_dim = self._snapshot_dim()
        weights = torch.ones(base_dim, device=device, dtype=dtype)
        levels = int(self.cfg.data.levels)
        if int(weights.numel()) < 2:
            return weights.repeat(self._prediction_horizon())

        weights[0] = 2.0
        weights[1] = 2.0
        ask_gap_start = 2
        bid_gap_start = ask_gap_start + max(0, levels - 1)
        size_start = bid_gap_start + max(0, levels - 1)

        for depth in range(max(0, levels - 1)):
            decay = 1.0 / float(depth + 1)
            ask_idx = ask_gap_start + depth
            bid_idx = bid_gap_start + depth
            if ask_idx < weights.numel():
                weights[ask_idx] = 1.5 * decay
            if bid_idx < weights.numel():
                weights[bid_idx] = 1.5 * decay

        for depth in range(levels):
            decay = 1.0 / float(depth + 1)
            ask_size_idx = size_start + depth
            bid_size_idx = size_start + levels + depth
            if ask_size_idx < weights.numel():
                weights[ask_size_idx] = 2.0 * decay
            if bid_size_idx < weights.numel():
                weights[bid_size_idx] = 2.0 * decay
        return weights.repeat(self._prediction_horizon())

    def _oracle_local_error_proxy(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        *,
        hist: torch.Tensor,
        conditioning_cache: ConditioningCache,
        dt: float,
        t_cur: float,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x_euler = x + dt * v
        x_half = x + 0.5 * dt * v
        t_mid = torch.full(
            (batch_size, 1),
            t_cur + 0.5 * dt,
            device=x.device,
            dtype=x.dtype,
        )
        v_mid = self._sample_field(
            x_half,
            t_mid,
            hist,
            conditioning_cache=conditioning_cache,
        )
        x_two_half = x_half + 0.5 * dt * v_mid
        return torch.sqrt(
            (x_euler - x_two_half).reshape(batch_size, -1).square().sum(dim=-1) + 1e-12
        )

    def _sample_impl(
        self,
        hist: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
        steps: Optional[int],
        solver: Optional[str],
        record_trace: bool,
        oracle_local_error: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if hist.ndim != 3:
            raise ValueError(
                f"hist must have shape [batch, history, context_dim], got {tuple(hist.shape)}."
            )
        if not hist.is_floating_point():
            raise TypeError("hist must be a floating-point tensor.")
        if oracle_local_error and not record_trace:
            raise ValueError("oracle_local_error requires record_trace=True.")

        configured_step_count = int(self.cfg.sample.steps)
        n_steps = int(configured_step_count if steps is None else steps)
        if n_steps <= 0:
            source_name = "sample.steps" if steps is None else "steps"
            raise ValueError(f"{source_name} must be positive, got {n_steps}.")

        solver_name = self._resolve_solver_name(solver)
        time_grid = self._resolved_time_grid(n_steps)
        batch_size = hist.shape[0]
        state_dim = self._sample_state_dim()
        x = torch.randn(
            batch_size,
            state_dim,
            device=hist.device,
            dtype=hist.dtype,
        )
        conditioning_cache = self.backbone.precompute(hist, cond=cond)

        prev_u: Optional[torch.Tensor] = None
        prev_t: Optional[float] = None
        ema_beta = 0.9

        if record_trace:
            ema_v: Optional[torch.Tensor] = None
            ema_v_sq: Optional[torch.Tensor] = None
            ema_u: Optional[torch.Tensor] = None
            top_book_weights = self._top_of_book_feature_weights(
                device=hist.device,
                dtype=x.dtype,
            )[None, :]
            trace_disagreement = []
            trace_velocity_norm = []
            trace_residual_norm = []
            trace_hybrid_signal = []
            trace_u_disagreement = []
            trace_u_residual_norm = []
            trace_u_hybrid_signal = []
            trace_variance_scaled_signal = []
            trace_top_book_disagreement = []
            trace_top_book_residual_norm = []
            trace_top_book_hybrid_signal = []
            trace_oracle_error = []
            trace_field_evals = []
            trace_time = []

        def field_at(x_state: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t_tensor = torch.full(
                (batch_size, 1),
                float(t_scalar),
                device=hist.device,
                dtype=x.dtype,
            )
            return self._sample_field(
                x_state,
                t_tensor,
                hist,
                conditioning_cache=conditioning_cache,
            )

        for i in range(n_steps):
            t_cur = float(time_grid[i])
            t_next = float(time_grid[i + 1])
            dt = float(t_next - t_cur)
            v = field_at(x, t_cur)
            tail_cur = max(1e-12, 1.0 - t_cur)
            u_state = x + tail_cur * v if (record_trace or solver_name == "dpmpp2m") else None

            if record_trace:
                v_flat = v.reshape(batch_size, -1)
                if ema_v is None:
                    ema_v = v_flat.detach().clone()
                if ema_v_sq is None:
                    ema_v_sq = v_flat.detach().square().clone()
                assert u_state is not None
                u_flat = u_state.reshape(batch_size, -1)
                if ema_u is None:
                    ema_u = u_flat.detach().clone()

                velocity_norm = torch.sqrt(v_flat.square().sum(dim=-1) + 1e-12)
                disagreement = 1.0 - F.cosine_similarity(
                    v_flat,
                    ema_v,
                    dim=-1,
                    eps=1e-8,
                ).clamp(-1.0, 1.0)
                residual_flat = v_flat - ema_v
                residual_norm = torch.sqrt(residual_flat.square().sum(dim=-1) + 1e-12)
                hybrid_signal = residual_norm * disagreement

                feature_var = torch.clamp(ema_v_sq - ema_v.square(), min=0.0)
                variance_scale = torch.sqrt(feature_var + 1e-6)
                variance_scaled_disagreement = 1.0 - F.cosine_similarity(
                    v_flat / variance_scale,
                    ema_v / variance_scale,
                    dim=-1,
                    eps=1e-8,
                ).clamp(-1.0, 1.0)
                variance_scaled_residual_norm = torch.sqrt(
                    (residual_flat / variance_scale).square().sum(dim=-1) + 1e-12
                )
                variance_scaled_signal = (
                    variance_scaled_residual_norm * variance_scaled_disagreement
                )

                weighted_v_flat = v_flat * top_book_weights
                weighted_ema_flat = ema_v * top_book_weights
                top_book_disagreement = 1.0 - F.cosine_similarity(
                    weighted_v_flat,
                    weighted_ema_flat,
                    dim=-1,
                    eps=1e-8,
                ).clamp(-1.0, 1.0)
                top_book_residual_flat = weighted_v_flat - weighted_ema_flat
                top_book_residual_norm = torch.sqrt(
                    top_book_residual_flat.square().sum(dim=-1) + 1e-12
                )
                top_book_hybrid_signal = top_book_residual_norm * top_book_disagreement

                u_disagreement = 1.0 - F.cosine_similarity(
                    u_flat,
                    ema_u,
                    dim=-1,
                    eps=1e-8,
                ).clamp(-1.0, 1.0)
                u_residual_flat = u_flat - ema_u
                u_residual_norm = torch.sqrt(u_residual_flat.square().sum(dim=-1) + 1e-12)
                u_hybrid_signal = u_residual_norm * u_disagreement

                oracle_error = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
                if oracle_local_error:
                    oracle_error = self._oracle_local_error_proxy(
                        x,
                        v,
                        hist=hist,
                        conditioning_cache=conditioning_cache,
                        dt=dt,
                        t_cur=t_cur,
                    )

            field_eval_count = 1
            if solver_name == "heun":
                x_pred = x + dt * v
                v_next = field_at(x_pred, t_next)
                x = x + dt * 0.5 * (v + v_next)
                field_eval_count = 2
            elif solver_name == "midpoint_rk2":
                x_mid = x + 0.5 * dt * v
                v_mid = field_at(x_mid, t_cur + 0.5 * dt)
                x = x + dt * v_mid
                field_eval_count = 2
            elif solver_name == "euler":
                x = x + dt * v
            else:
                assert solver_name == "dpmpp2m"
                assert u_state is not None
                tail_next = max(0.0, 1.0 - t_next)
                if prev_u is None or prev_t is None:
                    coeff_x = 0.0 if tail_next <= 1e-12 else tail_next / tail_cur
                    x = coeff_x * x + (1.0 - coeff_x) * u_state
                else:
                    prev_dt = max(t_cur - prev_t, 1e-12)
                    slope = (u_state - prev_u) / prev_dt
                    coeff_x = 0.0 if tail_next <= 1e-12 else tail_next / tail_cur
                    coeff_u = 1.0 - coeff_x
                    if tail_next <= 1e-12:
                        corr_coeff = tail_cur
                    else:
                        ratio = tail_next / tail_cur
                        corr_coeff = tail_next * ((1.0 / ratio) - 1.0 + math.log(ratio))
                    x = coeff_x * x + coeff_u * u_state + corr_coeff * slope
                prev_u = u_state
                prev_t = t_cur

            if record_trace:
                ema_v = ema_beta * ema_v + (1.0 - ema_beta) * v_flat.detach()
                ema_v_sq = ema_beta * ema_v_sq + (1.0 - ema_beta) * v_flat.detach().square()
                ema_u = ema_beta * ema_u + (1.0 - ema_beta) * u_flat.detach()

                trace_disagreement.append(disagreement.detach().cpu())
                trace_velocity_norm.append(velocity_norm.detach().cpu())
                trace_residual_norm.append(residual_norm.detach().cpu())
                trace_hybrid_signal.append(hybrid_signal.detach().cpu())
                trace_u_disagreement.append(u_disagreement.detach().cpu())
                trace_u_residual_norm.append(u_residual_norm.detach().cpu())
                trace_u_hybrid_signal.append(u_hybrid_signal.detach().cpu())
                trace_variance_scaled_signal.append(variance_scaled_signal.detach().cpu())
                trace_top_book_disagreement.append(top_book_disagreement.detach().cpu())
                trace_top_book_residual_norm.append(top_book_residual_norm.detach().cpu())
                trace_top_book_hybrid_signal.append(top_book_hybrid_signal.detach().cpu())
                trace_oracle_error.append(oracle_error.detach().cpu())
                trace_field_evals.append(
                    torch.full(
                        (batch_size,),
                        float(field_eval_count),
                        device="cpu",
                        dtype=x.dtype,
                    )
                )
                trace_time.append(t_cur)

        if not record_trace:
            return x, None

        disagreement_t = torch.stack(trace_disagreement, dim=1)
        velocity_norm_t = torch.stack(trace_velocity_norm, dim=1)
        residual_norm_t = torch.stack(trace_residual_norm, dim=1)
        hybrid_signal_t = torch.stack(trace_hybrid_signal, dim=1)
        u_disagreement_t = torch.stack(trace_u_disagreement, dim=1)
        u_residual_norm_t = torch.stack(trace_u_residual_norm, dim=1)
        u_hybrid_signal_t = torch.stack(trace_u_hybrid_signal, dim=1)
        variance_scaled_signal_t = torch.stack(trace_variance_scaled_signal, dim=1)
        top_book_disagreement_t = torch.stack(trace_top_book_disagreement, dim=1)
        top_book_residual_norm_t = torch.stack(trace_top_book_residual_norm, dim=1)
        top_book_hybrid_signal_t = torch.stack(trace_top_book_hybrid_signal, dim=1)
        oracle_error_t = torch.stack(trace_oracle_error, dim=1)
        field_evals_t = torch.stack(trace_field_evals, dim=1)
        trace: Dict[str, Any] = {
            "solver": solver_name,
            "steps": n_steps,
            "step_index": torch.arange(n_steps, dtype=torch.long),
            "time": torch.tensor(trace_time, dtype=x.dtype),
            "time_grid": torch.tensor(time_grid, dtype=x.dtype),
            "disagreement": disagreement_t,
            "velocity_norm": velocity_norm_t,
            "residual_norm": residual_norm_t,
            "hybrid_signal": hybrid_signal_t,
            "u_disagreement": u_disagreement_t,
            "u_residual_norm": u_residual_norm_t,
            "u_hybrid_signal": u_hybrid_signal_t,
            "variance_scaled_signal": variance_scaled_signal_t,
            "top_book_disagreement": top_book_disagreement_t,
            "top_book_residual_norm": top_book_residual_norm_t,
            "top_book_hybrid_signal": top_book_hybrid_signal_t,
            "oracle_local_error": oracle_error_t,
            "field_evals_by_step": field_evals_t,
            "mean_field_evals_per_step": float(field_evals_t.mean().item()),
            "mean_total_field_evals_per_rollout": float(field_evals_t.sum(dim=1).mean().item()),
        }
        return x, trace

    @torch.no_grad()
    def sample_trace(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        solver: Optional[str] = None,
        oracle_local_error: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample one next state and return per-solver-step trace statistics."""
        if self._is_non_autoregressive():
            raise RuntimeError("Non-autoregressive OTFlow uses sample_future_trace(...).")
        x, trace = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            solver=solver,
            record_trace=True,
            oracle_local_error=oracle_local_error,
        )
        assert trace is not None
        return x, trace

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        if self._is_non_autoregressive():
            raise RuntimeError("Non-autoregressive OTFlow uses sample_future(...).")
        x, _ = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            solver=solver,
            record_trace=False,
            oracle_local_error=False,
        )
        return x

    @torch.no_grad()
    def sample_future_trace(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        solver: Optional[str] = None,
        oracle_local_error: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, trace = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            solver=solver,
            record_trace=True,
            oracle_local_error=oracle_local_error,
        )
        assert trace is not None
        return self._reshape_sample_block(x), trace

    @torch.no_grad()
    def sample_future(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        x, _ = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            solver=solver,
            record_trace=False,
            oracle_local_error=False,
        )
        return self._reshape_sample_block(x)


__all__ = ["OTFlow"]
