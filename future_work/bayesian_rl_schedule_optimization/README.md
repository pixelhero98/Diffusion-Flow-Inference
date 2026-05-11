# Bayesian And RL Schedule Optimization

This sandbox is for exploring Bayesian optimization and reinforcement learning methods for inference schedule search.

The v1 workflow is offline and source-only:

- Build a numerical reference schedule from an existing PTG payload.
- Prefer SER/PTG local-defect traces; fall back to Info-growth traces only when local-defect data is unavailable.
- Generate KL-banded Sobol/random perturbations around the reference.
- Suggest closed-loop noisy BO batches with BoTorch `qLogNoisyExpectedImprovement` after observations are available.
- For forecast extrapolation, optimize the averaged relative error against uniform,
  `0.5 * (CRPS / CRPS_uniform + MASE / MASE_uniform)`. Lower is better; BO maximizes
  its negative after the KL penalty.
- Evaluate the uniform schedule once per optimization session, store that baseline at the top level, then append
  candidate CRPS/MASE after each downstream evaluation round.
- For checkpoint-backed forecast BO, split the existing validation set deterministically into a calibration pool
  and a BO-validation pool. Build the SER/PTG reference from calibration windows, tune BO on fixed BO-validation
  windows, then re-score the top candidates on the full BO-validation pool before locked-test comparison.
- Reuse locked-test `uniform` and `GITS` rows from prior runs when metadata and schedule hashes match. Reuse
  `ser_ptg_reference` only when the generated reference schedule hash is identical.
- Configure locked-test comparison schedules with `--comparison-schedules`. Deterministic schedules
  `uniform`, `ays`, `gits`, and `ots` are cache-reusable when metadata and schedule hashes match.
- Generate read-only BO trajectory and schedule-placement figures from completed run artifacts with
  `visualize-run`; this only writes `figures/` and `tables/` under the selected run root.
- Run V1 bandit KL-PPO schedule search with `run-forecast-ppo-bandit`; this reuses or builds BO
  warm-start artifacts, trains one diagonal Gaussian policy per `{dataset, solver, target_nfe}` cell,
  and logs smoothness/min-step diagnostics without penalizing them in the V1 reward.

The existing production package under `src/diffusion_flow_inference/` must not import from this folder. Promote reusable pieces only after they are reviewed and tested against the main inference workflow.

Install optional BO dependencies only when running `suggest-bo-batch`:

```bash
python -m pip install -r future_work/bayesian_rl_schedule_optimization/requirements.txt
```

Example commands:

```bash
python future_work/bayesian_rl_schedule_optimization/code/cli.py build-reference \
  --ptg-json outputs/ptg_observed_gain/ptg_observed_gain_inputs.json \
  --out-json outputs/future_work/bo_reference_electricity_euler_10.json \
  --dataset electricity \
  --solver euler \
  --target-nfe 10

python future_work/bayesian_rl_schedule_optimization/code/cli.py generate-perturbations \
  --reference-json outputs/future_work/bo_reference_electricity_euler_10.json \
  --out-json outputs/future_work/bo_initial_electricity_euler_10.json

python future_work/bayesian_rl_schedule_optimization/code/cli.py suggest-bo-batch \
  --reference-json outputs/future_work/bo_reference_electricity_euler_10.json \
  --observations-json outputs/future_work/bo_observations_electricity_euler_10.json \
  --out-json outputs/future_work/bo_batch_electricity_euler_10.json

python future_work/bayesian_rl_schedule_optimization/code/cli.py visualize-run \
  --run-root outputs/future_work/bo_schedule_search/sf_traffic_nfe10_bo100_20k_cal70_val30_ref16_basis5 \
  --top-k 5

python future_work/bayesian_rl_schedule_optimization/code/cli.py run-forecast-ppo-bandit \
  --workspace-root /home/yzn/work/Diffusion-Flow-Inference \
  --datasets san_francisco_traffic,solar_energy_10m \
  --target-nfes 10,12,16 \
  --solvers euler,heun,midpoint_rk2,dpmpp2m \
  --otflow-train-steps 20000 \
  --ppo-batch-size 8 \
  --ppo-updates 20 \
  --calibration-fraction 0.7 \
  --calibration-windows 64 \
  --selector-windows 0 \
  --num-eval-samples 5 \
  --reference-macro-factor 16 \
  --final-test-seeds 0,1,2 \
  --out-root outputs/future_work/kl_ppo_bandit/sf_solar_nfe10_12_16_20k_cal70_val30_v1

python future_work/bayesian_rl_schedule_optimization/code/cli.py run-forecast-bo \
  --workspace-root /home/yzn/work/Diffusion-Flow-Inference \
  --dataset san_francisco_traffic \
  --target-nfe 10 \
  --solvers euler,dpmpp2m \
  --otflow-train-steps 20000 \
  --bo-budget 100 \
  --calibration-fraction 0.7 \
  --calibration-windows 64 \
  --bo-val-windows 64 \
  --confirm-top-k 5 \
  --confirm-val-windows 0 \
  --n-initial 16 \
  --num-eval-samples 5 \
  --reference-macro-factor 16 \
  --final-test-seeds 0,1,2 \
  --comparison-schedules uniform,gits,ser_ptg_reference,bo_best \
  --baseline-cache-roots outputs/future_work/bo_schedule_search/sf_traffic_nfe10_bo100_20k_val64_s5_ref16 \
  --out-root outputs/future_work/bo_schedule_search/sf_traffic_nfe10_bo100_20k_cal70_val30_ref16_basis5
```

Solar transfer-schedule example:

```bash
python future_work/bayesian_rl_schedule_optimization/code/cli.py run-forecast-bo \
  --workspace-root /home/yzn/work/Diffusion-Flow-Inference \
  --dataset solar_energy_10m \
  --target-nfe 10 \
  --solvers euler,heun,midpoint_rk2,dpmpp2m \
  --otflow-train-steps 20000 \
  --bo-budget 100 \
  --calibration-fraction 0.7 \
  --calibration-windows 64 \
  --bo-val-windows 64 \
  --confirm-top-k 5 \
  --confirm-val-windows 0 \
  --n-initial 16 \
  --num-eval-samples 5 \
  --reference-macro-factor 16 \
  --final-test-seeds 0,1,2 \
  --comparison-schedules uniform,ays,gits,ots,ser_ptg_reference,bo_best \
  --out-root outputs/future_work/bo_schedule_search/solar_nfe10_bo100_20k_cal70_val30_ref16_basis5_transfer3
```

Forecast observations should keep the uniform baseline fixed for the session:

```json
{
  "uniform_baseline": {
    "crps": 4.0,
    "mase": 2.0
  },
  "observations": [
    {
      "theta": [0.0, 0.0, 0.0, 0.0],
      "crps": 3.8,
      "mase": 1.9
    }
  ]
}
```

The code stores `relative_crps_ratio`, `relative_mase_ratio`, `metric_val`, and
`objective_value = -metric_val - lambda_kl * KL(q || q_ref)`. Existing rows with a
precomputed `metric_val` are still accepted for non-forecast experiments. Legacy forecast rows
that repeat `uniform_crps` and `uniform_mase` are also accepted, but if a session-level
`uniform_baseline` is present, row-level baseline values must match it.

The default residual basis is now 5D: early/late tilt, quadratic curvature, and broad local
bumps centered near 0.25, 0.50, and 0.75. Four-dimensional legacy observations still load with
the older two-bump basis so previous run artifacts remain inspectable.
