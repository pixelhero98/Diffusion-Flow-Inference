from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from botorch_bo import suggest_bo_batch
from baseline_precompute import add_precompute_forecast_baselines_parser, run_precompute_forecast_baselines
from forecast_bo_runner import add_run_forecast_bo_parser, run_forecast_bo
from joint_progression_ppo_schedule import add_run_forecast_joint_progression_ppo_parser, run_forecast_joint_progression_ppo
from reference_schedule import build_reference_from_payload, load_json, write_json
from residual_parameterization import generate_initial_perturbations
from visualize_bo_runs import add_visualize_run_parser, run_visualize_command


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Future-work Bayesian/RL schedule-search utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_ref = subparsers.add_parser("build-reference", help="Build a numerical reference schedule from a PTG payload.")
    build_ref.add_argument("--ptg-json", type=Path, required=True)
    build_ref.add_argument("--out-json", type=Path, required=True)
    build_ref.add_argument("--dataset", type=str, required=True)
    build_ref.add_argument("--solver", type=str, required=True)
    build_ref.add_argument("--target-nfe", type=int, required=True)
    build_ref.add_argument("--density-floor-eta", type=float, default=0.05)
    build_ref.add_argument("--defect-eps", type=float, default=1e-12)

    perturb = subparsers.add_parser("generate-perturbations", help="Generate Sobol/random KL-banded perturbations.")
    perturb.add_argument("--reference-json", type=Path, required=True)
    perturb.add_argument("--out-json", type=Path, required=True)
    perturb.add_argument("--n-initial", type=int, default=16)
    perturb.add_argument("--seed", type=int, default=0)
    perturb.add_argument("--random-only", action="store_true", help="Use deterministic random directions instead of Torch Sobol.")

    suggest = subparsers.add_parser("suggest-bo-batch", help="Suggest a candidate batch with BoTorch qLogNEI.")
    suggest.add_argument("--reference-json", type=Path, required=True)
    suggest.add_argument("--observations-json", type=Path, required=True)
    suggest.add_argument("--out-json", type=Path, required=True)
    suggest.add_argument("--batch-size", type=int, default=2)
    suggest.add_argument("--lambda-kl", type=float, default=0.05)
    suggest.add_argument("--theta-bound", type=float, default=3.0)
    suggest.add_argument("--raw-samples", type=int, default=128)
    suggest.add_argument("--num-restarts", type=int, default=16)
    suggest.add_argument("--mc-samples", type=int, default=128)
    suggest.add_argument("--seed", type=int, default=0)

    add_run_forecast_bo_parser(subparsers)
    add_run_forecast_joint_progression_ppo_parser(subparsers)
    add_precompute_forecast_baselines_parser(subparsers)
    add_visualize_run_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.command == "build-reference":
        payload = load_json(args.ptg_json)
        out = build_reference_from_payload(
            payload,
            dataset=str(args.dataset),
            solver=str(args.solver),
            target_nfe=int(args.target_nfe),
            eta=float(args.density_floor_eta),
            eps=float(args.defect_eps),
        )
        write_json(out, args.out_json)
        print(str(Path(args.out_json).resolve()))
        return
    if args.command == "generate-perturbations":
        reference = load_json(args.reference_json)
        out = generate_initial_perturbations(
            reference,
            n_initial=int(args.n_initial),
            seed=int(args.seed),
            use_sobol=not bool(args.random_only),
        )
        write_json(out, args.out_json)
        print(str(Path(args.out_json).resolve()))
        return
    if args.command == "suggest-bo-batch":
        reference = load_json(args.reference_json)
        observations = load_json(args.observations_json)
        out = suggest_bo_batch(
            reference,
            observations,
            batch_size=int(args.batch_size),
            lambda_kl=float(args.lambda_kl),
            theta_bound=float(args.theta_bound),
            raw_samples=int(args.raw_samples),
            num_restarts=int(args.num_restarts),
            n_mc_samples=int(args.mc_samples),
            seed=int(args.seed),
        )
        write_json(out, args.out_json)
        print(str(Path(args.out_json).resolve()))
        return
    if args.command == "run-forecast-bo":
        run_forecast_bo(args)
        print(str(Path(args.out_root).resolve()))
        return
    if args.command == "run-forecast-joint-progression-ppo":
        run_forecast_joint_progression_ppo(args)
        print(str(Path(args.out_root).resolve()))
        return
    if args.command == "precompute-forecast-baselines":
        run_precompute_forecast_baselines(args)
        print(str(Path(args.out_root).resolve()))
        return
    if args.command == "visualize-run":
        out = run_visualize_command(args)
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
