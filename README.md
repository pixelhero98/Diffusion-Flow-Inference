# Diffusion-Flow-Inference

Diffusion-Flow-Inference evaluates fixed schedules after mapping them onto normalized flow time for fixed OTFlow backbones. The active schedules are `uniform`, `late_power_3`, `flowts_power_sampling`, `ays`, `gits`, and `ots`; the transferred diffusion schedules are `ays`, `gits`, and `ots`.

## Source Layout

- `src/diffusion_flow_inference/data/`: Monash, LOB, medical dataset definitions, experiment plans, and project paths.
- `src/diffusion_flow_inference/models/`: OTFlow configuration, conditioning, backbone modules, training, and model utilities.
- `src/diffusion_flow_inference/schedule_transfer/`: schedule grids, registries, table helpers, signal traces, and diagnostics.
- `src/diffusion_flow_inference/evaluation/`: checkpoint loading, runner support, solver mappings, and sampling helpers.
- `src/diffusion_flow_inference/visualization/`: fixed-schedule diagnostic figure builders.
- `scripts/`: thin command-line wrappers for the packaged entry points.

## Data, Outputs, And Backbones

This repository is source-only. Local runs may use `data/`, `paper_datasets/`, `outputs/`, and `.venv/`, but those large or machine-local directories are intentionally ignored and are not part of the public source tree.

Generated outputs default to:

```text
outputs/
```

The default backbone manifest path is:

```text
outputs/backbone_matrix/backbone_manifest.json
```

A prepared local backbone matrix should report 40 ready checkpoint artifacts and 0 missing artifacts.

## Environment

Install the package in editable mode:

```bash
python -m pip install -e .
```

Or install runtime dependencies directly:

```bash
python -m pip install -r requirements.txt
```

Conda users can create the environment with:

```bash
conda env create -f environment.conda.yml
```

Raw medical dataset preparation requires `OTFLOW_MEDICAL_STAGING_ROOT` to point at the local staging directory. Prepared dataset evaluation uses the processed files in `data/`.

## CPU Smoke Checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

Dry-run prep from the repository root accepts project-relative paths:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 dfi-run-schedules --forecast_datasets '' --lob_datasets '' --backbone_manifest outputs/backbone_matrix/backbone_manifest.json
```
