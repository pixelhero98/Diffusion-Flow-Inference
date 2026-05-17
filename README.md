
# Diffusion-Flow-Inference

Diffusion-Flow-Inference evaluates optimized diffusion schedules after mapping them onto normalized flow time for fixed OTFlow backbones. The active schedules are `uniform`, `late_power_3`, `ays`, `gits`, and `ots`; the transferred diffusion schedules are `ays`, `gits`, and `ots`.

## Active Code Path

- `code/diffusion_flow_time_reparameterization.py`: fixed-schedule evaluation entrypoint.
- `code/diffusion_flow_schedules.py`: schedule construction for uniform, late-power-3, AYS, GITS, and OTS.
- `code/otflow_evaluation_support.py`: checkpoint loading, dataset split resolution, solver mappings, and metric helpers.
- `code/otflow_paper_registry.py`: method, schedule, and solver registry.
- `code/otflow_model.py`, `code/conditioning.py`, `code/config.py`, `code/modules.py`, and `code/otflow_train_val.py`: OTFlow backbone model and training/evaluation utilities.

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

Pip:

```bash
python -m pip install -r requirements.txt
```

Conda:

```bash
conda env create -f environment.conda.yml
```

Raw medical dataset preparation requires `OTFLOW_MEDICAL_STAGING_ROOT` to point at the local staging directory. Prepared dataset evaluation uses the processed files in `data/`.

## CPU Smoke Checks

```bash
cd code
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test_*.py'
```

Dry-run prep from the repository root accepts project-relative paths:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python code/diffusion_flow_time_reparameterization.py --forecast_datasets '' --lob_datasets '' --backbone_manifest outputs/backbone_matrix/backbone_manifest.json
```
