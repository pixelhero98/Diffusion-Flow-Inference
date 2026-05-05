
# Diffusion-Flow-Inference

Diffusion-Flow-Inference evaluates optimized diffusion schedules after mapping them onto normalized flow time for fixed OTFlow backbones. The active schedules are `uniform`, `late_power_3`, `ays`, `gits`, and `ots`; the transferred diffusion schedules are `ays`, `gits`, and `ots`.

## Package Layout

- `diffusion_flow_inference.datasets`: dataset settings, loaders, audits, and experiment plans.
- `diffusion_flow_inference.backbones.settings`: backbone configs, modules, model definitions, and baseline architectures.
- `diffusion_flow_inference.backbones.training`: backbone training, benchmarking, readiness audits, and manifest registry.
- `diffusion_flow_inference.schedules`: diffusion-to-flow schedule construction and schedule registry.
- `diffusion_flow_inference.solvers`: solver runtime settings.
- `diffusion_flow_inference.evaluation`: fixed-schedule evaluation, metric support, and paper table helpers.
- `diffusion_flow_inference.diagnostics`: PTG, adaptive-solver, hardness, and signal-trace diagnostics.

## Data, Outputs, And Backbones

This repository is source-only. Large datasets, generated outputs, trained backbone artifacts, and local virtual environments are intentionally not committed.

A reduced public processed-data bundle for review is available on Hugging Face:

```text
https://huggingface.co/datasets/pixelhero98/d2f-dataset
```

The Hugging Face bundle includes the cryptos processed file and processed Monash manifests, audits, and `.tsf` sources. It excludes ES MBP-10, Sleep-EDF, raw Monash download zips, and other restricted or machine-local artifacts.

Generated outputs default to:

```text
outputs/
```

Backbones are trained or retrained locally with the package training workflow, then evaluated using the generated artifacts under `outputs/`.

## Environment

Pip:

```bash
python -m pip install -e .
```

Conda:

```bash
conda env create -f environment.conda.yml
```

Raw medical dataset preparation requires `OTFLOW_MEDICAL_STAGING_ROOT` to point at the local staging directory. For the public review subset, download the processed bundle from Hugging Face and pass it to the evaluation runner.

## CPU Fast Checks

Run the public, artifact-independent smoke tests with your active Python environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q
```

Download the public review bundle from Hugging Face with your active Python environment. If needed, install the lightweight Hub client with `python -m pip install huggingface_hub`.

```python
from huggingface_hub import hf_hub_download

bundle = hf_hub_download(
    repo_id="pixelhero98/d2f-dataset",
    filename="d2f_review_processed_datasets_public.zip",
    repo_type="dataset",
)
```

Pass the downloaded zip path to the evaluation runner and let it extract processed inputs under `outputs/`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization \
  --dataset_bundle_zip /path/to/d2f_review_processed_datasets_public.zip \
  --dataset_bundle_mode auto \
  --dataset_bundle_extract_root outputs/dataset_bundles/extracted
```

## License

MIT. See `LICENSE`.
