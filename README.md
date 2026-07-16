# Diffusion-Flow-Inference

Diffusion-Flow-Inference is a source-first toolkit for evaluating fixed ODE solver
schedules after mapping them to normalized flow time. It provides reusable OTFlow
models, data adapters, schedule construction, evaluation support, and diagnostic
figure builders.

The built-in schedules are `uniform`, `late_power_3`, `flowts_power_sampling`,
`ays`, `gits`, and `ots`. The last three are transferred diffusion schedules.

## Installation

Python 3.11 or newer is required. Install the core package in editable mode:

```bash
python -m pip install -e .
```

Optional dependencies are grouped by use case:

```bash
python -m pip install -e ".[plot]"       # diagnostic figures
python -m pip install -e ".[medical]"    # Sleep-EDF preparation
python -m pip install -e ".[test]"       # development and validation
```

`requirements.txt` installs the plotting and medical-data extras for convenience.
Conda users can create the environment and then install the package without
re-resolving its dependencies:

```bash
conda env create -f environment.conda.yml
conda activate diffusion-flow-inference
python -m pip install -e . --no-deps
```

## Project layout

- `src/diffusion_flow_inference/data/`: dataset definitions, preparation, and paths
- `src/diffusion_flow_inference/models/`: configuration, conditioning, OTFlow, and metrics
- `src/diffusion_flow_inference/schedule_transfer/`: schedule grids and diagnostics
- `src/diffusion_flow_inference/evaluation/`: artifact loading and evaluation support
- `src/diffusion_flow_inference/visualization/`: diagnostic figure builders
- `scripts/`: thin wrappers for the installed command-line tools

## Data and model artifacts

The repository does not distribute datasets, prepared arrays, model checkpoints, or
generated results. By default, relative paths are resolved from the current working
directory. Set `DFI_PROJECT_ROOT` when commands are launched elsewhere:

```bash
export DFI_PROJECT_ROOT=/path/to/your/workspace
```

Forecast dataset manifests and backbone manifests store relative paths so their
directories can be moved together. The default backbone manifest location is
`outputs/backbone_matrix/backbone_manifest.json`; loading rejects unsupported schemas,
inconsistent counts, and missing required fields.

Sleep-EDF evaluation is read-only and requires an explicitly prepared `.npz` file plus
its metadata. Raw preparation is a separate step: set `DFI_MEDICAL_STAGING_ROOT`, then
call `prepare_sleep_edf_dataset(...)` from
`diffusion_flow_inference.data.otflow_medical_datasets`.

## Usage

Inspect the installed interfaces with:

```bash
dfi-run-schedules --help
dfi-build-velocity-variation-figure --help
dfi-build-ptg-figure --help
```

Without `--allow_execute`, the schedule runner summarizes its requested setup without
loading models or running evaluations. For example:

```bash
dfi-run-schedules \
  --forecast_datasets '' \
  --conditional_generation_datasets '' \
  --schedule-names uniform,ays
```

Actual evaluation additionally requires `--allow_execute` and the requested datasets,
manifest, and checkpoint artifacts. Results default to
`outputs/diffusion_flow_time_reparameterization/` and can be redirected with
`--out_root`.

Schedule grids are also available as a small Python API:

```python
from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import (
    build_schedule_grid,
)

time_grid = build_schedule_grid("ays", n_steps=10)
```

## Validation

The local and CI checks are:

```bash
ruff check .
ruff format --check .
python -m compileall -q src tests scripts
python -m pytest -q
python -m build
python -m pip check
```

See `CONTRIBUTING.md` for contribution and portability guidelines.
