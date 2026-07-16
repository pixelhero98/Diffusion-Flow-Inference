# Contributing

This repository is source-only. Do not commit datasets, generated outputs,
checkpoints, run logs, local caches, virtual environments, or machine-specific
paths.

Before opening a change, run:

```bash
python -m pip install -e ".[test]"
ruff check .
ruff format --check .
python -m compileall -q src tests scripts
python -m pytest -q
python -m build
python -m pip check
```

Keep compatibility branches and historical result-reuse logic out of active
code paths. Current checkpoints, manifests, and result rows should fail clearly
when their protocol, schedule, conditioning, or dataset metadata do not match.

Use `ruff format .` for repository-wide formatting. Keep public configuration,
paths, and generated manifests portable: do not add machine-specific defaults or
serialize absolute local paths.
