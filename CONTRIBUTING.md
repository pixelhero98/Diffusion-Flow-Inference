# Contributing

This repository is source-only. Do not commit datasets, generated outputs,
checkpoints, run logs, local caches, virtual environments, or machine-specific
paths.

Before opening a change, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

Keep compatibility branches and historical result-reuse logic out of active
code paths. Current checkpoints, manifests, and result rows should fail clearly
when their protocol, schedule, conditioning, or dataset metadata do not match.
