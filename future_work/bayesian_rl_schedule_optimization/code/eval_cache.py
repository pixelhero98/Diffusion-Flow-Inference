from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class JsonlEvalCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("cache_key", ""))
                if key:
                    self.rows[key] = dict(row)

    @staticmethod
    def make_key(
        *,
        split_id: str,
        schedule_hash: str,
        seed: int,
        num_eval_samples: int,
        dataset: Optional[str] = None,
        target_nfe: Optional[int] = None,
        solver_key: Optional[str] = None,
        runtime_nfe: Optional[int] = None,
        checkpoint_id: Optional[str] = None,
        split_indices_hash: Optional[str] = None,
        eval_examples: Optional[int] = None,
    ) -> str:
        fields = [
            str(split_id),
            str(schedule_hash),
            str(seed),
            str(num_eval_samples),
        ]
        extras = {
            "dataset": dataset,
            "target_nfe": target_nfe,
            "solver_key": solver_key,
            "runtime_nfe": runtime_nfe,
            "checkpoint_id": checkpoint_id,
            "split_indices_hash": split_indices_hash,
            "eval_examples": eval_examples,
        }
        fields.extend(f"{key}={value}" for key, value in sorted(extras.items()) if value is not None)
        return "|".join(fields)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.rows.get(str(key))
        return None if row is None else dict(row)

    def put(self, key: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        row = {"cache_key": str(key), **dict(metrics)}
        self.rows[str(key)] = dict(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return dict(row)
