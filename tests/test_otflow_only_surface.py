from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from diffusion_flow_inference.backbones.training.train_val import SUPPORTED_MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tracked_paths() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "src", "README.md", "pyproject.toml"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = [PROJECT_ROOT / line for line in proc.stdout.splitlines() if line.strip()]
    return [path for path in paths if path.exists()]


class OTFlowOnlySourceSurfaceTests(unittest.TestCase):
    def test_supported_model_names_are_otflow_only(self) -> None:
        self.assertEqual(SUPPORTED_MODEL_NAMES, ("otflow",))

    def test_non_otflow_model_files_are_absent(self) -> None:
        removed = [
            "src/diffusion_flow_inference/backbones/settings/baselines.py",
            "src/diffusion_flow_inference/backbones/settings/deepmarket_baselines.py",
            "src/diffusion_flow_inference/backbones/settings/temporal_baselines.py",
            "src/diffusion_flow_inference/backbones/settings/otflow_baselines.py",
        ]
        for rel in removed:
            self.assertFalse((PROJECT_ROOT / rel).exists(), rel)

    def test_source_docs_do_not_expose_non_otflow_names(self) -> None:
        forbidden = (
            "bi" + "flow",
            "tr" + "ades",
            "c" + "gan",
            "deep" + "market",
            "time" + "gan",
            "time" + "causalvae",
            "ko" + "vae",
            "conditional" + "realnvp",
            "baseline" + "_latent_dim",
            "gan" + "_noise_dim",
            "diffusion" + "_steps",
            "vae" + "_kl_weight",
            "kovae" + "_",
            "timegan" + "_",
            "cgan" + "_",
            "nf" + "config",
        )
        offenders: list[str] = []
        for path in _tracked_paths():
            if path.suffix not in {".py", ".md", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for term in forbidden:
                if term in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {term}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
