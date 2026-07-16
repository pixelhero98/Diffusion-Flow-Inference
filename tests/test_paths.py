from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diffusion_flow_inference.data.otflow_paths import (
    PROJECT_ROOT_ENV_VAR,
    project_root,
    resolve_project_path,
)


class ProjectPathTests(unittest.TestCase):
    def test_current_working_directory_is_the_default_project_root(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(PROJECT_ROOT_ENV_VAR, None)
                    self.assertEqual(project_root(), Path(temporary_directory).resolve())
                    self.assertEqual(
                        resolve_project_path("outputs/result.json"),
                        Path(temporary_directory, "outputs", "result.json").resolve(),
                    )
            finally:
                os.chdir(original_directory)

    def test_environment_variable_sets_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: temporary_directory}):
                self.assertEqual(project_root(), Path(temporary_directory).resolve())


if __name__ == "__main__":
    unittest.main()
