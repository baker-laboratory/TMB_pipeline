"""Checkpoint discovery regressions; these tests use only the standard library."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "checkpoint_utils", ROOT / "TMB_MPNN/TMB_MPNN_code/v2.1.1/checkpoint_utils.py"
)
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)


class CheckpointPathTests(unittest.TestCase):
    def test_default_is_repo_relative_from_any_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            try:
                os.chdir(directory)
                self.assertEqual(
                    Path(checkpoints.default_sc_checkpoint()),
                    ROOT / "weights/ligandmpnn_sc_v_32_002_16.pt",
                )
            finally:
                os.chdir(original_cwd)

    def test_custom_weights_directory(self):
        with patch.dict(os.environ, {"WEIGHTS_DIR": "/custom weights"}, clear=True):
            self.assertEqual(
                checkpoints.default_sc_checkpoint(),
                "/custom weights/ligandmpnn_sc_v_32_002_16.pt",
            )

    def test_checkpoint_environment_override_wins(self):
        with patch.dict(os.environ, {
            "WEIGHTS_DIR": "/other",
            "TMB_MPNN_SC_CKPT_PATH": "/custom/packer.pt",
        }, clear=True):
            self.assertEqual(checkpoints.default_sc_checkpoint(), "/custom/packer.pt")

    def test_empty_environment_values_use_repo_default(self):
        with patch.dict(os.environ, {
            "WEIGHTS_DIR": "", "TMB_MPNN_SC_CKPT_PATH": "",
        }, clear=True):
            self.assertEqual(
                Path(checkpoints.default_sc_checkpoint()).parent, ROOT / "weights"
            )

    def test_explicit_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "custom.pt"
            checkpoint.write_bytes(b"checkpoint presence test")
            with patch.dict(os.environ, {"TMB_MPNN_SC_CKPT_PATH": "/missing/other.pt"}):
                self.assertEqual(checkpoints.require_sc_checkpoint(checkpoint), str(checkpoint))

    def test_missing_empty_and_directory_paths_explain_how_to_download(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.pt"
            empty.touch()
            for path in [Path(directory) / "missing.pt", empty, Path(directory)]:
                with self.subTest(path=path), self.assertRaises(FileNotFoundError) as error:
                    checkpoints.require_sc_checkpoint(path)
                self.assertIn(str(path), str(error.exception))
                self.assertIn("download_assets.sh", str(error.exception))
                self.assertIn("--checkpoint_path_sc", str(error.exception))


if __name__ == "__main__":
    unittest.main()
