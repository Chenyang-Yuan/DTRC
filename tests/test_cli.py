"""Exercise the training entry point on small synthetic trajectories."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dtrc.agent import VARIANTS
from fixtures import SMALL, make_cache


class TrainingCLITests(unittest.TestCase):
    def run_training(self, cache, output, variant, steps=3, resume=None):
        args = [
            sys.executable,
            "-m",
            "dtrc.cli.train",
            "--latent-cache",
            str(cache),
            "--out",
            str(output),
            "--variant",
            variant,
            "--device",
            "cpu",
            "--train-steps",
            str(steps),
            "--batch-size",
            "4",
            "--span",
            "4",
            "--log-every",
            "1",
            "--save-every",
            "1",
            "--holdout-every",
            "0",
        ]
        for key, value in SMALL.items():
            if key == "span" or (variant == "symmetric" and key == "asym_dim"):
                continue
            args.extend(["--override", f"{key}={json.dumps(value)}"])
        if resume is not None:
            args.extend(["--resume", str(resume)])
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        return subprocess.run(args, capture_output=True, text=True, env=env, timeout=90)

    def test_all_variants_train_and_save(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache = root / "cache"
            make_cache(cache)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    output = root / variant
                    result = self.run_training(cache, output, variant)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    state = torch.load(
                        output / "checkpoints/final.pt", weights_only=False
                    )
                    self.assertEqual(state["step"], 3)
                    self.assertEqual(state["buffer_meta"]["completed_steps"], 3)
                    self.assertTrue((output / "DTRC_TRAIN_COMPLETE").is_file())

    def test_resume_preserves_step_count_and_rejects_completed_target(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache, output = root / "cache", root / "run"
            make_cache(cache)
            result = self.run_training(cache, output, "temporal")
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint = output / "checkpoints/final.pt"
            result = self.run_training(
                cache, output, "temporal", steps=4, resume=checkpoint
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            before = checkpoint.read_bytes()
            result = self.run_training(
                cache, output, "temporal", steps=3, resume=checkpoint
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must exceed the resumed step", result.stderr)
            self.assertEqual(before, checkpoint.read_bytes())


if __name__ == "__main__":
    unittest.main()
