"""Episode indexing, image preprocessing, and evaluation protocol checks."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dtrc.backends.lewm import preprocess_uint8
from dtrc.environments.ogbench import episode_layout, load_manifest, sample_problems
from dtrc.metrics import trajectory_metrics


class ProtocolTests(unittest.TestCase):
    def test_preprocess_shape_and_dtype(self):
        pixels = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        output = preprocess_uint8(pixels)
        self.assertEqual(output.shape, (2, 3, 224, 224))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_episode_layout_and_reproducible_pairs(self):
        terminal = np.zeros(90, dtype=bool)
        terminal[[29, 59, 89]] = True
        starts, lengths = episode_layout(terminal)
        np.testing.assert_array_equal(starts, [0, 30, 60])
        np.testing.assert_array_equal(lengths, [30, 30, 30])
        first = sample_problems(terminal, 10, 8, 42)
        second = sample_problems(terminal, 10, 8, 42)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(np.all(first[2] + 8 < lengths[first[1]]))

    def test_manifest_validation(self):
        terminal = np.zeros(60, dtype=bool)
        terminal[[29, 59]] = True
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "manifest.json"
            payload = {
                "format": "dtrc_paired_manifest_v1",
                "num_eval": 2,
                "goal_offset": 8,
                "episodes": [0, 1],
                "starts": [0, 5],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            load_manifest(path, terminal, 2, 8)
            with self.assertRaises(ValueError):
                load_manifest(path, terminal, 2, 9)

    def test_action_second_difference(self):
        jerk, monotonicity, episode_monotonic, _ = trajectory_metrics(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [3.0, 2.0, 1.0]
        )
        self.assertEqual(jerk, 0)
        self.assertEqual(monotonicity, 1)
        self.assertTrue(episode_monotonic)


if __name__ == "__main__":
    unittest.main()
