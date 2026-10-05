"""HDF5-to-latent integration using a small checkpoint-compatible encoder."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dtrc.backends.lewm import StandaloneLeWM, load_lewm_backend
from dtrc.cache import LatentCache


@unittest.skipUnless(importlib.util.find_spec("h5py"), "requires data dependencies")
class CachePipelineTests(unittest.TestCase):
    def test_hdf5_pixels_to_normalized_cache(self):
        import h5py
        from dtrc.cli.cache_hdf5 import main

        torch.set_num_threads(1)
        config = {
            "encoder": {
                "image_size": 32,
                "patch_size": 8,
                "depth": 1,
                "heads": 2,
                "intermediate": 32,
            },
            "predictor": {
                "num_frames": 3,
                "input_dim": 8,
                "depth": 1,
                "heads": 2,
                "mlp_dim": 32,
                "dim_head": 4,
                "dropout": 0.0,
                "emb_dropout": 0.0,
            },
            "action_encoder": {"input_dim": 2, "emb_dim": 8},
            "projector": {"input_dim": 8, "hidden_dim": 16},
            "pred_proj": {"input_dim": 8, "hidden_dim": 16},
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            weights = root / "encoder"
            weights.mkdir()
            (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
            torch.save(StandaloneLeWM(config).state_dict(), weights / "weights.pt")
            pixels = np.random.default_rng(4).integers(
                0, 256, (12, 32, 32, 3), dtype=np.uint8
            )
            with h5py.File(root / "trajectories.h5", "w") as data:
                data["pixels"] = pixels
                data["action"] = np.zeros((12, 2), dtype=np.float32)
                data["step_idx"] = np.arange(12)
                data["episode_idx"] = np.zeros(12, dtype=np.int64)
            arguments = [
                "cache",
                "--task",
                "pusht",
                "--h5",
                str(root / "trajectories.h5"),
                "--lewm-checkpoint",
                str(weights),
                "--output-dir",
                str(root / "cache"),
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--batch-size",
                "4",
            ]
            with patch("sys.argv", arguments):
                main()
            cached = LatentCache(root / "cache")
            self.assertEqual(cached.feature_dim, 8)
            self.assertEqual(cached.row_count, 12)
            self.assertFalse((root / "cache" / "progress.json").exists())
            self.assertEqual(cached.meta["source_columns_read_by_encoder"], ["pixels"])
            backend = load_lewm_backend(weights, device="cpu")
            self.assertTrue(
                all(not p.requires_grad for p in backend.model.parameters())
            )
            actual = backend.encode_uint8(torch.from_numpy(pixels)).numpy()
            stored = np.load(root / "cache" / "latents.npy")
            np.testing.assert_allclose(stored, actual, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
