"""Check raw trajectory conversion and bounded HDF5 cache indexing."""

from pathlib import Path
import importlib.util
import io
import tempfile
import tarfile
import unittest
from unittest import mock

import h5py
import numpy as np

from dtrc.cli.prepare_ogbench import extract_pixels, read_arrays
from dtrc.cli.cache_hdf5 import write_index
from dtrc.environments.ogbench import episode_layout


class PreparationTests(unittest.TestCase):
    def test_lewm_download_extracts_only_expected_hdf5(self):
        try:
            import zstandard
        except ImportError:
            self.skipTest("zstandard is part of the full data environment")
        script = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
        spec = importlib.util.spec_from_file_location("download_data", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
                for name, value in [
                    ("../../unwanted", b"ignore"),
                    ("tworoom.h5", b"hdf5-test"),
                ]:
                    item = tarfile.TarInfo(name)
                    item.size = len(value)
                    archive.addfile(item, io.BytesIO(value))
            compressed = root / "source.zst"
            compressed.write_bytes(
                zstandard.ZstdCompressor().compress(tar_buffer.getvalue())
            )
            fake_hf = mock.Mock()
            fake_hf.hf_hub_download.return_value = str(compressed)
            with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                module.download_lewm("tworoom", root)
                self.assertEqual(
                    (root / "datasets/tworoom.h5").read_bytes(), b"hdf5-test"
                )
                self.assertFalse((root.parent / "unwanted").exists())
                self.assertFalse((root / "datasets/tworoom.h5.partial").exists())
                module.download_lewm("tworoom", root)
                fake_hf.hf_hub_download.assert_called_once()

    def test_visual_prefix_preserves_pixels_and_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pixels = np.arange(10 * 4 * 4 * 3, dtype=np.uint8).reshape(10, 4, 4, 3)
            source = root / "source.npz"
            np.savez_compressed(source, observations=pixels)
            actual, source_rows = extract_pixels(source, root / "pixels.npy", 6)
            self.assertEqual(source_rows, 10)
            np.testing.assert_array_equal(actual, pixels[:6])
            del actual

    def test_state_observations_are_not_treated_as_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / "source.npz", observations=np.zeros((10, 12), np.float32))
            with self.assertRaisesRegex(ValueError, "RGB"):
                extract_pixels(root / "source.npz", root / "pixels.npy", 6)

    def test_raw_episode_endpoints_are_preserved(self):
        starts, lengths = episode_layout([0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(starts, [0, 3])
        np.testing.assert_array_equal(lengths, [3, 3])

    def test_missing_reset_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.npz"
            np.savez(
                path,
                observations=np.zeros((8, 2)),
                actions=np.zeros((8, 2)),
                terminals=np.ones(8),
            )
            with self.assertRaisesRegex(ValueError, "qpos"):
                read_arrays(path, "pointmaze")

    def test_hdf5_prefix_keeps_complete_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with h5py.File(root / "source.h5", "w") as handle:
                handle["action"] = np.zeros((12, 2))
                handle["episode_idx"] = np.repeat([3, 4, 5], 4)
                handle["step_idx"] = np.tile(np.arange(4), 3)
                handle["pixels"] = np.zeros((12, 4, 4, 3), np.uint8)
            meta = write_index(root / "source.h5", root, max_episodes=2)
            self.assertEqual(meta["rows"], 8)
            self.assertEqual(meta["num_episodes"], 2)
            with np.load(root / "index.npz") as data:
                np.testing.assert_array_equal(data["episode_end"], [3] * 4 + [7] * 4)


if __name__ == "__main__":
    unittest.main()
