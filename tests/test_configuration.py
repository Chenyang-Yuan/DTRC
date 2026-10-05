"""Reject ambiguous configuration before starting a training run."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from dtrc.cache import LatentCache
from dtrc.cli.train import agent_settings, parse_overrides
from dtrc.data import TemporalBuffer
from fixtures import make_cache


class ConfigurationTests(unittest.TestCase):
    def args(self, **values):
        defaults = dict(agent_settings={}, override=[], chunk=1, span=8, horizon_max=50)
        return Namespace(**(defaults | values))

    def test_default_horizons(self):
        self.assertEqual(
            agent_settings(self.args()),
            dict(chunk=1, span=8, horizon_max=50, n_step=8, dyn_rollout=4),
        )

    def test_explicit_agent_values_take_precedence(self):
        settings = agent_settings(
            self.args(agent_settings={"n_step": 2}, override=["n_step=4"])
        )
        self.assertEqual(settings["n_step"], 4)

    def test_conflicting_sampler_settings(self):
        for name in ("chunk", "span", "horizon_max"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "conflicts"),
            ):
                agent_settings(self.args(override=[f"{name}=99"]))

    def test_reserved_cache_dimensions(self):
        for name in ("obs_dim", "action_dim", "device"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "determined"),
            ):
                agent_settings(self.args(agent_settings={name: 1}))

    def test_override_types(self):
        self.assertEqual(
            parse_overrides(["n_candidates=8", "use_support=false"]),
            {"n_candidates": 8, "use_support": False},
        )
        with self.assertRaises(ValueError):
            parse_overrides(["n_candidates"])

    def test_single_episode_partition_has_clear_error(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "cache"
            make_cache(path, episodes=1)
            cache = LatentCache(path)
            with self.assertRaisesRegex(ValueError, "two eligible episodes"):
                TemporalBuffer(cache, span=4, device="cpu", split_mode="episode")
            sampler = TemporalBuffer(
                cache, span=4, device="cpu", split_mode="episode", train_split=1
            )
            self.assertGreater(len(sampler.valid_indices), 0)


if __name__ == "__main__":
    unittest.main()
