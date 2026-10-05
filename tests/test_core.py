"""Numerical and gradient checks for the temporal learning objectives."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dtrc.agent import VARIANTS, build_agent, load_agent
from dtrc.cache import LatentCache
from dtrc.data import TemporalBuffer
from dtrc.models import DynamicsEnsemble, FlowNet, Quasimetric
from dtrc.policy import DTRCPolicy
from fixtures import SMALL, make_cache


def setUpModule():
    torch.set_num_threads(1)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name) / "cache"
        make_cache(directory)
        self.cache = LatentCache(directory)
        self.buffer = TemporalBuffer(self.cache, span=4, device="cpu", seed=0)

    def agent(self, variant="dtrc", **overrides):
        values = {**SMALL, **overrides}
        if variant == "symmetric":
            values["asym_dim"] = 0
        agent = build_agent(
            variant, self.cache.feature_dim, self.cache.action_dim, "cpu", values
        )
        agent.buffer_meta = self.buffer.metadata()
        agent.buffer_meta["variant"] = variant
        return agent

    def test_cache_mmap_matches_eager(self):
        mapped = LatentCache(self.cache.cache_dir, mmap=True)
        rows = np.arange(20)
        np.testing.assert_allclose(mapped.normalized(rows), self.cache.normalized(rows))

    def test_segments_stay_inside_episode(self):
        rows = self.buffer.valid_indices
        self.assertTrue(np.all(rows + 4 <= self.cache.episode_end[rows]))
        batch = self.buffer.sample(64)
        self.assertEqual(batch["seq_obs"].shape, (64, 5, 12))
        self.assertEqual(batch["seq_actions"].shape, (64, 4, 2))
        self.assertTrue(torch.all((batch["h_norm"] > 0) & (batch["h_norm"] <= 1)))

    def test_near_goals_are_recorded_successors(self):
        sampler = TemporalBuffer(
            self.cache, span=4, device="cpu", p_current=1, p_traj=0, p_random=0
        )
        batch = sampler.sample(16)
        torch.testing.assert_close(batch["goals"], batch["seq_obs"][:, 1])
        self.assertTrue(torch.all(batch["gaps"] == 1))

    def test_known_gap_definition(self):
        rows = np.repeat(self.buffer.valid_indices, 8)
        goals, gaps = self.buffer._sample_goals(rows)
        known = (self.cache.episode_idx[goals] == self.cache.episode_idx[rows]) & (
            goals > rows
        )
        np.testing.assert_array_equal(gaps, np.where(known, goals - rows, -1))
        self.assertTrue(np.any(gaps == -1))

    def test_episode_start_partition(self):
        sampler = TemporalBuffer(
            self.cache, span=4, device="cpu", split_mode="episode", train_split=0.75
        )
        train = set(self.cache.episode_idx[sampler.valid_indices])
        holdout = set(self.cache.episode_idx[sampler.holdout_indices])
        self.assertFalse(train & holdout)

    def test_distance_components(self):
        distance = Quasimetric(8, 4, 4)
        x, y = torch.randn(2, 16, 8)
        torch.testing.assert_close(distance(x, x), torch.zeros(16), atol=1e-6, rtol=0)
        self.assertTrue(torch.all(distance(x, y) >= 0))
        self.assertFalse(torch.allclose(distance(x, y), distance(y, x)))
        symmetric = Quasimetric(8, 4, 0)
        torch.testing.assert_close(symmetric(x, y), symmetric(y, x))

    def test_smoothing_differs_from_exact_metric(self):
        distance = Quasimetric(1, 1, 0).double()
        distance.sym = torch.nn.Identity()
        x = torch.tensor([[0.0]], dtype=torch.float64)
        y = torch.tensor([[1e-4]], dtype=torch.float64)
        z = torch.tensor([[2e-4]], dtype=torch.float64)
        self.assertGreater(
            distance(x, z).item(), (distance(x, y) + distance(y, z)).item()
        )

    def test_lower_expectile_orientation(self):
        agent = self.agent("temporal")
        over = agent._expectile(torch.tensor([2.0]), torch.tensor([1.0])).item()
        under = agent._expectile(torch.tensor([0.0]), torch.tensor([1.0])).item()
        self.assertAlmostEqual(over, 0.7, places=6)
        self.assertAlmostEqual(under, 0.3, places=6)

    def test_goal_and_all_local_targets(self):
        agent = self.agent("temporal")
        batch = self.buffer.sample(4)
        batch["gaps"] = torch.tensor([1, 4, 9, -1])
        with torch.no_grad():
            expected = 4 + agent._d_target(batch["seq_obs"][:, 4], batch["goals"])
            expected[:2] = torch.tensor([1.0, 4.0])
            expected = expected.clamp(max=agent.cfg.d_max)
        original = agent._expectile
        targets = []

        def capture(pred, target):
            targets.append(target.detach().clone())
            return original(pred, target)

        with patch.object(agent, "_expectile", side_effect=capture):
            agent.update(batch)
        torch.testing.assert_close(targets[0], expected)
        torch.testing.assert_close(targets[1], torch.arange(1.0, 5.0).expand(4, -1))

    def test_policy_condition_is_detached(self):
        agent = self.agent()
        batch = self.buffer.sample(8)
        cond = agent._conditioning(
            batch["seq_obs"][:, 0], batch["goals"], batch["h_norm"]
        )
        self.assertFalse(cond.requires_grad)
        loss = agent.policy.loss(batch["seq_actions"][:, 0], cond)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in agent.rep.parameters()))
        self.assertTrue(any(p.grad is not None for p in agent.policy.parameters()))

    def test_target_ema(self):
        agent = self.agent()
        before = [p.clone() for p in agent.rep_t.parameters()]
        with torch.no_grad():
            for p in agent.rep.parameters():
                p.add_(1)
        agent._ema()
        for old, online, target in zip(
            before, agent.rep.parameters(), agent.rep_t.parameters()
        ):
            torch.testing.assert_close(target, old.lerp(online, agent.cfg.tau))
        self.assertTrue(all(not p.requires_grad for p in agent.rep_t.parameters()))

    def test_empty_candidate_set(self):
        agent = self.agent()
        batch = self.buffer.sample(8)
        z, g, h = batch["seq_obs"][:, 0], batch["goals"], batch["h_norm"]
        agent.support_threshold = -1
        agent.disagreement_threshold = -1
        _, _, adv, use, info = agent._imagine(
            z, g, h, agent._conditioning(z, g, h), torch.ones(8)
        )
        self.assertFalse(use.any())
        self.assertTrue(torch.equal(adv, torch.zeros(8)))
        self.assertEqual(info["im/valid_frac"], 0)

    def test_flow_weighting_batch_mean(self):
        flow = FlowNet(3, 2, hidden=8, depth=1)
        target, cond = torch.randn(4, 2), torch.randn(4, 3)
        torch.manual_seed(10)
        unweighted = flow.loss(target, cond)
        torch.manual_seed(10)
        doubled = flow.loss(target, cond, torch.full((4,), 2.0))
        torch.testing.assert_close(doubled, 2 * unweighted)
        self.assertEqual(flow.loss(target, cond, torch.zeros(4)).item(), 0)

    def test_dynamics_disagreement_definition(self):
        dynamics = DynamicsEnsemble(4, 2, hidden=8, depth=1, heads=3)
        z, a = torch.randn(5, 4), torch.randn(5, 2)
        members = dynamics.all_next(z, a)
        mean, disagreement = dynamics.rollout(z, a[:, None])
        torch.testing.assert_close(mean, members.mean(0))
        torch.testing.assert_close(disagreement, members.std(0).mean(-1))

    def test_all_variants_update_and_reload(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                agent = self.agent(variant)
                for _ in range(3):
                    report = agent.update(self.buffer.sample(12))
                    self.assertTrue(all(np.isfinite(v) for v in report.values()))
                checkpoint = Path(self.temp.name) / (variant + ".pt")
                torch.save(agent.state_dict(), checkpoint)
                restored, meta = load_agent(
                    torch.load(checkpoint, weights_only=False), "cpu"
                )
                batch = self.buffer.sample(4)
                z, g, h = batch["seq_obs"][:, 0], batch["goals"], batch["h_norm"]
                agent.generator.manual_seed(12)
                restored.generator.manual_seed(12)
                torch.testing.assert_close(agent.act(z, g, h), restored.act(z, g, h))
                self.assertEqual(meta["variant"], variant)

    def test_single_sample_execution_does_not_roll_out_dynamics(self):
        agent = self.agent()

        class ImageEncoder:
            def encode_frames(self, frames):
                return torch.zeros(len(frames), 12)

        class Environment:
            num_envs = 2

            class single_action_space:
                low = np.full(2, -1.0, dtype=np.float32)
                high = np.full(2, 1.0, dtype=np.float32)

        policy = DTRCPolicy(agent, ImageEncoder(), agent.buffer_meta, budget=50)
        policy.set_env(Environment())
        info = {
            "pixels": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "goal": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "_needs_flush": np.ones(2, dtype=bool),
        }
        with patch.object(
            agent.dyn, "rollout", side_effect=AssertionError("unexpected rollout")
        ):
            action = policy.get_action(info)
        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(np.all(np.abs(action) <= 1))


if __name__ == "__main__":
    unittest.main()
