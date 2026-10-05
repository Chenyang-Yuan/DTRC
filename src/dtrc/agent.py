"""Alternating temporal-critic and advantage-weighted policy updates."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F
from .models import (
    DynamicsEnsemble,
    FlowNet,
    GoalCondEncoder,
    LatentStep,
    Quasimetric,
    RepHead,
)

VARIANTS = {
    "flow_bc": {"use_critic": False, "use_model": False, "use_support": False},
    "temporal": {"use_critic": True, "use_model": False, "use_support": False},
    "dtrc": {"use_critic": True, "use_model": True, "use_support": True},
    "no_support": {"use_critic": True, "use_model": True, "use_support": False},
    "symmetric": {
        "use_critic": True,
        "use_model": True,
        "use_support": True,
        "asym_dim": 0,
    },
    "no_model_augmentation": {
        "use_critic": True,
        "use_model": True,
        "use_support": True,
        "use_imagination": False,
    },
    "no_horizon": {
        "use_critic": True,
        "use_model": True,
        "use_support": True,
        "use_horizon": False,
    },
}


@dataclass
class DTRCConfig:
    obs_dim: int = 192
    action_dim: int = 2
    chunk: int = 1
    span: int = 8
    hidden: int = 512
    depth: int = 3
    dropout: float = 0.0
    rep_dim: int = 128
    sym_dim: int = 64
    asym_dim: int = 64
    lr: float = 0.0003
    tau: float = 0.005
    horizon_max: int = 50
    use_critic: bool = True
    n_step: int = 8
    d_max: float = 200.0
    critic_expectile: float = 0.3
    local_pairs_coef: float = 1.0
    consistency_coef: float = 0.1
    flow_steps: int = 8
    awr_temp: float = 1.0
    awr_clip: float = 20.0
    policy_use_rep: bool = True
    use_horizon: bool = True
    action_clip: float = 3.0
    use_model: bool = False
    use_support: bool = False
    use_imagination: bool = True
    dyn_heads: int = 4
    dyn_depth: int = 2
    dyn_rollout: int = 4
    improve_start: int = 10000
    n_candidates: int = 16
    improve_coef: float = 1.0
    improve_margin: float = 0.0
    aug_exact_targets: bool = False
    support_quantile: float = 0.9
    disagreement_quantile: float = 0.9
    threshold_ema: float = 0.99
    act_mode: str = "single"
    act_candidates: int = 16
    device: str = "cuda"


class DTRCAgent:
    """Train temporal structure and a separately optimized conditional policy.

    ``rep`` and ``quasi`` form the online temporal critic; the ``_t`` modules
    are its EMA targets. ``latent_step`` predicts in control space, whereas
    ``dyn`` predicts in frozen visual-feature space for candidate evaluation.
    Policy conditions and supervision are detached from both prediction paths.
    """

    def __init__(self, cfg: DTRCConfig):
        if cfg.n_step > cfg.span or cfg.dyn_rollout > cfg.span or cfg.chunk > cfg.span:
            raise ValueError("n_step, dyn_rollout and chunk must not exceed span")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        (d, a, k) = (cfg.obs_dim, cfg.action_dim, cfg.chunk)
        self.chunk_dim = a * k
        self.step = 0
        self.rep = RepHead(d, cfg.rep_dim, cfg.hidden, 2).to(self.device)
        self.quasi = Quasimetric(cfg.rep_dim, cfg.sym_dim, cfg.asym_dim).to(self.device)
        self.rep_t = RepHead(d, cfg.rep_dim, cfg.hidden, 2).to(self.device)
        self.quasi_t = Quasimetric(cfg.rep_dim, cfg.sym_dim, cfg.asym_dim).to(
            self.device
        )
        self.rep_t.load_state_dict(self.rep.state_dict())
        self.quasi_t.load_state_dict(self.quasi.state_dict())
        self.rep_t.requires_grad_(False)
        self.quasi_t.requires_grad_(False)
        self.latent_step = LatentStep(cfg.rep_dim, a).to(self.device)
        self.critic_opt = None
        if cfg.use_critic:
            self.critic_opt = torch.optim.Adam(
                [
                    *self.rep.parameters(),
                    *self.quasi.parameters(),
                    *self.latent_step.parameters(),
                ],
                lr=cfg.lr,
            )
        self.cond = GoalCondEncoder(
            d, cfg.rep_dim, use_rep=cfg.policy_use_rep and cfg.use_critic
        )
        self.policy = FlowNet(
            self.cond.out_dim,
            self.chunk_dim,
            cfg.hidden,
            cfg.depth,
            dropout=cfg.dropout,
        ).to(self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.dyn = self.support = None
        self.dyn_opt = self.support_opt = None
        if cfg.use_model:
            self.dyn = DynamicsEnsemble(
                d, a, cfg.hidden, cfg.dyn_depth, cfg.dyn_heads
            ).to(self.device)
            self.dyn_opt = torch.optim.Adam(self.dyn.parameters(), lr=cfg.lr)
            self.support = FlowNet(d, self.chunk_dim, cfg.hidden, cfg.depth).to(
                self.device
            )
            self.support_opt = torch.optim.Adam(self.support.parameters(), lr=cfg.lr)
        self.support_threshold = float("inf")
        self.disagreement_threshold = float("inf")
        self.generator = torch.Generator(
            device=self.device if self.device.type == "cuda" else "cpu"
        )
        self.generator.manual_seed(0)
        self.buffer_meta: dict = {}

    def _d(self, e_s, e_g):
        return self.quasi(e_s, e_g)

    @torch.no_grad()
    def _d_target(self, z_s, z_g):
        return self.quasi_t(self.rep_t(z_s), self.rep_t(z_g)).clamp(max=self.cfg.d_max)

    def _expectile(self, pred, target):
        """Use residual target - prediction; tau < 0.5 favors lower costs."""
        u = target - pred
        weight = torch.abs(self.cfg.critic_expectile - (u < 0).float())
        return (weight * u.pow(2)).mean()

    def _conditioning(self, z, g, h_norm):
        if not self.cfg.use_horizon:
            h_norm = torch.zeros_like(h_norm)
        if self.cond.use_rep:
            with torch.no_grad():
                (e, e_g) = (self.rep(z), self.rep(g))
        else:
            e = e_g = None
        return self.cond(z, g, e, e_g, h_norm)

    def _ema(self):
        with torch.no_grad():
            for src, dst in ((self.rep, self.rep_t), (self.quasi, self.quasi_t)):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.lerp_(p, self.cfg.tau)

    def _update_thresholds(self, z, a_chunk_flat):
        cfg = self.cfg
        with torch.no_grad():
            score = self.support.score(a_chunk_flat, z)
            (_, dis) = self.dyn.rollout(z, a_chunk_flat.view(len(z), cfg.chunk, -1))
            qs = torch.quantile(score, cfg.support_quantile).item()
            qd = torch.quantile(dis, cfg.disagreement_quantile).item()
        if not self.support_threshold < float("inf"):
            (self.support_threshold, self.disagreement_threshold) = (qs, qd)
            return
        m = cfg.threshold_ema
        self.support_threshold = m * self.support_threshold + (1 - m) * qs
        self.disagreement_threshold = m * self.disagreement_threshold + (1 - m) * qd

    @torch.no_grad()
    def _imagine(self, z, g, h_norm, cond, cost_data):
        """Select supported candidates; an empty valid set yields no update."""
        cfg = self.cfg
        (B, N, k) = (len(z), cfg.n_candidates, cfg.chunk)
        cond_rep = cond.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        cand = self.policy.sample(cond_rep, cfg.flow_steps, generator=self.generator)
        cand = cand.clamp(-cfg.action_clip, cfg.action_clip)
        z_rep = z.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        (z_im, dis) = self.dyn.rollout(z_rep, cand.view(B * N, k, -1))
        g_rep = g.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        cost = (k + self._d_target(z_im, g_rep)).view(B, N)
        valid = torch.ones(B, N, dtype=torch.bool, device=z.device)
        if cfg.use_support:
            score = self.support.score(cand, z_rep).view(B, N)
            valid &= score <= self.support_threshold
        valid &= dis.view(B, N) <= self.disagreement_threshold
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        (best_cost, idx) = cost.min(dim=1)
        rows = torch.arange(B, device=z.device)
        a_best = cand.view(B, N, -1)[rows, idx]
        z_best = z_im.view(B, N, -1)[rows, idx]
        adv = cost_data - best_cost
        use = torch.isfinite(best_cost) & (adv > cfg.improve_margin)
        adv = torch.where(use, adv, torch.zeros_like(adv))
        info = {
            "im/valid_frac": float(valid.float().mean()),
            "im/use_frac": float(use.float().mean()),
            "im/adv_mean_used": float(adv[use].mean()) if use.any() else 0.0,
            "im/support_thr": float(self.support_threshold),
            "im/disagreement_thr": float(self.disagreement_threshold),
        }
        return (a_best, z_best, adv, use, info)

    def update(self, batch: dict) -> dict:
        """Update auxiliary models, temporal networks, then the flow policy.

        A batch contains visual features [B, span + 1, obs_dim], standardized
        actions [B, span, action_dim], goal features, observed gaps, and
        normalized horizons. Unknown positive temporal offsets have gap -1.
        """
        cfg = self.cfg
        self.step += 1
        (k, n) = (cfg.chunk, cfg.n_step)
        (z_seq, a_seq) = (batch["seq_obs"], batch["seq_actions"])
        (g, gaps, h_norm) = (batch["goals"], batch["gaps"], batch["h_norm"])
        B = len(z_seq)
        (z, z_k, z_n) = (z_seq[:, 0], z_seq[:, k], z_seq[:, n])
        a_chunk = a_seq[:, :k].reshape(B, -1)
        info: dict = {}
        # Fit auxiliary models using recorded transitions only.
        if cfg.use_model:
            dyn_loss = self.dyn.rollout_loss(
                z_seq[:, : cfg.dyn_rollout + 1], a_seq[:, : cfg.dyn_rollout]
            )
            self.dyn_opt.zero_grad(set_to_none=True)
            dyn_loss.backward()
            self.dyn_opt.step()
            sup_loss = self.support.loss(a_chunk, z)
            self.support_opt.zero_grad(set_to_none=True)
            sup_loss.backward()
            self.support_opt.step()
            info["loss/dyn"] = float(dyn_loss.detach())
            info["loss/support"] = float(sup_loss.detach())
            if self.step % 50 == 0 or self.support_threshold == float("inf"):
                self._update_thresholds(z, a_chunk)
        # Freeze the policy's inputs and targets before updating the critic.
        cond = self._conditioning(z, g, h_norm)
        weights = torch.ones(B, device=self.device)
        imagined = None
        if cfg.use_critic:
            with torch.no_grad():
                cost_data = k + self._d_target(z_k, g)
                d_now_t = self._d_target(z, g)
                adv_data = d_now_t - cost_data
                weights = torch.exp(adv_data / cfg.awr_temp).clamp(max=cfg.awr_clip)
            info["value/adv_data_mean"] = float(adv_data.mean())
            info["value/awr_weight_mean"] = float(weights.mean())
            if (
                cfg.use_model
                and cfg.use_imagination
                and (self.step >= cfg.improve_start)
            ):
                imagined = self._imagine(z, g, h_norm, cond, cost_data)
                info.update(imagined[4])
        if cfg.use_critic:
            # Local offsets anchor the temporal scale; distant goals bootstrap.
            e_seq = self.rep(z_seq)
            (e, e_g) = (e_seq[:, 0], self.rep(g))
            with torch.no_grad():
                boot = n + self._d_target(z_n, g)
                exact = (gaps > 0) & (gaps <= n)
                target = torch.where(exact, gaps.float(), boot).clamp(max=cfg.d_max)
                if imagined is not None:
                    (_, z_best, _, use, _) = imagined
                    aug = k + self._d_target(z_best, g)
                    better = use & (aug < target)
                    if not cfg.aug_exact_targets:
                        better &= ~exact
                    target = torch.where(better, aug, target)
                    info["im/critic_aug_frac"] = float(better.float().mean())
            goal_loss = self._expectile(self._d(e, e_g), target)
            js = torch.arange(1, cfg.span + 1, device=self.device, dtype=torch.float32)
            d_local = self._d(e.unsqueeze(1).expand_as(e_seq[:, 1:]), e_seq[:, 1:])
            local_loss = self._expectile(d_local, js.unsqueeze(0).expand_as(d_local))
            with torch.no_grad():
                e1_t = self.rep_t(z_seq[:, 1])
            cons_loss = F.mse_loss(self.latent_step(e, a_seq[:, 0]), e1_t)
            critic_loss = (
                goal_loss
                + cfg.local_pairs_coef * local_loss
                + cfg.consistency_coef * cons_loss
            )
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self._ema()
            info.update(
                {
                    "loss/critic_goal": float(goal_loss.detach()),
                    "loss/critic_local": float(local_loss.detach()),
                    "loss/consistency": float(cons_loss.detach()),
                    "value/d_goal_mean": float(self._d(e, e_g).mean().detach()),
                    "value/target_mean": float(target.mean()),
                    "value/exact_frac": float(exact.float().mean()),
                }
            )
        # Only the policy receives gradients from these weighted flow losses.
        pi_loss = self.policy.loss(a_chunk, cond, weights)
        total = pi_loss
        if imagined is not None:
            (a_best, _, adv_im, use, _) = imagined
            w_im = (
                torch.exp(adv_im / cfg.awr_temp).clamp(max=cfg.awr_clip) * use.float()
            )
            im_loss = self.policy.loss(a_best, cond, w_im)
            total = total + cfg.improve_coef * im_loss
            info["loss/pi_imagined"] = float(im_loss.detach())
        self.policy_opt.zero_grad(set_to_none=True)
        total.backward()
        self.policy_opt.step()
        info["loss/pi"] = float(pi_loss.detach())
        return info

    @torch.no_grad()
    def act(
        self, z: torch.Tensor, goal: torch.Tensor, h_norm: torch.Tensor
    ) -> torch.Tensor:
        """Return standardized actions with shape (batch, chunk, action_dim)."""
        cfg = self.cfg
        was_training = self.policy.training
        self.policy.eval()
        cond = self._conditioning(z, goal, h_norm)
        (B, k) = (len(z), cfg.chunk)
        if cfg.act_mode == "best_of_n" and cfg.use_model:
            N = cfg.act_candidates
            cond_rep = cond.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            cand = self.policy.sample(
                cond_rep, cfg.flow_steps, generator=self.generator
            )
            cand = cand.clamp(-cfg.action_clip, cfg.action_clip)
            z_rep = z.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            g_rep = goal.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            (z_im, dis) = self.dyn.rollout(z_rep, cand.view(B * N, k, -1))
            cost = (k + self._d_target(z_im, g_rep)).view(B, N)
            valid = dis.view(B, N) <= self.disagreement_threshold
            if cfg.use_support:
                valid &= (
                    self.support.score(cand, z_rep).view(B, N) <= self.support_threshold
                )
            cost = torch.where(valid, cost, cost + 1000000.0)
            idx = cost.argmin(dim=1)
            action = cand.view(B, N, -1)[torch.arange(B, device=z.device), idx]
        else:
            action = self.policy.sample(cond, cfg.flow_steps, generator=self.generator)
            action = action.clamp(-cfg.action_clip, cfg.action_clip)
        self.policy.train(was_training)
        return action.view(B, k, -1)

    @torch.no_grad()
    def distance(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self._d_target(z, goal)

    def state_dict(self) -> dict:
        state = {
            "cfg": asdict(self.cfg),
            "step": self.step,
            "policy": self.policy.state_dict(),
            "rep": self.rep.state_dict(),
            "rep_t": self.rep_t.state_dict(),
            "quasi": self.quasi.state_dict(),
            "quasi_t": self.quasi_t.state_dict(),
            "latent_step": self.latent_step.state_dict(),
            "support_threshold": self.support_threshold,
            "disagreement_threshold": self.disagreement_threshold,
            "buffer_meta": self.buffer_meta,
        }
        if self.cfg.use_model:
            state["dyn"] = self.dyn.state_dict()
            state["support"] = self.support.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        for name in ("rep", "rep_t", "quasi", "quasi_t", "latent_step"):
            getattr(self, name).load_state_dict(state[name])
        if self.cfg.use_model and "dyn" in state:
            self.dyn.load_state_dict(state["dyn"])
            self.support.load_state_dict(state["support"])
        self.support_threshold = float(state.get("support_threshold", float("inf")))
        self.disagreement_threshold = float(
            state.get("disagreement_threshold", float("inf"))
        )
        self.step = int(state.get("step", 0))


def build_agent(
    variant: str,
    obs_dim: int,
    action_dim: int,
    device: str,
    overrides: dict | None = None,
) -> DTRCAgent:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choices={sorted(VARIANTS)}")
    values = dict(obs_dim=obs_dim, action_dim=action_dim, device=device)
    values.update(VARIANTS[variant])
    if overrides:
        values.update(overrides)
    return DTRCAgent(DTRCConfig(**values))


def load_agent(
    checkpoint: dict, device: str, act_mode: str | None = None
) -> tuple[DTRCAgent, dict]:
    meta = checkpoint.get("buffer_meta") or {}
    if not str(meta.get("format", "")).endswith("_frozen_lewm_latent_v1"):
        raise ValueError("not a dtrc checkpoint (buffer_meta.format mismatch)")
    cfg_values = dict(checkpoint["cfg"])
    cfg_values["device"] = device
    if act_mode:
        cfg_values["act_mode"] = act_mode
    agent = DTRCAgent(DTRCConfig(**cfg_values))
    agent.load_state_dict(checkpoint)
    agent.buffer_meta = meta
    for module in (agent.policy, agent.rep, agent.quasi, agent.rep_t, agent.quasi_t):
        module.eval().requires_grad_(False)
    if agent.dyn is not None:
        agent.dyn.eval().requires_grad_(False)
        agent.support.eval().requires_grad_(False)
    return (agent, meta)
