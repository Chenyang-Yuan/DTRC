# Code guide

## Read the implementation

1. `models.py` defines the representation and distance parameterization.
2. `data.py` constructs recorded segments and relabeled goals.
3. `agent.py` constructs targets and performs one training update.
4. `policy.py` applies the trained policy to current and goal images.

The encoder remains frozen. The cache and online frontend apply the same
feature normalization stored in the training metadata.

## Representations and prediction spaces

| Quantity | Implementation | Space |
|---|---|---|
| Visual feature z | `LeWMBackend`, `LatentFrontend` | Frozen, standardized LeWM features |
| Control representation e | `RepHead` / `rep` | Trainable control features |
| Symmetric coordinates S(e) | `Quasimetric.sym` | Symmetric distance component |
| Directed potentials A(e) | `Quasimetric.asym` | Ordered residual component |
| Target temporal critic | `rep_t`, `quasi_t` | Exponential moving averages |
| One-step consistency model | `LatentStep` / `latent_step` | Control-representation space e |
| Dynamics ensemble | `DynamicsEnsemble` / `dyn` | Standardized visual-feature space z |
| Goal-conditioned policy | `FlowNet` / `policy` | Standardized actions |
| Behavior-support model | `FlowNet` / `support` | Recorded actions conditioned on z |

The consistency model regularizes the control representation. The dynamics
ensemble predicts candidate outcomes for model-assisted training. Neither
replaces or retrains the frozen LeWM encoder.

## One update

`DTRCAgent.update(batch)` follows this order:

1. Fit dynamics and behavior support on recorded data.
2. Compute detached policy inputs and recorded-action temporal advantages.
3. After warm-up, sample candidates, apply both filters, and select an
   alternative only when its predicted cost improves on the recorded action.
4. Fit the temporal critic with goal, local-offset, and consistency losses.
5. Update target representation and distance networks by EMA.
6. Fit the policy with weighted recorded actions and accepted candidates.

Policy gradients do not update the representation. Policy inputs and weights
are computed before the current critic update. The [method description](method.md)
specifies numerical conventions and target clipping.

## Inputs, checkpoints, and evaluation

`TemporalBuffer.sample` returns `seq_obs`, `seq_actions`, `goals`, `gaps`,
and `h_norm`. A gap of -1 means no positive same-episode offset is available;
it does not mean the goal is unreachable.

Checkpoints preserve network names and shapes, normalizers, filter thresholds,
and the update count. `--resume` is a weight restart, not exact optimizer or
random-state restoration. `latest.pt` is overwritten at each save interval;
the trainer does not retain a complete checkpoint history.

Both public evaluators select `act_mode=single`. The optional internal
`best_of_n` branch is not used by these evaluation commands. Simulator state
is confined to reset and success evaluation; `DTRCPolicy` receives current/goal
pixels and the remaining budget.
