# Configuration

`dtrc-train --config configs/dtrc.json` loads training settings and network
settings from JSON. Explicit command-line arguments override training values.
Repeated `--override name=value` arguments override agent values.
Dataset paths and output locations are supplied at runtime.

Set `chunk`, `span`, and `horizon_max` in the training section or through their
dedicated command-line flags. Conflicting agent overrides are rejected so the
sampler and networks cannot silently use different horizons. Feature and action
dimensions come from the cache; choose the device with `--device`.

## Temporal and optimization settings

| Paper quantity | Code field | Default |
|---|---|---:|
| Action horizon k | chunk | 1 |
| Local offset limit J | span | 8 |
| Bootstrap horizon n | n_step | 8 |
| Policy horizon cap H | horizon_max | 50 |
| Dynamics training horizon | dyn_rollout | 4 |
| Expectile | critic_expectile | 0.3 |
| Local loss coefficient | local_pairs_coef | 1.0 |
| Consistency coefficient | consistency_coef | 0.1 |
| Policy weight temperature | awr_temp | 1.0 |
| Maximum policy weight | awr_clip | 20 |
| Model-assisted coefficient | improve_coef | 1.0 |
| Imagination warm-up | improve_start | 10,000 |
| Candidates per condition | n_candidates | 16 |
| Target update rate | tau | 0.005 |
| Filter quantiles | support_quantile, disagreement_quantile | 0.9 |
| Filter EMA coefficient | threshold_ema | 0.99 |
| Target cost cap | d_max | 200 |
| Adam learning rate | lr | 0.0003 |
| Batch size | batch_size | 1,024 |
| Training updates | train_steps | 200,000 |

`span` controls both available segment length and the largest local offset.
Changing the bootstrap horizon alone does not change the local offset limit.
The bootstrap horizon, action horizon, and dynamics horizon must fit inside
the sampled segment.

## Network architecture

| Module | Architecture |
|---|---|
| Control representation | Two width-512 Linear/LayerNorm/GELU blocks, linear output 128, output LayerNorm |
| Symmetric and directional heads | Width-256 hidden GELU layer, output 64 per head |
| Consistency head | Width-256 hidden GELU layer, residual output 128 |
| Goal-conditioned flow | Three width-512 Linear/LayerNorm/GELU blocks, linear action output |
| Behavior-support flow | Same flow architecture, conditioned on visual features only |
| Dynamics ensemble | Four residual networks, each with two width-512 Linear/LayerNorm/GELU blocks |
| Horizon and flow-time embeddings | 64-dimensional sinusoidal embeddings |

Visual-feature and action dimensions come from cache metadata. The reference
LeWM checkpoints produce 192-dimensional features. The flow output dimension
equals `chunk * action_dim`. Dropout defaults to zero.

## Component variants

| Variant | Change from the full method |
|---|---|
| dtrc | Full method |
| flow_bc | Unweighted flow matching on visual features and horizon; no learned control representation in policy input |
| temporal | Temporal representation and advantage weighting; no dynamics or model-assisted targets |
| no_support | Disable the behavior-support filter; retain dynamics agreement |
| symmetric | Set directional-head dimension to zero |
| no_model_augmentation | Train dynamics and support models but disable model-assisted targets |
| no_horizon | Replace normalized horizon input by zero |

The `no_support` variant still trains the support model. The
`no_model_augmentation` variant still trains the auxiliary models. These
choices preserve the corresponding component definitions rather than equating
them with a different variant.

`configs/dtrc_episode_split.json` explicitly selects episode-start splitting.
The other configuration files retain the row-start default. This is a sampling
choice, not a claim about which configuration produced a particular reported
checkpoint. Compare results only after matching configurations, checkpoints,
manifests, interaction budgets, and evaluation seeds.
