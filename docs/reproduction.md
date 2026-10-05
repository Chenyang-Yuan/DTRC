# Data and evaluation

## Required inputs

Obtain the LeWM benchmark datasets and matching pretrained LeWM checkpoints
from their original distribution. For visual OGBench, use the visual trajectory
datasets and corresponding encoder checkpoints. A checkpoint must match the
task and the feature cache used for DTRC training.

The LeWM adapter accepts a directory containing `config.json` and `weights.pt`,
or a checkpoint name/path supported by Stable-WorldModel. It does not train a
new visual encoder. The trainable residual dynamics ensemble is separate from
the pretrained LeWM predictor.

Paths below are relative examples. Supply actual paths to the external data
and checkpoints. The source package contains none of these external assets.

## HDF5 feature caching

```sh
dtrc-cache-hdf5 --task pusht --h5 data/pusht_expert_train.h5 --lewm-checkpoint weights/pusht --output-dir data/pusht_latents --device cuda
```

The HDF5 input needs `pixels`, `action`, `step_idx`, and `ep_idx` or
`episode_idx`. The encoder reads only `pixels`. Images use the pretrained
LeWM preprocessing: ImageNet channel normalization and resizing to 224 by 224.
The cache records feature and action statistics over all stored frames.

| Task | Training dataset | Evaluation dataset |
|---|---|---|
| pusht | pusht_expert_train.h5 | pusht_expert_train.h5 |
| tworoom | tworoom.h5 | tworoom.h5 |
| reacher | reacher.h5 | dmc/reacher_random.h5 |
| cube | ogbench/cube_single_expert.h5 | ogbench/cube_single_expert.h5 |

## Lance feature caching

```sh
dtrc-cache-lance --task pointmaze --lance data/visual-pointmaze-large-navigate-v0.lance --state-npz data/pointmaze-large-navigate-v0.npz --lewm-checkpoint weights/pointmaze --output-dir data/pointmaze_latents --device cuda
```

Lance input requires RGB `pixels`, `action`, `episode_idx`, and `step_idx`.
Its companion `<dataset>.lance.metadata.json` records `source`, `num_rows`,
and `action_dim`. If `--state-npz` is supplied, it must identify the source
used to construct the visual Lance dataset. Training and evaluation NPZ files
must not be interchanged merely because their row counts match.

The cache layout is:

```text
latents.npy   float32 per-frame visual features
index.npz     actions, episode indices, step indices, episode boundaries
meta.json     feature/action statistics and encoder identity
```

Interrupted caching can resume from `progress.json`. Use a new output directory
when changing the dataset or encoder. Cached features must be regenerated
after changing preprocessing or encoder weights.

## Training and component comparisons

```sh
dtrc-train --config configs/dtrc.json --latent-cache data/pusht_latents --out runs/pusht --device cuda
dtrc-train --config configs/temporal.json --latent-cache data/pusht_latents --out runs/pusht_temporal --device cuda
dtrc-train --config configs/flow_bc.json --latent-cache data/pusht_latents --out runs/pusht_flow_bc --device cuda
```

Use a fresh output directory for each run. The trainer writes configuration,
scalar logs, and `checkpoints/latest.pt` and `checkpoints/final.pt`. Checkpoint
metadata contains normalization statistics needed by online image encoding.

An episode-start split is an explicit alternative:

```sh
dtrc-train --config configs/dtrc_episode_split.json --latent-cache data/pusht_latents --out runs/pusht_episode_split --device cuda
```

## Matched LeWM evaluation

Stable-WorldModel resolves evaluation datasets using its configured data cache.
Construct a manifest once and reuse it for all compared methods.

```sh
dtrc-manifest-lewm --task pusht --num-eval 50 --goal-offset 25 --seed 42 --output manifests/pusht.json
dtrc-evaluate-lewm --task pusht --checkpoint runs/pusht/checkpoints/final.pt --lewm-checkpoint weights/pusht --manifest manifests/pusht.json --budget 50 --seed 42 --output results/pusht.json
```

## Matched visual OGBench evaluation

Evaluation NPZ files must include `observations`, `terminals`, `qpos`, and
`qvel`. Scene and Puzzle also require `button_states`. The evaluator renders
the goal from the exact goal simulator state. It does not use an unrelated
training-image row as the goal.

```sh
dtrc-manifest-ogbench --task pointmaze --state-npz data/pointmaze-large-navigate-v0.npz --goal-offset 25 --num-eval 50 --seed 42 --output manifests/pointmaze.json
dtrc-evaluate-ogbench --task pointmaze --checkpoint runs/pointmaze/checkpoints/final.pt --lewm-checkpoint weights/pointmaze --state-npz data/pointmaze-large-navigate-v0.npz --manifest manifests/pointmaze.json --goal-offset 25 --num-eval 50 --budget 50 --seed 42 --output results/pointmaze.json
```

`--record-success-dir` saves lossless post-action RGB frames and the goal image
for every successful episode. Action and distance traces remain in the result
JSON. These artifacts are runtime outputs, not part of the source distribution.

Newly sampled manifests do not establish exact reproduction of an existing
result table. Exact table reproduction also requires the matching pretrained
weights, trained checkpoints, stored evaluation manifests, configurations,
and simulator versions. These assets are not supplied in this source-only
package. Baseline implementations are separate from the DTRC component variants.
Success predicates and reset behavior follow the installed simulator stack.
The package declares the published Stable-WorldModel 0.1.1 interface; a locally
modified installation with the same version label need not be numerically
equivalent. Record dependency revisions when reproducing control results.

## Diagnostics

```sh
dtrc-diagnose --checkpoint runs/pusht/checkpoints/final.pt --latent-cache data/pusht_latents --gaps 1,2,4,8,16,25,50 --output results/pusht_diagnostics.json
```

Read [Evaluation definitions](evaluation.md) before interpreting diagnostic
statistics or combining trajectory-quality outputs from the two evaluators.
