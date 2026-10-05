# Data preparation

## LeWM tasks

Download datasets from the [LeWM collection](https://huggingface.co/collections/quentinll/lewm).
Extract the HDF5 files under `STABLEWM_HOME`, preserving these relative paths:

| Task | Training dataset | Evaluation dataset |
|---|---|---|
| `pusht` | `pusht_expert_train.h5` | `pusht_expert_train.h5` |
| `tworoom` | `tworoom.h5` | `tworoom.h5` |
| `reacher` | `reacher.h5` | `dmc/reacher_random.h5` |
| `cube` | `ogbench/cube_single_expert.h5` | `ogbench/cube_single_expert.h5` |

```sh
export STABLEWM_HOME="$PWD/data"

dtrc-cache-hdf5 --task pusht \
  --h5 data/pusht_expert_train.h5 \
  --lewm-checkpoint models/encoders/pusht \
  --output-dir data/pusht_latents --device cuda
```

HDF5 input needs `pixels`, `action`, `step_idx`, and `ep_idx` or `episode_idx`.
The encoder reads only RGB pixels, resized to 224 × 224 with ImageNet channel
normalization. The released encoder directories contain `config.json` and
`weights.pt`; the loader also accepts Stable-WorldModel checkpoint identifiers.

## Visual OGBench tasks

Task identifiers are `pointmaze`, `cube-double`, `scene`, `antmaze-medium`,
`humanoidmaze-medium`, and `puzzle-3x3`.

Training uses a Lance image dataset with `pixels`, `action`, `episode_idx`,
and `step_idx`. Its companion `<dataset>.lance.metadata.json` records
`source`, `num_rows`, and `action_dim`. The source NPZ must match the episodes
and row ordering used to construct the image dataset.

```sh
dtrc-cache-lance --task pointmaze \
  --lance data/visual-pointmaze-large-navigate-v0.lance \
  --state-npz data/pointmaze-large-navigate-v0.npz \
  --lewm-checkpoint models/encoders/pointmaze \
  --output-dir data/pointmaze_latents --device cuda
```

Evaluation uses the simulator-state NPZ to reset start states and render goal
images. It requires `observations`, `terminals`, `qpos`, and `qvel`; Scene and
Puzzle also require `button_states`. A state-vector-only dataset without the
required reset fields is not sufficient. The policy receives RGB observations,
not these simulator states.

These visual datasets and state-augmented NPZ files are not bundled with the
model release. The commands assume that they have already been prepared;
downloading the model archives alone does not provide the evaluation data.

## Feature caches

Both caching commands produce:

```text
latents.npy   # per-frame frozen visual features
index.npz     # actions, episode indices, step indices, and episode boundaries
meta.json     # feature/action statistics and encoder identity
```

Interrupted caching resumes from `progress.json`. Use a new cache directory
after changing the dataset, preprocessing, or encoder. Cache statistics cover
all stored frames. Keep each cache paired with the encoder that produced it.

## Temporal diagnostics

```sh
dtrc-diagnose --checkpoint models/dtrc/pusht/final.pt \
  --latent-cache data/pusht_latents --gaps 1,2,4,8,16,25,50 \
  --output results/pusht_diagnostics.json
```

See [evaluation details](evaluation.md) for metric definitions and the scope
of these diagnostics.
