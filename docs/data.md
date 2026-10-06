# Data preparation

The policies use RGB images. Recorded simulator states serve only to render
PointMaze training images and to restore evaluation starts and goals.
Datasets are distributed by their original authors, not inside this repository.

## LeWM tasks

Sources: [LeWM repository](https://github.com/lucas-maes/le-wm) and
[official dataset/model collection](https://huggingface.co/collections/quentinll/lewm).

| Task | Official data download | HDF5 path under `data/datasets/` | Archive / extracted size |
|---|---|---|---|
| `pusht` | [pusht_expert_train.h5.zst](https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst) | `pusht_expert_train.h5` | 13.1 / 46.3 GB |
| `tworoom` | [tworoom.tar.zst](https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst) | `tworoom.h5` | 3.4 / 12.8 GB |
| `reacher` | [reacher.tar.zst](https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst) | `reacher.h5` | 23.8 / 98.9 GB |
| `cube` | [cube_single_expert.tar.zst](https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst) | `ogbench/cube_single_expert.h5` | 46.2 / 101.9 GB |

Sizes are approximate decimal GB. The download script keeps both the archive
and extracted HDF5, so allow space for both, plus the feature cache.

Original LeWM encoder pages: [PushT](https://huggingface.co/quentinll/lewm-pusht),
[TwoRoom](https://huggingface.co/quentinll/lewm-tworooms),
[Reacher](https://huggingface.co/quentinll/lewm-reacher), and
[Cube](https://huggingface.co/quentinll/lewm-cube). Our encoder archive packages
these four encoders together with the six project-trained OGBench encoders;
use the matching `models/encoders/<task>/` directory in the commands below.

```sh
python scripts/download_data.py --task tworoom
export STABLEWM_HOME="$PWD/data"

dtrc-cache-hdf5 --task tworoom \
  --h5 data/datasets/tworoom.h5 \
  --lewm-checkpoint models/encoders/tworoom \
  --output-dir data/tworoom_latents --device cuda
```

The script handles both `.h5.zst` and `.tar.zst`, places the HDF5 correctly,
and retains compressed downloads under `data/downloads/<task>/`. It only
renames the extracted HDF5 into place after extraction finishes.

For evaluation, pass the table's HDF5 path relative to `data/datasets/` as
`--dataset` to **both** LeWM manifest and evaluation commands. For example:

```sh
dtrc-manifest-lewm --task reacher --dataset reacher.h5 \
  --num-eval 50 --goal-offset 25 --seed 42 --output manifests/reacher.json

dtrc-evaluate-lewm --task reacher --dataset reacher.h5 \
  --checkpoint models/dtrc/reacher/final.pt \
  --lewm-checkpoint models/encoders/reacher \
  --manifest manifests/reacher.json --budget 50 --seed 42 \
  --output results/reacher.json
```

The historical Reacher default is `dmc/reacher_random.h5`; the official
download above contains `reacher.h5`. The explicit override avoids depending
on that historical local file. It does not establish that the downloaded
dataset and a previous paper evaluation manifest are identical.

HDF5 inputs contain `pixels`, `action`, `step_idx`, and `ep_idx` or
`episode_idx`. The encoder reads only RGB pixels, with ImageNet normalization
and resizing to 224 × 224. No HDF5-to-Lance conversion is needed for these tasks.

## OGBench tasks

Sources: [OGBench repository](https://github.com/seohongpark/ogbench) and
[official downloader](https://github.com/seohongpark/ogbench/blob/master/ogbench/utils.py).
The following links point to its official dataset server.

| Task | Training NPZ | Evaluation NPZ |
|---|---|---|
| `pointmaze` | [pointmaze-large-navigate-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/pointmaze-large-navigate-v0.npz) | Same file |
| `cube-double` | [visual-cube-double-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/visual-cube-double-play-v0.npz) | [cube-double-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-v0.npz) |
| `scene` | [visual-scene-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/visual-scene-play-v0.npz) | [scene-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/scene-play-v0.npz) |
| `antmaze-medium` | [visual-antmaze-medium-navigate-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/visual-antmaze-medium-navigate-v0.npz) | [antmaze-medium-navigate-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/antmaze-medium-navigate-v0.npz) |
| `humanoidmaze-medium` | [visual-humanoidmaze-medium-navigate-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/visual-humanoidmaze-medium-navigate-v0.npz) | [humanoidmaze-medium-navigate-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/humanoidmaze-medium-navigate-v0.npz) |
| `puzzle-3x3` | [visual-puzzle-3x3-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/visual-puzzle-3x3-play-v0.npz) | [puzzle-3x3-play-v0.npz](https://rail.eecs.berkeley.edu/datasets/ogbench/puzzle-3x3-play-v0.npz) |

### Download

```sh
python scripts/download_data.py --task cube-double --purpose both
```

Files go under `data/ogbench/`. `--purpose train` downloads the training
column; `--purpose eval` downloads the evaluation column. The official
OGBench downloader also fetches each dataset's `-val.npz` file. DTRC's full
training commands use the file **without** `-val`.

Keep the raw NPZ files. OGBench's processed dataset API can change terminal
markers or drop rows and reset information; it is not a substitute for the
raw NPZ in the commands below.

### Convert images and cache features

```sh
dtrc-prepare-ogbench --task cube-double \
  --source data/ogbench/visual-cube-double-play-v0.npz \
  --output data/ogbench/visual-cube-double-play-v0.lance \
  --work-dir data/preparation

dtrc-cache-lance --task cube-double \
  --lance data/ogbench/visual-cube-double-play-v0.lance \
  --lewm-checkpoint models/encoders/cube-double \
  --output-dir data/cube-double_latents --device cuda

dtrc-train --config configs/dtrc.json \
  --latent-cache data/cube-double_latents \
  --out runs/cube-double --seed 42 --device cuda
```

For the five visual NPZs, conversion reads the recorded RGB observations.
For PointMaze, there is no official visual NPZ: select `--task pointmaze`
with `pointmaze-large-navigate-v0.npz`. The converter restores each recorded
`qpos`/`qvel` and renders a 64 × 64 image with the project's Stable-WorldModel
PointMaze environment. Both paths preserve action and episode ordering.

Conversion writes a Lance directory and a `.lance.metadata.json` sidecar.
Keep them together. It stages uncompressed RGB pixels in `--work-dir` and
encodes them as JPEG with quality 95. Allow `rows × height × width × 3`
bytes for this work file, in addition to the source NPZ, Lance data, and cache.
For 64 × 64 images, one million rows need about 12.3 GB of pixel work space.
The work file is not needed after successful conversion; the converter leaves
it in place rather than deleting files automatically.

`--max-episodes 2` converts only the first two complete episodes for a short
pipeline check. Use different output and work paths for a later full conversion.

The optional `dtrc-cache-lance --state-npz` argument, if supplied, must name
the **same training NPZ** used for conversion. Do not supply the state-based
evaluation NPZ: visual and nonvisual datasets need not share trajectories or
row indices. Omitting this option uses the recorded Lance provenance.

### Evaluate

Use the evaluation NPZ from the table, not the visual training NPZ:

```sh
dtrc-manifest-ogbench --task cube-double \
  --state-npz data/ogbench/cube-double-play-v0.npz \
  --goal-offset 25 --num-eval 50 --seed 42 --output manifests/cube-double.json

dtrc-evaluate-ogbench --task cube-double \
  --checkpoint models/dtrc/cube-double/final.pt \
  --lewm-checkpoint models/encoders/cube-double \
  --state-npz data/ogbench/cube-double-play-v0.npz \
  --manifest manifests/cube-double.json \
  --goal-offset 25 --num-eval 50 --budget 50 --seed 42 \
  --output results/cube-double.json
```

The raw evaluation NPZ provides `observations`, `terminals`, `qpos`, and
`qvel`; Scene and Puzzle also provide `button_states`. These arrays support
simulator reset and success measurement. They are not policy inputs.

## Feature caches

Both caching commands produce:

```text
latents.npy   # per-frame frozen visual features
index.npz     # actions, episode indices, step indices, and episode boundaries
meta.json     # feature/action statistics and encoder identity
```

Interrupted caching resumes from `progress.json`. Use a new cache directory
after changing the data, episode limit, preprocessing, or encoder. Keep each
cache paired with the encoder that produced it. A 192-dimensional float32
latent cache uses 768 bytes per frame, plus the trajectory index.

## Temporal diagnostics

```sh
dtrc-diagnose --checkpoint models/dtrc/pusht/final.pt \
  --latent-cache data/pusht_latents --gaps 1,2,4,8,16,25,50 \
  --output results/pusht_diagnostics.json
```

See [evaluation details](evaluation.md) for metric definitions and protocol
limitations, and [configuration](configuration.md) for training options.

## Common setup issues

- **HDF5 not found:** set `STABLEWM_HOME` to the absolute `data/` directory,
  not `data/datasets/`; pass the table's relative filename as `--dataset`.
- **CUDA out of memory during caching:** lower `--batch-size`, for example
  to 32. This changes encoder batching, not the selected training data.
- **Worker or shared-memory errors during caching:** use `--num-workers 0`.
- **EGL initialization fails:** check that the GPU is exposed to the process
  and the system has EGL installed. On Ubuntu, the relevant system package is
  `libegl1`; a Conda Python environment alone does not install GPU drivers.
- **An `ale-py` warning appears:** it concerns unused Atari environments and
  does not prevent these ten tasks from running. The environment file installs
  the needed task dependencies without Stable-WorldModel's full environment extra.
- **State NPZ supplied to a visual conversion:** use the training column,
  including the `visual-` prefix, except for PointMaze.
- **Output already exists:** reuse a completed cache only with its original
  inputs. Use a new directory for a different dataset, encoder, or episode limit.
