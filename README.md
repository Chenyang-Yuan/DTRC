# Directed Temporal Representations for Control

**DTRC** learns a directed temporal distance on frozen LeWorldModel (LeWM)
features. Offline trajectories organize the representation by goal-reaching
cost in environment steps. The temporal critic guides policy learning through
recorded actions and supported model-predicted improvements.

[Installation](#installation) · [Pretrained models](#pretrained-models) ·
[Evaluation](#evaluation) · [Training](#training)

![DTRC overview](assets/overview.webp)

## Installation

Use Linux with Python 3.10 and a CUDA-compatible NVIDIA GPU. From the
repository root:

```sh
conda env create -f environment.yml
conda activate dtrc
```

`environment.yml` contains the dependency versions used with the released
models, including PyTorch 2.6.0 (CUDA 12.4), Stable-WorldModel 0.1.1, and
OGBench 1.2.1. For evaluation on a headless machine, set `MUJOCO_GL=egl`.

## Pretrained models

Download both archives from [GitHub Releases](../../releases):

| File | Contents |
|---|---|
| **DTRC-checkpoints.zip** | Final trained DTRC checkpoints for all ten tasks |
| **DTRC-encoders.zip** | The ten matching frozen LeWM encoders and their configurations |

| Task group | Included tasks |
|---|---|
| LeWM | PushT, TwoRoom, Reacher, Cube |
| Visual OGBench | PointMaze, Cube-Double, Scene, AntMaze-Medium, HumanoidMaze-Medium, Puzzle-3x3 |

For the four LeWM tasks, we use the official pretrained LeWM encoders.
For the six OGBench tasks, we include the visual encoders used in our runs.
**All encoders stay frozen during DTRC training.** Each DTRC checkpoint was
trained for 200,000 updates with seed 42 and contains the control representation,
temporal critic, policy, auxiliary models, and normalization statistics.
These are the final trained models, not encoder-only checkpoints.

Extract both archives in the repository root:

```sh
unzip DTRC-checkpoints.zip
unzip DTRC-encoders.zip
```

```text
models/
├── model_index.json
├── dtrc/<task>/final.pt
└── encoders/<task>/
    ├── config.json
    └── weights.pt
```

Task identifiers use lowercase names, such as `pusht`, `cube-double`, and
`antmaze-medium`. Use each checkpoint with its matching encoder. To check
the model files and loading interfaces:

```sh
python scripts/verify_models.py --root . --device cuda
```

## Evaluation

Both task groups use RGB current and goal observations. Evaluation samples
start–goal pairs from recorded trajectories; `--goal-offset` counts recorded
transitions and `--budget` sets the interaction limit. Save and reuse the
same manifest to compare methods on identical pairs.

### LeWM tasks

Download the datasets from the [LeWM collection](https://huggingface.co/collections/quentinll/lewm)
and extract them under `data/`. See [dataset filenames](docs/data.md#lewm-tasks).

```sh
export STABLEWM_HOME="$PWD/data"
export MUJOCO_GL=egl

dtrc-manifest-lewm --task pusht --num-eval 50 \
  --goal-offset 25 --seed 42 --output manifests/pusht.json

dtrc-evaluate-lewm --task pusht \
  --checkpoint models/dtrc/pusht/final.pt \
  --lewm-checkpoint models/encoders/pusht \
  --manifest manifests/pusht.json \
  --budget 50 --seed 42 --output results/pusht.json
```

### Visual OGBench tasks

Evaluation requires trajectory data with simulator states for resetting the
environment and rendering goal images. The policy only receives images.
See [OGBench data requirements](docs/data.md#visual-ogbench-tasks).

```sh
export MUJOCO_GL=egl

dtrc-manifest-ogbench --task pointmaze \
  --state-npz data/pointmaze-large-navigate-v0.npz \
  --goal-offset 25 --num-eval 50 --seed 42 \
  --output manifests/pointmaze.json

dtrc-evaluate-ogbench --task pointmaze \
  --checkpoint models/dtrc/pointmaze/final.pt \
  --lewm-checkpoint models/encoders/pointmaze \
  --state-npz data/pointmaze-large-navigate-v0.npz \
  --manifest manifests/pointmaze.json \
  --goal-offset 25 --num-eval 50 --budget 50 --seed 42 \
  --output results/pointmaze.json
```

This recorded future-goal evaluation differs from the official OGBench
fixed-goal protocol. Datasets and the paper's evaluation manifests are not
included. Exact table reproduction requires the matching manifests and
simulator versions. See [evaluation details](docs/evaluation.md) for metrics.

## Training

First cache the frozen encoder features, then train DTRC on the cache:

```sh
dtrc-cache-hdf5 --task pusht \
  --h5 data/pusht_expert_train.h5 \
  --lewm-checkpoint models/encoders/pusht \
  --output-dir data/pusht_latents --device cuda

dtrc-train --config configs/dtrc.json \
  --latent-cache data/pusht_latents \
  --out runs/pusht --device cuda
```

For OGBench, use [`dtrc-cache-lance`](docs/data.md#visual-ogbench-tasks) to
prepare the cache. Training writes logs, the resolved configuration, and
`checkpoints/latest.pt` and `checkpoints/final.pt` under the run directory.

[Configuration details](docs/configuration.md) list the hyperparameters and
component comparisons. The main implementation is in
[`models.py`](src/dtrc/models.py) (representation and networks),
[`agent.py`](src/dtrc/agent.py) (training), and
[`policy.py`](src/dtrc/policy.py) (image-conditioned execution).
See [implementation details](docs/method.md) for the training objectives.

Run the numerical and interface tests with:

```sh
python -m unittest discover -s tests -v
```

## Acknowledgements

DTRC builds on [LeWorldModel](https://github.com/lucas-maes/le-wm),
[Stable-WorldModel](https://github.com/galilai-group/stable-worldmodel), and
[OGBench](https://github.com/seohongpark/ogbench), and draws on
[Hilbert representations](https://github.com/seohongpark/HILP) and
metric–residual directed distances. This repository includes DTRC and its
component variants, not the comparison baselines.
See [third-party notices](licenses/README.md) for attribution and licensing.
