<h1 align="center">DTRC</h1>
<h3 align="center">Directed Temporal Representations for Control</h3>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#data">Data</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="docs/method.md">Method</a>
</p>

<p align="center"><img src="assets/overview.webp" width="100%" alt="DTRC learns directed temporal representations from offline visual trajectories and uses temporal progress to guide action learning."></p>

DTRC learns a **directed temporal distance** on frozen LeWorldModel features.
Offline trajectories organize the representation by goal-reaching cost in
environment steps. The temporal critic weights recorded actions and selects
supported model-predicted improvements during policy training.

This repository contains DTRC, component configurations, feature caching, and
visual goal-conditioned evaluation. Final models and their matching visual
encoders are distributed separately; see [pretrained models](docs/checkpoints.md).
Datasets and baseline implementations are not bundled.

## Installation

The reference visual-control environment uses Linux, Python 3.10, and CUDA
12.4. Create it from the repository root:

```sh
conda env create -f environment.yml
conda activate dtrc
python -m unittest discover -s tests -v
```

For core training without simulators, install
[PyTorch](https://pytorch.org/get-started/locally/) for your hardware, then run:

```sh
python -m pip install -e ".[data]"
python -m unittest discover -s tests -v
```

Training from a prepared cache requires only PyTorch and NumPy. The Conda
configuration also installs the data and simulator dependencies. See
[environment setup](docs/environment.md) for versions, paths, and CPU setup.

## Data

Obtain the four-task datasets and pretrained visual encoders from
[LeWorldModel](https://github.com/lucas-maes/le-wm#data).
The adapter accepts a directory containing `config.json` and `weights.pt`,
or a checkpoint identifier supported by Stable-WorldModel.

Cache visual features once, then reuse them across training runs:

```sh
dtrc-cache-hdf5 --task pusht \
  --h5 data/pusht_expert_train.h5 \
  --lewm-checkpoint weights/pusht \
  --output-dir data/pusht_latents --device cuda
```

Visual OGBench uses a Lance image dataset and its matching simulator-state NPZ.
The policy receives images, not simulator states. See
[data preparation](docs/reproduction.md) for data formats and task names.

## Training

```sh
dtrc-train --config configs/dtrc.json \
  --latent-cache data/pusht_latents \
  --out runs/pusht --device cuda
```

The trainer writes scalar logs, the resolved configuration, and
`checkpoints/latest.pt` and `checkpoints/final.pt`. All trainable DTRC components
live above the frozen visual encoder. Default evaluation samples an action
directly, without model rollout or trajectory search.

Component comparisons change the configuration file:

| Configuration | Components |
|---|---|
| [`flow_bc.json`](configs/flow_bc.json) | Unweighted flow behavioral cloning |
| [`temporal.json`](configs/temporal.json) | Temporal representation and advantage weighting |
| [`dtrc.json`](configs/dtrc.json) | Full method with model-assisted targets |
| [`symmetric.json`](configs/symmetric.json) | Full method without the directional residual |
| [`no_support.json`](configs/no_support.json) | Full method without behavior-support filtering |

See [configuration](docs/configuration.md) for all variants and hyperparameters.

## Evaluation

Generate a start–goal manifest once and reuse it across compared methods:

```sh
dtrc-manifest-lewm --task pusht --num-eval 50 \
  --goal-offset 25 --seed 42 --output manifests/pusht.json

dtrc-evaluate-lewm --task pusht \
  --checkpoint runs/pusht/checkpoints/final.pt \
  --lewm-checkpoint weights/pusht \
  --manifest manifests/pusht.json \
  --budget 50 --seed 42 --output results/pusht.json
```

Stable-WorldModel resolves the dataset from `STABLEWM_HOME`. The additional
OGBench tasks use `dtrc-evaluate-ogbench`; see
[evaluation commands](docs/reproduction.md#matched-visual-ogbench-evaluation).
This is a recorded future-goal protocol, not the official fixed-goal benchmark.

Temporal-distance and model diagnostics:

```sh
dtrc-diagnose --checkpoint runs/pusht/checkpoints/final.pt \
  --latent-cache data/pusht_latents \
  --output results/pusht_diagnostics.json
```

See [metric definitions](docs/evaluation.md) before interpreting diagnostics.
Exact reproduction of a reported table also requires the corresponding
evaluation manifests and matching simulator versions. Downloaded final models
do not replace these protocol requirements.

## Code structure

| File | Role |
|---|---|
| [`models.py`](src/dtrc/models.py) | Representation, distance heads, flow networks, and dynamics ensemble |
| [`agent.py`](src/dtrc/agent.py) | Temporal targets, alternating updates, and model-assisted improvement |
| [`data.py`](src/dtrc/data.py) | Episode-bounded temporal sampling and goal relabeling |
| [`policy.py`](src/dtrc/policy.py) | Image-conditioned closed-loop execution |
| [`backends/`](src/dtrc/backends/) | Frozen LeWM loading and preprocessing |
| [`cli/`](src/dtrc/cli/) | Data preparation, training, evaluation, and diagnostics |

Start with `Quasimetric` and `DTRCAgent.update`.
The [code guide](docs/code.md) connects the modules to the method's notation.
Tests use small synthetic trajectories and do not require pretrained weights.

## Acknowledgements

DTRC builds on [LeWorldModel](https://github.com/lucas-maes/le-wm),
[Stable-WorldModel](https://github.com/galilai-group/stable-worldmodel),
and [OGBench](https://github.com/seohongpark/ogbench).
The representation draws on Hilbert representations and metric–residual
directed distances. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution and licensing scope.
