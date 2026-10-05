# Environment setup

## Reference runtime

Create the visual-control environment from the repository root:

```sh
conda env create -f environment.yml
conda activate dtrc
```

`environment.yml` names the environment `dtrc` and does not prescribe an
absolute installation prefix. Conda chooses the installation directory.
Use `conda env list` or `python -c "import sys; print(sys.prefix)"` to locate it.

The dependency profile records the runtime used to load the trained models:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.6.0, CUDA 12.4 build |
| TorchVision | 0.21.0 |
| NumPy | 2.2.6 |
| HDF5 interface / compression plugins | h5py 3.16.0 / hdf5plugin 7.0.0 |
| Stable-WorldModel / Stable-Pretraining | 0.1.1 / 0.1.7 |
| OGBench | 1.2.1 |
| Gymnasium / MuJoCo | 1.3.0 / 3.10.0 |
| Lance / Arrow | pylance 8.0.0 / pyarrow 24.0.0 |

`requirements-environment.txt` pins the principal simulation, data, and
visualization dependencies. It is a dependency profile, not a complete
platform-specific lock of every transitive package. The environment file
uses the official PyTorch CUDA 12.4 wheel index; see the
[PyTorch 2.6 installation commands](https://pytorch.org/get-started/previous-versions/#v260).
GPU execution also requires a compatible NVIDIA driver on the host.

## Data and rendering paths

Place downloaded model assets under `models/` in the repository root.
Keep datasets and latent caches outside the source distribution. For example:

```sh
export STABLEWM_HOME="$PWD/data"
export MUJOCO_GL=egl
```

`STABLEWM_HOME` is the Stable-WorldModel data root, not the Conda directory.
LeWM dataset names resolve relative to this data root. The OGBench evaluator
accepts explicit image-dataset and simulator-state paths. See
[data preparation](reproduction.md) for the expected layouts.

Use Linux for simulator evaluation. `MUJOCO_GL=egl` selects headless GPU
rendering; a working EGL runtime must be available on the host. Changing the
Conda environment name does not change the model or evaluation protocol.

## CPU-only core installation

The CUDA environment file is not required for latent-cache training or
numerical tests. In a Python 3.10 environment, install the CPU build first:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python -m pip install -e ".[data]"
python -m unittest discover -s tests -v
```

## Validation scope

Tests use synthetic trajectories and mocked encoder outputs where appropriate.
They check numerical behavior and interfaces, not benchmark success rates.
Model-loading checks do not establish a fresh reproduction of the reported
control results. Exact reproduction additionally requires matched datasets,
manifests, preprocessing, and simulator versions.

Load only trusted checkpoints. The packaged final models use tensor state
dictionaries; external object-format checkpoints may require Python
deserialization.
