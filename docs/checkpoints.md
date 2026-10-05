# Pretrained models

The final-model collection covers PushT, TwoRoom, Reacher, Cube, PointMaze,
Cube-Double, Scene, AntMaze-Medium, HumanoidMaze-Medium, and Puzzle-3x3.
Each DTRC model was trained for 200,000 updates with training seed 42.

## Model files

Model assets are separate from the source distribution:

| Archive | Contents |
|---|---|
| `DTRC-checkpoints.zip` | Ten final DTRC checkpoints and a model index |
| `DTRC-encoders.zip` | Ten matching frozen LeWM models and architecture configurations |

Extract both archives in the repository root. A task then has this layout:

```text
models/
  model_index.json
  dtrc/pusht/final.pt
  encoders/pusht/config.json
  encoders/pusht/weights.pt
```

The index records SHA-256 checksums, dimensions, training seed, and update
count. Always pair a DTRC checkpoint with its indexed encoder. Visual features
from a different encoder are not interchangeable, even when their dimensions
match.

The exported checkpoints retain the learned parameters, temporal-critic and
dynamics modules, support thresholds, and feature/action normalizers.
Only deployment metadata and local paths are normalized. This collection
contains final models, not intermediate snapshots or optimizer state.

## Check and evaluate

```sh
python scripts/verify_models.py --root .

dtrc-evaluate-lewm --task pusht \
  --checkpoint models/dtrc/pusht/final.pt \
  --lewm-checkpoint models/encoders/pusht \
  --manifest manifests/pusht.json \
  --budget 50 --seed 42 --output results/pusht.json
```

The verification command checks every file digest and model interface. It
does not run a simulator benchmark. Evaluation still requires the task data
and the intended start--goal manifest. Checkpoint training seed and evaluation
seed are separate quantities.

See [evaluation protocols](evaluation.md) and
[reproduction commands](reproduction.md) before comparing success rates.
Dataset and pretrained-model terms apply separately; see
[third-party notices](../THIRD_PARTY_NOTICES.md).
