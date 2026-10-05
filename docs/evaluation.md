# Evaluation definitions

## Goal-conditioned control

LeWM evaluation covers PushT, TwoRoom, Reacher, and Cube. The additional visual
OGBench evaluation covers PointMaze, Cube-Double, Scene, AntMaze-Medium,
HumanoidMaze-Medium, and Puzzle-3x3. Manifests identify the start and goal
indices used for every method. A goal offset counts recorded transitions and
does not specify a minimum reaching time.

The default evaluation uses 50 pairs, a goal offset of 25, and a budget of 50
interactions. Use separate manifests for different offsets or problem seeds.
Keep the checkpoint fixed when estimating evaluation variability. Variability
over evaluation seeds is distinct from variability across training runs.

The four-task evaluator delegates resets and success detection to
Stable-WorldModel. The six-task evaluator restores simulator states from an
evaluation trajectory and renders both current and goal images. Privileged
states are used only for reset, goal rendering, and success evaluation; the
policy receives RGB observations, goal images, and a remaining horizon.
This future-goal evaluation is not the fixed-goal official OGBench protocol.

For maze tasks, success thresholds the native position error using the task
tolerance. Cube-Double requires both cube positions to match. Scene checks
object position, button states, drawer position, and window position. Puzzle
requires matching logical button states. Exact definitions are in
`environments/ogbench.py`.

## Temporal and model diagnostics

`dtrc-diagnose` samples 10% of episodes using its diagnostic seed. It does not
recover the training sampler's partition, so this subset must not be described
as a verified training-disjoint holdout.

- Offset MAE compares target-critic predictions to recorded future offsets.
- Progress rate measures the fraction of pairs with `d(next, goal) < d(current, goal)`.
- Reverse gap averages `d(goal, current) - d(current, goal)`.
- `ordinal_rank_distance_gap` correlates ordinal ranks assigned by a stable
  sort. Ties are not averaged. This statistic is not tie-corrected Spearman
  correlation; the explicit output name distinguishes the implementation.
- Dynamics diagnostics recursively use the ensemble-mean prediction as the
  next input. This differs from independent per-member autoregressive rollouts
  in dynamics training and multi-step candidate scoring.
- Support AUC compares recorded actions with Gaussian-perturbed standardized
  actions. Noise standard deviation is 2 and clipping bounds are [-3, 3].
  The rank-based estimator does not correct ties. Perturbed actions are a
  diagnostic contrast, not verified out-of-support actions.

The diagnostic implementation is intended for the default one-action policy.
For larger action chunks, its support comparison repeats a single action
rather than using the recorded multi-action sequence.

## Trajectory quality and timing

Action smoothness averages the Euclidean norm of second action differences.
It is not physical-space jerk. Distance monotonicity permits changes up to
1e-8. Four-task traces use cosine distance in frozen, unstandardized LeWM
features. The six-task evaluator's traces instead use native task goal error;
these two monotonicity quantities must not be pooled under one distance label.

The four-task recorder measures distance at each decision state before the
action. It does not append a post-terminal observation. The six-task trace
contains the initial error and the post-action error at every executed step.

Decision latency excludes simulator stepping and diagnostic distance
computation. It includes policy-side image encoding and synchronizes CUDA
around timing. Four-task evaluation times a batched policy call; six-task
evaluation times one environment's call. Report batch size and hardware when
comparing latencies.
