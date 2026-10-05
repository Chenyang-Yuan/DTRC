# Implementation details

## Representation and directed temporal distance

The frozen encoder maps RGB observations to LeWM features. Cache statistics
standardize these features before a trainable representation head maps them to
control representations. A symmetric head and an order-sensitive head define
the temporal distance. A learned positive scale converts the combined output
to the temporal scale of the training targets.

The symmetric component uses `sqrt(r_squared + 1e-8) - 1e-4`. This smoothing
preserves zero self-distance up to floating-point precision. It differs from
the exact Euclidean norm in the ideal metric--residual construction; exact
triangle-inequality claims do not apply to this smoothed implementation.
The directional component is the largest positive coordinate difference from
the current representation to the goal representation.

## Temporal sampling

Each start index supplies eight recorded transitions within one episode.
Goals come from the one-step successor (probability 0.1), a geometrically
sampled future state (0.6), or a uniformly sampled cache row (0.3).
The geometric distribution has success probability 0.02. Future indices are
clipped to the final state of the episode.

A positive observed offset is available only for a strictly later index in
the same episode. The sampler assigns -1 to all other pairs, including a
dataset-wide sample that coincides with the current index. These pairs use the
bootstrap branch. There is no separate self-goal target override.

The reference command-line default partitions eligible starts by row with
training fraction 0.9. `--split-mode episode` partitions starts by episode.
Neither mode restricts dataset-wide goal sampling or full-cache normalization
statistics to the training-start subset. An episode-start partition therefore
does not establish a fully observation-disjoint validation set.

## Critic objective

For observed offsets from one through eight, the goal target is the recorded
offset. Other targets add eight to the target critic's prediction at the
recorded eighth successor. The implementation caps target-critic predictions
and final goal targets at 200.

Both goal regression and local regression use expectile 0.3 with residual
`target - prediction`. Every local offset from one through eight contributes
to the local loss. The action-conditioned representation head predicts the
observed one-step target representation. Its squared error is averaged over
batch and representation coordinates. The critic objective adds the goal
loss, the local loss, and 0.1 times the consistency loss.

Recorded offsets are trajectory durations, not shortest-path labels.
Lower-expectile regression favors lower costs within the supervised target
distribution; it does not guarantee the minimum feasible reaching time.

## Policy and model-assisted targets

The conditional flow policy receives the current and goal visual features,
their control representations, and a sinusoidal embedding of the normalized
remaining horizon. Its actual concatenation order is visual features, horizon
embedding, then control representations. Known offsets determine the horizon
up to a cap of 50. Unknown offsets use a uniform integer from 1 through 50.

Flow matching interpolates Gaussian noise and a standardized recorded action
at a uniform time in [0, 1]. The recorded-action weight is
`min(20, exp(temporal_advantage / 1.0))`. The temperature stays fixed; no
annealing schedule is applied. Weighted losses use a batch mean without
renormalizing by the sum of weights.

Four independently parameterized residual dynamics networks train on recorded
four-step sequences in standardized visual-feature space. An auxiliary
goal-free flow model trains on recorded actions conditioned on the current
visual feature. It is separate from the goal-conditioned policy.

After 10,000 updates, the policy samples 16 candidates per condition using
independent Gaussian initial noise and eight Euler integration steps. Actions
are clipped to [-3, 3] in standardized coordinates. Two flow residual samples
at times in [0.1, 0.9] estimate behavior-support error. Dynamics disagreement
averages the sample standard deviation across ensemble members, latent
coordinates, and predicted time steps. It is an agreement statistic, not a
calibrated probability of model accuracy.

The first model update initializes filter thresholds from the recorded-action
90th percentiles. Subsequent calibration occurs every 50 updates and uses
`threshold = 0.99 * threshold + 0.01 * current_quantile`.
All candidates receive predicted temporal costs. Invalid candidates receive
infinite selection cost. The lowest-cost valid candidate contributes only
when it improves over the recorded action under the target critic. An empty
valid set contributes zero model-assisted policy loss.

Accepted candidates receive the same exponential weighting rule using their
improvement over the recorded action. Their supplementary loss coefficient
is 1.0. They can also lower bootstrapped critic targets. Direct observed-offset
targets remain unchanged by default. These are training-time heuristics;
model agreement and behavior filtering do not guarantee true improvement.

## Alternating updates

Each iteration performs the following operations:

1. Sample a recorded segment and a relabeled goal.
2. Update dynamics and behavior-support models when enabled.
3. Compute detached policy conditions and recorded-action weights.
4. Evaluate model-assisted targets after warm-up when enabled.
5. Update the representation, distance heads, and consistency head.
6. Update representation and distance target networks with rate 0.005.
7. Update the flow policy using the detached targets and weights.

Policy gradients do not update the representation, critic, dynamics, or
support model. The policy condition and weights are computed before that
iteration's critic update. Adam optimizers are separate for the critic,
policy, dynamics, and support model.

Checkpoints store networks, thresholds, training-step count, and normalization
metadata. They do not store optimizer or random-generator states. `--resume`
continues from saved weights and step count, but is not a bitwise continuation
of an uninterrupted run.
