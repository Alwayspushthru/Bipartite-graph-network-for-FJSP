# Code Feasibility Analysis for Reviewer Comments

Status: `DESIGN_COMPLETE_IMPLEMENTATION_PENDING`

This analysis is based on B0 commit `d145d92f178ecff6438da0dbb90f9d3bc37cb72f`. The four designs below are hypotheses to be tested. None is assumed to be effective or suitable for the revised manuscript before verified results are available.

## B0 facts relevant to the review

- The action space contains only the first unscheduled operation of each unfinished job and its eligible machines.
- The eight operation features already include the remaining-operation ratio and the sum of expected processing times of all remaining operations. B0 therefore retains coarse future-work information, but it does not encode the distribution of the waiting operations.
- The five machine features already include the number of currently feasible candidate operations, machine ready time, expected workload from all unscheduled operations, accumulated idle time, and utilization.
- The six pair features contain processing time, earliest start time, two waiting-time directions, and two processing-time ratios. They do not explicitly contain machine queue length or cumulative assigned processing load.
- The GRU input is the concatenation of mean-pooled job, machine, and valid-pair embeddings. `GRUCell` already has internal reset and update gates, but B0 has no explicit decision-urgency signal controlling the amount of historical state retained at each step.
- The reward is the decrease in the maximum lower-bound completion time. With `gamma=1`, its undiscounted episode return telescopes to the negative final lower-bound increase and is dense across the construction sequence.

## R1.1: waiting-operation information

Feasibility: `FEASIBLE_WITH_MODERATE_INTERFACE_CHANGE`

The environment stores every operation and can identify the unscheduled suffix of each job. The main constraint is that `EnvState`, PPO memory, single-step inference, full-sequence recomputation, and beam inference currently pass only job, machine, and pair tensors. Adding an operation-level waiting tensor would therefore affect the entire state interface and weaken the compact-state claim.

### Minimum candidate B0+Q

Keep the action graph unchanged and add one fixed-size waiting-operation summary per job. Excluding the current candidate operation, compute four descriptors from the remaining suffix:

1. waiting-operation count divided by the job length;
2. waiting expected workload divided by the total expected workload of the job;
3. mean eligible-machine ratio of the waiting operations;
4. mean normalized processing time of the waiting operations.

Embed this four-dimensional vector with a separate MLP and fuse it into the current job embedding through a learned sigmoid gate:

```text
h_wait = MLP_wait(x_wait)
g_wait = sigmoid(W_gate([h_job, h_wait]))
h_job_new = LayerNorm(h_job + g_wait * h_wait)
```

This design implicitly represents all waiting operations while preserving one actionable operation node per job and fixed graph size. It is preferred over adding every waiting operation as a graph node.

Required code areas: `env/FJSPEnv.py`, `EnvState`, `model/ppo.py`, `model/BiGraphNetwork.py`, and the call sites in `train.py` and `test.py`.

Primary risks:

- the new summary partly overlaps with the existing `rem_ops` and `rem_work` features;
- an incorrect suffix mask could leak completed or padded operations;
- the wider state interface increases implementation risk, especially in sequence recomputation;
- the added summary may not explain Brandimarte performance if the limitation instead arises from distribution shift or decoding.

Acceptance checks: zero summary for jobs with no waiting operation; no padded-operation contribution; identical action mask and action-space size to B0; valid single-step and `forward_sequence` shapes; no measurable growth with the total number of operations beyond environment-side summary construction.

## R1.2: explicit machine-load pair features

Feasibility: `STRAIGHTFORWARD`

B0 already exposes related information through machine embeddings, but the information reaches pair attention only indirectly through the query/key projections. The reviewer specifically asks for explicit pair-level load signals.

### Minimum candidate B0+L

Extend the pair feature dimension from 6 to 8 by broadcasting two normalized machine quantities over the job dimension:

1. assigned queue length: number of operations already assigned to the machine divided by the total number of operations in the instance;
2. cumulative processing load: sum of normalized processing times assigned to the machine divided by the total expected normalized workload of the instance.

The features are machine-specific but repeated for all candidate-operation pairs involving that machine. They enter the existing pair MLP and pair-conditioned attention without changing the action space or message-passing topology.

Required code areas: `env/FJSPEnv.py`, `params.py`, and no architectural change beyond the pair-MLP input dimension.

Primary risks:

- in constructive scheduling, queue length means the number of operations already appended to a machine sequence, not the number of physically waiting operations at simulated time;
- cumulative processing load overlaps with the existing utilization and machine-ready features;
- instance-wise z-score normalization must not erase the absolute progress signal requested by the reviewer.

Acceptance checks: both features start at zero; only the selected machine changes after a step; values remain finite and within the intended normalized range; the first six pair features are bitwise unchanged; parameter-count and inference-time changes are recorded.

## R1.3: urgency-conditioned history update

Feasibility: `STRAIGHTFORWARD_WITH_REDUNDANCY_RISK`

The current `GRUCell` is already input-conditioned internally. The defensible interpretation of the comment is therefore not that the GRU lacks gates, but that the model lacks an explicit decision-dependent control over how much of the previous scheduling history is retained.

### Minimum candidate B0+G

Keep the existing GRU candidate update and add an explicit learned history-update intensity derived from the current graph representation:

```text
h_candidate = GRUCell(h_graph, h_previous)
g_urgency = sigmoid(MLP_urgency(h_graph))
h_new = h_previous + g_urgency * (h_candidate - h_previous)
```

Use a vector gate of size `hist_dim`, which allows different history channels to update at different rates. The graph representation already embeds remaining workload, delay ratio, and criticality, so the gate is conditioned on decision urgency without adding a hand-tuned urgency formula.

The same update helper must be called by both `forward` and `forward_sequence`; otherwise rollout and PPO recomputation would use different policies.

Required code area: `model/BiGraphNetwork.py` only, plus a new revision-variant switch in `params.py`.

Primary risks:

- the extra gate may be redundant with the internal GRU update gate;
- a gate biased toward zero can impede learning early in training;
- an unconstrained implementation can accidentally differ between rollout, beam inference, and BPTT recomputation.

Acceptance checks: initialize the final gate bias so updates are initially allowed; use one shared helper in all execution paths; verify hidden-state reset at episode boundaries; compare gradient flow and training stability with B0.

## R1.4: long-horizon credit assignment and load-aware shaping

Feasibility: `FEASIBLE_IF_OBJECTIVE_PRESERVATION_IS_ENFORCED`

Adding an ordinary utilization bonus would change the optimization target and could make the response scientifically weaker. The minimum candidate therefore uses potential-based shaping whose total contribution is zero over every complete episode.

### Minimum candidate B0+R

Define progress as `p_t = scheduled_operations / total_operations` and a normalized machine-load imbalance measure `D_t`. Use the potential

```text
Phi(s_t) = -4 * p_t * (1 - p_t) * D_t
```

and shape the B0 reward as

```text
r'_t = r_t + beta * (Phi(s_{t+1}) - Phi(s_t)).
```

Because `Phi=0` at both `p=0` and `p=1`, the shaping terms telescope to zero in every complete episode. The final scheduling objective is therefore unchanged when `gamma=1`. `D_t` should be the standard deviation of machine processing-load fractions, not machine ready time, so idle periods do not masquerade as processing load.

Use one fixed `beta` selected before the confirmatory run. A short numerical sanity run may be used only to prevent a scale error, not to select the best test performance.

Required code areas: `env/FJSPEnv.py` and `params.py`.

Primary risks:

- if terminal potential is not exactly zero, the shaping changes the episode objective;
- using true final information would leak future outcomes;
- an excessively large `beta` can increase variance even though the total return is preserved;
- the reviewer explicitly mentions sparse milestones, whereas this candidate addresses the same credit-assignment concern through dense load-aware potential shaping.

Acceptance checks: for complete trajectories, verify numerically that the sum of shaping terms is zero within tolerance; confirm that the B0 reward is bitwise unchanged when the variant is disabled; verify no future schedule information is used; log both base and shaping rewards separately during debugging.

## Fair-comparison controls

- Add a dedicated `--revision_variant` switch with choices `b0`, `q`, `l`, `g`, and `r`; do not overload the existing ablation switch.
- Use `--ablation none` for all revision experiments.
- Change only the code required by one reviewer item in each individual variant.
- Keep SD3, 10 jobs, 5 machines, 1,000 updates, seed 300, optimizer, PPO settings, validation data, checkpoint selection rule, and stochastic beam width 10 fixed.
- Evaluate one checkpoint per configuration on the same ordered 130-instance `BenchData` collection.
- Retain per-instance output so every comparison with B0 is paired.
- Use additional training seeds only when the seed-300 outcome is close to a decision threshold, visibly unstable, or used to support a stability claim.

## Implementation order

The recommended order is B0, B0+L, B0+G, B0+Q, and B0+R. B0+L and B0+G have the smallest implementation surface and can validate the experiment harness before the more invasive state-interface and reward changes.
