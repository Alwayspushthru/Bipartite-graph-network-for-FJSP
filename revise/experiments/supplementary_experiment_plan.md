# Supplementary Experiment Plan

## Objective

Assess the feasibility and empirical effect of waiting-operation aggregation, explicit machine-load features, urgency-conditioned memory, and reward shaping on operation-intensive Brandimarte instances and the Hurink subsets. These configurations are candidate responses to the reviewer comments rather than changes assumed to be effective in advance.

Detailed code feasibility and minimum designs are recorded in `code_feasibility_analysis.md`. The frozen reference configuration and commands are recorded in `b0_frozen_protocol.md`.

## Configurations

| Label | Configuration | Reviewer item |
|---|---|---|
| B0 | Current full model | Reference |
| B0+Q | Gated fusion of a fixed-size waiting-operation summary | R1.1 |
| B0+L | Two explicit pair features for assigned queue length and cumulative processing load | R1.2 |
| B0+G | Graph-conditioned gate over the GRU history-update increment | R1.3 |
| B0+R | Endpoint-zero, load-aware potential-based reward shaping | R1.4 |
| B0+Best | Optional combination of individually supported components | Overall validation |

## Fixed protocol

- Training distribution: SD3, 10 jobs and 5 machines.
- Training budget: 1,000 updates for the confirmatory runs.
- Primary training seed: 300.
- Stability seeds: additional seeds are used only when an outcome requires confirmation, for example when the effect is close to the retention threshold, the run is unstable, or the conclusion materially depends on a single-seed result.
- Test sets: Brandimarte and Hurink edata, rdata, and vdata.
- Decoding: stochastic beam search with beam width 10.
- Primary metric: mean relative Gap on Brandimarte.
- Secondary metrics: median Gap, Gap standard deviation, inference time, parameter count, and the three Hurink mean Gaps.
- Statistical comparison: paired instance-level Gap differences, paired Wilcoxon signed-rank test, and a 95% bootstrap confidence interval for the mean difference.

## Retention rule

A component is eligible for `B0+Best` when it reduces the Brandimarte mean Gap by at least 1 percentage point, does not increase the mean Gap of any Hurink subset by more than 1 percentage point, and does not increase inference time by more than 15%. A result close to these thresholds or showing abnormal variance should be checked with additional seeds before a stability claim is made. `B0+Best` is run only when at least one component is supported. Final decisions must use verified results rather than preliminary training curves.

## Outcome handling

- If a candidate is feasible and supported by verified results, incorporate the supported design and report the evidence.
- If a candidate is feasible but does not improve performance, report the negative result honestly and decide whether clarification of the existing design, claim softening, or a limitation is the appropriate manuscript response.
- If the suggested mechanism cannot be implemented or evaluated reliably, document the technical reason and use a partial, scope-limited, or claim-softening response only when justified by the evidence.
- Do not decide the response category before the feasibility check and experiment are complete.

## Required records

For every run, record the commit identifier, configuration label, seed, command, start and end times, checkpoint path, raw result path, decoding settings, and any deviation from the fixed protocol.

## Execution status

- B0: `RESULT_VERIFIED` for seed 300 on CPU. See `b0_frozen_protocol.md` and `../results/experiment_results.md`.
- B0+Q: `RESULT_VERIFIED_NOT_RETAINED`; threshold-adjacent seed-300 result checked with seeds 301 and 302 and not reproduced.
- B0+L: `RESULT_VERIFIED_NOT_RETAINED`.
- B0+G: `RESULT_VERIFIED_NOT_RETAINED`; threshold-adjacent seed-300 result checked with seeds 301 and 302 and not reproduced.
- B0+R: `RESULT_VERIFIED_NOT_RETAINED`; evaluation timing is invalid because of system suspension, but the quality criteria already determine non-retention.
- B0+Best: `NOT_RUN_NO_ELIGIBLE_COMPONENT`.

The complete statistics, integrity hashes, and decision rationale are recorded in `../results/experiment_results.md`.
