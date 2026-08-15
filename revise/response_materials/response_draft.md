# Response Draft

Package readiness: `experiment_evidence_complete_response_and_manuscript_pending`

This file is reserved for the final point-by-point response. Each response must contain the preserved reviewer comment, the evidence-based answer, the exact manuscript change, and a traceable section or line location.

Do not state that an experiment or manuscript revision has been completed while the corresponding tracker item remains `PENDING_EXPERIMENT` or `AUTHOR_INPUT_NEEDED`.

## Reviewer 1

### R1.1

Reviewer comment: The current compact bipartite graph only retains one candidate operation for each job. It is recommended to add implicit encoding for suboptimal operations in the waiting queue (such as through gate aggregation mechanism) to avoid decision information loss due to excessive compression, especially in operation intensive Brandimarte instances.

Response: `RESULT_AVAILABLE_RESPONSE_PENDING`. At seed 300, B0+Q reduces the Brandimarte mean Gap by 0.826 percentage points, below the frozen 1-point threshold, while increasing the edata and vdata mean Gaps by 1.882 and 1.128 points. Because this result is threshold-adjacent, seeds 301 and 302 were added. Their Brandimarte Gaps are 23.05% and 27.64%, and the three-seed mean is 23.86%, compared with 21.70% for frozen B0. The tested component is therefore not retained. The final response must present this as evidence about the tested implementation, not as a rejection of all waiting-operation aggregation methods.

Changes in the manuscript: `RESULT_AVAILABLE_MANUSCRIPT_PENDING`

Location: Sections 3.1, 4.2, and 4.3; final page and line numbers pending.

### R1.2

Reviewer comment: The current 6-dimensional paired features lack an explicit expression of the machine load balancing state. It is recommended to add normalized features of the current queue length and cumulative processing time of the machine, so that the attention mechanism can more accurately evaluate the global impact of operation machine matching.

Response: `RESULT_AVAILABLE_RESPONSE_PENDING`. B0+L increases the Brandimarte mean Gap by 6.717 percentage points and worsens every Hurink subset. The tested explicit load features are therefore not retained.

Changes in the manuscript: `RESULT_AVAILABLE_MANUSCRIPT_PENDING`

Location: Sections 3.1 and 4; final page and line numbers pending.

### R1.3

Reviewer comment: The current GRU integrates graph level representations in a fixed manner. It is recommended to introduce attention gating mechanism to enable adaptive adjustment of historical state updates based on the urgency of current decisions (such as the urgency of remaining project duration), reducing the interference of irrelevant historical information.

Response: `RESULT_AVAILABLE_RESPONSE_PENDING`. At seed 300, B0+G reduces the Brandimarte mean Gap by 0.818 percentage points, below the frozen 1-point threshold, and increases the edata mean Gap by 1.958 points. Because this result is threshold-adjacent, seeds 301 and 302 were added. Their Brandimarte Gaps are 23.10% and 33.45%, and the three-seed mean is 25.81%, compared with 21.70% for frozen B0. The tested gate is therefore not retained, and the final text should clarify rather than overstate the selectivity of the existing GRU.

Changes in the manuscript: `RESULT_AVAILABLE_MANUSCRIPT_PENDING`

Location: Sections 3.3 and 4; final page and line numbers pending.

### R1.4

Reviewer comment: The current reward is based on changes in the lower bound completion time. It is recommended to add sparse milestone rewards (such as reaching a certain threshold for the first time) and multi-objective reward components (such as machine utilization) to alleviate the credit allocation problem of PPO in long sequence scheduling.

Response: `RESULT_AVAILABLE_RESPONSE_PENDING`. B0+R increases the Brandimarte mean Gap by 1.490 percentage points and worsens all three Hurink subsets. The tested potential-based shaping is therefore not retained. Its inference-time measurement is invalid because of system suspension, but this does not affect the quality-based decision.

Changes in the manuscript: `RESULT_AVAILABLE_MANUSCRIPT_PENDING`

Location: Sections 3.1, 3.4, and 4; final page and line numbers pending.
