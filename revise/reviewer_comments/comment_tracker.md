# Comment-Response Tracker

Package readiness: `draft_with_placeholders`; manuscript and response drafts exist, but final page and line references remain pending.

| ID | Reviewer concern | Category | Severity | Proposed action | Evidence or manuscript location | Status |
|---|---|---|---|---|---|---|
| R1.1 | Information loss caused by retaining only one candidate operation per job | Method and evidence | Major | Report the tested waiting-operation gated aggregation and its non-retention; clarify the information represented by the existing state without claiming the tested alternative is generally ineffective | Section 3.1, Conclusion, Appendix A, and response evidence | RESPONSE_DRAFTED |
| R1.2 | Lack of explicit machine queue-length and cumulative-load pair features | Method and evidence | Major | Report the tested normalized queue-length and cumulative-load features and their non-retention; clarify how current features encode load-related information | Section 3.1, Conclusion, Appendix A, and response evidence | RESPONSE_DRAFTED |
| R1.3 | Fixed GRU update may retain irrelevant historical information | Method and evidence | Major | Report the tested graph-conditioned gate and its non-retention; clarify the current GRU update and soften any unsupported history-selection claim | Section 3.3, Conclusion, and response evidence | RESPONSE_DRAFTED |
| R1.4 | Reward design and long-horizon credit assignment | Method and evidence | Major | Report the endpoint-zero load-aware potential comparison and its non-retention; clarify the dense lower-bound reward and acknowledge the remaining long-horizon limitation | Section 3.1, Conclusion, and response evidence | RESPONSE_DRAFTED |

## Status vocabulary

- `PENDING_EXPERIMENT`: implementation or evaluation is incomplete.
- `AUTHOR_INPUT_NEEDED`: factual information must be supplied by an author.
- `RESULT_VERIFIED`: outputs and settings have been checked.
- `MANUSCRIPT_UPDATED`: the supported change has been inserted into the manuscript.
- `RESPONSE_READY`: the response is traceable to evidence and manuscript changes.
