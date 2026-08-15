# Supplementary Experiment Results

Status: `ALL_SINGLE_COMPONENT_RESULTS_VERIFIED`; no component met the frozen retention rule, so `B0+Best` is `NOT_RUN_NO_ELIGIBLE_COMPONENT`.

All values below were recomputed from the named raw workbooks by `analyze_revision_experiments.py`. Negative paired differences favor the variant.

## Run registry

| Run ID | Configuration | Seed | Checkpoint | Raw result | Training record | Protocol deviation |
|---|---|---:|---|---|---|---|
| B0-S300-CPU-20260814 | B0 | 300 | `trained_network/revise_b0_s300.pth` | `test_results/BenchData/revise_b0_s300_Bgnn-SB10_20260814_222847.xlsx` | 1,000 updates; best validation mean makespan 508.66; 1,087.84 s | CPU used with author approval because CUDA and MPS were unavailable |
| B0Q-S300-CPU-20260814 | B0+Q | 300 | `trained_network/revise_b0q_s300.pth` | `test_results/BenchData/revise_b0q_s300_Bgnn-SB10_20260814_230136.xlsx` | 1,000 updates; best validation mean makespan 506.96; 1,216.10 s | Same approved CPU environment |
| B0L-S300-CPU-20260814 | B0+L | 300 | `trained_network/revise_b0l_s300.pth` | `test_results/BenchData/revise_b0l_s300_Bgnn-SB10_20260814_232721.xlsx` | 1,000 updates; best validation mean makespan 507.37; 1,356.52 s | Same approved CPU environment |
| B0G-S300-CPU-20260814 | B0+G | 300 | `trained_network/revise_b0g_s300.pth` | `test_results/BenchData/revise_b0g_s300_Bgnn-SB10_20260814_235336.xlsx` | 1,000 updates; best validation mean makespan 510.14; 1,457.44 s | Same approved CPU environment |
| B0R-S300-CPU-20260815 | B0+R | 300 | `trained_network/revise_b0r_s300.pth` | `test_results/BenchData/revise_b0r_s300_Bgnn-SB10_20260815_034932.xlsx` | 1,000 updates; best validation mean makespan 510.93; wall time 13,092.57 s | Same CPU environment; system suspension inflated wall-clock training and evaluation time |
| B0Q-S301-CPU-20260815 | B0+Q | 301 | `trained_network/revise_b0q_s301.pth` | `test_results/BenchData/revise_b0q_s301_Bgnn-SB10_20260815_122417.xlsx` | 1,000 updates; best validation mean makespan 505.31; 1,140.11 s | Stability run triggered by seed-300 proximity to the Brandimarte threshold |
| B0Q-S302-CPU-20260815 | B0+Q | 302 | `trained_network/revise_b0q_s302.pth` | `test_results/BenchData/revise_b0q_s302_Bgnn-SB10_20260815_130607.xlsx` | 1,000 updates; best validation mean makespan 506.70; 1,221.89 s | Stability run triggered by seed-300 proximity to the Brandimarte threshold |
| B0G-S301-CPU-20260815 | B0+G | 301 | `trained_network/revise_b0g_s301.pth` | `test_results/BenchData/revise_b0g_s301_Bgnn-SB10_20260815_132743.xlsx` | 1,000 updates; best validation mean makespan 509.65; 1,161.20 s | Stability run triggered by seed-300 proximity to the Brandimarte threshold |
| B0G-S302-CPU-20260815 | B0+G | 302 | `trained_network/revise_b0g_s302.pth` | `test_results/BenchData/revise_b0g_s302_Bgnn-SB10_20260815_135157.xlsx` | 1,000 updates; best validation mean makespan 511.11; 1,259.10 s | Stability run triggered by seed-300 proximity to the Brandimarte threshold |

Shared protocol: SD3 10 jobs x 5 machines, 1,000 updates, seed 300, and stochastic beam search with width 10 on all 130 BenchData instances. The implementation state is identified by the source-file hashes recorded below because the working tree was not committed between variants.

## Confirmatory results

Each Gap cell reports mean / median / population standard deviation.

| Configuration | Brandimarte Gap | edata Gap | rdata Gap | vdata Gap | Time per instance | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 21.70 / 17.31 / 17.92% | 13.44 / 14.83 / 6.43% | 8.80 / 8.27 / 5.74% | 3.51 / 3.39 / 2.05% | 0.5188 s | 550,162 |
| B0+Q | 20.88 / 13.38 / 17.59% | 15.32 / 15.85 / 9.43% | 8.36 / 7.56 / 6.02% | 4.64 / 4.19 / 2.54% | 0.5413 s | 600,466 |
| B0+L | 28.42 / 17.31 / 26.02% | 17.87 / 18.43 / 10.46% | 11.84 / 10.24 / 10.06% | 10.23 / 3.47 / 15.08% | 0.6574 s | 550,418 |
| B0+G | 20.89 / 14.38 / 18.78% | 15.40 / 17.03 / 5.75% | 9.75 / 8.64 / 6.41% | 3.80 / 3.21 / 2.19% | 0.6331 s | 557,458 |
| B0+R | 23.19 / 19.81 / 19.76% | 19.95 / 20.40 / 9.38% | 12.50 / 10.29 / 8.57% | 4.93 / 4.24 / 3.13% | INVALID | 550,162 |
| B0+Best | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT APPLICABLE |

The B0+R timing value in the raw workbook is 14.5083 s per instance, but it is excluded because system suspension occurred during evaluation. B0+R already fails the quality criteria, so this invalid timing does not affect its retention decision.

## Stability confirmation for threshold-adjacent variants

Each entry reports the mean Gap for seeds 300 / 301 / 302, followed by the across-seed mean and population standard deviation. B0 remains the frozen seed-300 reference; these extra runs test whether the candidate improvement itself is reproducible and are not a matched multi-seed significance comparison against B0.

| Configuration | Brandimarte | edata | rdata | vdata | Mean time |
|---|---:|---:|---:|---:|---:|
| B0+Q | 20.88 / 23.05 / 27.64; 23.86 ± 2.82% | 15.32 / 13.55 / 16.40; 15.09 ± 1.17% | 8.36 / 10.31 / 14.63; 11.10 ± 2.62% | 4.64 / 6.27 / 7.79; 6.23 ± 1.29% | 0.5409 s |
| B0+G | 20.89 / 23.10 / 33.45; 25.81 ± 5.48% | 15.40 / 13.74 / 23.72; 17.62 ± 4.37% | 9.75 / 10.00 / 17.76; 12.50 ± 3.72% | 3.80 / 6.46 / 9.27; 6.51 ± 2.23% | 0.5748 s |

## Statistical analysis

The paired analysis covers the 10 Brandimarte instances. Confidence intervals use 10,000 paired bootstrap resamples. Exact two-sided Wilcoxon signed-rank p-values are Holm-adjusted across the four comparisons.

| Comparison | Mean paired difference | 95% bootstrap CI | Wilcoxon statistic | Adjusted p-value | Retention decision |
|---|---:|---|---:|---:|---|
| B0+Q vs B0 | -0.826 pp | [-5.290, 3.620] pp | 15.0 | 1.000 | NOT RETAINED: improvement below 1 pp; edata and vdata regress by more than 1 pp |
| B0+L vs B0 | +6.717 pp | [1.523, 12.672] pp | 2.0 | 0.375 | NOT RETAINED: Brandimarte and all Hurink subsets regress |
| B0+G vs B0 | -0.818 pp | [-3.576, 1.949] pp | 9.0 | 1.000 | NOT RETAINED: improvement below 1 pp; edata regresses by more than 1 pp |
| B0+R vs B0 | +1.490 pp | [-0.022, 3.166] pp | 9.0 | 0.750 | NOT RETAINED: Brandimarte and all Hurink subsets regress |

## Verified findings and scope

- Every checkpoint loads successfully, every evaluation covers all 130 instances, and the result analysis found no missing values.
- None of the four candidates satisfies the frozen retention rule. Consequently, no supported component exists for B0+Best, and the combined run is intentionally omitted.
- Because the seed-300 Brandimarte effects of Q and G were threshold-adjacent, both were repeated with seeds 301 and 302. Neither improvement reproduced: the three-seed Brandimarte means are 23.86% for Q and 25.81% for G, compared with 21.70% for frozen B0.
- L and R are single-seed architecture-screening results because their seed-300 quality regressions are broad and not threshold-adjacent. The evidence supports non-retention of the tested variants, not a broad claim that all possible aggregation, load, gating, or shaping designs are ineffective.

## Integrity hashes

| Artifact | SHA-256 |
|---|---|
| B0 checkpoint | `3710abc86c61a745d517ba3fcc7891410752cf63722a3c5e4b087de8623f30a8` |
| B0+Q checkpoint | `f4be8c302ac4d8611ebedc6d901854ee957d73d901d4e4c68b8cea7196c9d476` |
| B0+L checkpoint | `3280eb4bcf56c129ec31a78ddef66ad8274d692a4eefe35a45042da8973f3da9` |
| B0+G checkpoint | `305560e8ae4c7156189ed7a9750bafafb91b15280a00340880d95ae80575c275` |
| B0+R checkpoint | `7d37756b960618d9640a0ff48a3758dbe16c7f54e873fcb3507a96d04282a162` |
| B0+Q seed-301 checkpoint | `af279c9214d6da954566aed215e3caf48b3adddd30bf1a5d20224dfbc53af8e8` |
| B0+Q seed-302 checkpoint | `3c8f14a78803dfa8976e0ed88af8e1051da56494f0cb6afe1bf8f077bd42a2f2` |
| B0+G seed-301 checkpoint | `2370751530aef2dc42e4691ab7a0cfcfcbac51823075b925117367628d31cf82` |
| B0+G seed-302 checkpoint | `a1c0bead2d68949e5c7dbc9cb01561a8567618154d4809cf5c4fb0aec213b0ed` |
| B0 raw result | `e266b757667e1fc9f4faea8d9bf5b5c7d1401af71fae0ffc674903ae0dc0d5fa` |
| B0+Q raw result | `7ec6cd262358f78760322a678f3c5ed8930f3a88b11e19cf88ee782fad8d0b7c` |
| B0+L raw result | `3d75738d53e65c8a41ab04cee9858e9bc923e892a9fc71236418db97fea6b3aa` |
| B0+G raw result | `04c9f9af7efb52362eeb1c9dd924f6080acaae8ddc5dbf0824952adcbbaf81` |
| B0+R raw result | `a1f54915f107737c5b1355a5639ecfa99156e32696effb467c72ff5be39f8e77` |
| B0+Q seed-301 raw result | `02876177997953dac58a4c24e0725e156cd0083ca143af96b008c43da7a4a1e5` |
| B0+Q seed-302 raw result | `9a52e510647c14f5a50c55666cedc95946a75f1408ee953657a3bc0801d2eb1f` |
| B0+G seed-301 raw result | `bf196e744eff104ca57ba5cb4ec24fdcb503f866fa93715514779672fbad3526` |
| B0+G seed-302 raw result | `02522bf99d923ec526afb7e29dc377e847619eac44b0d73f92b9ab99e778b500` |

Shared source SHA-256 values: `params.py` `c97023568b132b1f0077d474614963e00e5e3576712713c8289fccc30f565558`; `train.py` `38278246709db71721c2c54d727293f095b616f618247c070bdf903a800c754b`; `test.py` `7fba424647a0deeba453b064339baefee8dd6b357b5a283d141ec86e9716dd90`; `env/FJSPEnv.py` `905a9e4ed77e370f9fa7c4d8e80d4a50f73c87db6688c07dbdb80ee4d893165e`; `model/BiGraphNetwork.py` `aef2867da5d9ee66f44c5edc4214a182240fda23f2ef4b67d2e70ad42cc962c5`; `model/ppo.py` `b117404ddda7252b7c970e9797da547848d91141c0646a7f5d339fb16b073a5a`; `utils/common_utils.py` `8c996e31b26d6a420ef15076334639729cbb661a5820708b387db637a43b0592`.
