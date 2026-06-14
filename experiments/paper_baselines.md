# 对比论文数据（Paper Baselines）

> **用途**：集中收录所有需要做对比的**外部论文 / 公开方法**在相同数据集上的报告结果，作为本项目模型的横向对照。
> **配套文件**：本项目自身基线见 [baseline.md](baseline.md)；实验记录见 `results.tsv`。
> **填写规则**：
> - 每篇论文一个小节，先填元信息（标题 / 出处 / 方法 / 链接），再按数据集填表。
> - 指标统一用 **gap mean (%)**（越低越好）；若论文只给 makespan，则在 makespan 列填值、gap 列留空并注明参考解来源。
> - 数据集 / 测试规模命名与 baseline.md 保持一致（SD1、SD2、Brandimarte、Hurink_edata/rdata/vdata），便于直接对照。
> - 数字无法对齐到本项目规模时，在该行备注说明（如论文用的实例集不同）。

---

## 论文索引

> 所有对比方法均为 DRL，故"方法类型"一列记录其**特征提取方式**（如 MLP / GNN / 二部图等），用于区分。

| 编号 | 简称 | 标题 | 年份 | 方法类型 | 链接 |
|------|------|------|------|----------|------|
| P1 | DRL-FJSP (2024) | Solving flexible job shop scheduling problems via deep reinforcement learning | 2024 | MLP 提取特征 | — |
| P2 | DANIEL (2024) | Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning | 2024 | 双注意力网络（Dual Attention） | — |
| P3 | [31] (2022) | Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning | 2022 | GNN 提取特征 | — |

---

## P1 · DRL-FJSP (2024)

- **标题**：Solving flexible job shop scheduling problems via deep reinforcement learning
- **出处 / 会议或期刊**：（待补）
- **方法**：DRL，**MLP 提取特征**
- **链接**：（待补）
- **备注**：下表 gap 为论文报告的 "Ours" 结果；Time 为论文报告的单实例平均推理耗时（秒）。SD1/SD2 论文未给出对应数字。

### SD1

| 测试规模 | makespan | gap mean |
|---------|---------:|---------:|
| 30×10 | — | — |
| 40×10 | — | — |

### SD2（mixed size）

| 测试规模 | makespan | gap mean |
|---------|---------:|---------:|
| 30×10 + mix | — | — |
| 40×10 + mix | — | — |

### BenchData 公开基准

| 数据集 | makespan | gap mean | Time (s) |
|-------|---------:|---------:|---------:|
| Brandimarte | — | **13.24%** | 0.406 |
| Hurink_edata | — | **15.54%** | 0.271 |
| Hurink_rdata | — | **12.09%** | 0.275 |
| Hurink_vdata | — | **5.37%** | 0.272 |

---

## P2 · DANIEL (2024)

- **标题**：Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning
- **出处 / 会议或期刊**：（待补）
- **方法**：DRL，**双注意力网络（Dual Attention Network）提取特征**
- **链接**：（待补）
- **备注**：模型在 10×5 规模上训练，下表为其在更大规模上的泛化结果；`Time (s)` 为论文报告的单实例平均推理耗时。

### SD1

| 测试规模 | makespan | gap mean | Time (s) |
|---------|---------:|---------:|---------:|
| 30×10 | — | **5.10%** | 2.78 |
| 40×10 | — | **3.65%** | 3.77 |

### SD2（mixed size）

| 测试规模 | makespan | gap mean | Time (s) |
|---------|---------:|---------:|---------:|
| 30×10 | — | **14.85%** | 2.80 |
| 40×10 | — | **0.52%** | 3.76 |

### BenchData 公开基准

| 数据集 | makespan | gap mean | Time (s) |
|-------|---------:|---------:|---------:|
| Brandimarte | — | **13.58%** | 1.29 |
| Hurink_edata | — | **16.33%** | 1.37 |
| Hurink_rdata | — | **11.42%** | 1.37 |
| Hurink_vdata | — | **3.28%** | 1.37 |

---

## P3 · [31] (2022)

- **标题**：Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning
- **出处 / 会议或期刊**：（待补）
- **方法**：DRL，**GNN 提取特征**
- **链接**：（待补）
- **备注**：模型在 10×5 规模上训练，下表为其在更大规模上的泛化结果；`Time (s)` 为论文报告的单实例平均推理耗时。

### SD1

| 测试规模 | makespan | gap mean | Time (s) |
|---------|---------:|---------:|---------:|
| 30×10 | — | **14.61%** | 2.86 |
| 40×10 | — | **14.21%** | 3.82 |

### SD2（mixed size）

| 测试规模 | makespan | gap mean | Time (s) |
|---------|---------:|---------:|---------:|
| 30×10 | — | **126.55%** | 2.93 |
| 40×10 | — | **109.87%** | 3.87 |

### BenchData 公开基准

| 数据集 | makespan | gap mean | Time (s) |
|-------|---------:|---------:|---------:|
| Brandimarte | — | **28.52%** | 1.26 |
| Hurink_edata | — | **15.53%** | 1.40 |
| Hurink_rdata | — | **11.15%** | 1.40 |
| Hurink_vdata | — | **4.25%** | 1.37 |

---

## 本项目当前基线 · SBeam×10

- **模型规模**：10×5
- **解码模式**：SBeam×10
- **实验记录**：`test_results/test_log.txt`，运行时间戳 20260604_092947 / 20260604_094058 / 20260604_095212
- **备注**：仅记录 makespan 与 gap mean；std 不纳入本文对比表。

### SD1

| 测试规模 | makespan | gap mean |
|---------|---------:|---------:|
| 30×10 | 288.75 | 5.15% |
| 40×10 | 384.48 | 5.08% |

### SD2（mixed size）

| 测试规模 | makespan | gap mean |
|---------|---------:|---------:|
| 30×10 + mix | 784.93 | 13.45% |
| 40×10 + mix | 984.96 | 0.68% |

### BenchData 公开基准

| 数据集 | makespan | gap mean |
|-------|---------:|---------:|
| Brandimarte | 205.00 | 20.67% |
| Hurink_edata | 1179.08 | 14.47% |
| Hurink_rdata | 1005.17 | 7.83% |
| Hurink_vdata | 941.38 | 2.59% |

---

## 横向对比汇总（待填）

> 各论文填完后，把关键 gap mean 汇总到此处，连同本项目"当前基线"一起对照。

### SD / gap mean (%)

> P2、P3 模型均在 10×5 规模训练，下表为大规模泛化下的 gap mean。

| 数据集 | 测试规模 | 本项目（SBeam×10） | P2 (DANIEL 2024) | P3 ([31] 2022) |
|-------|---------|------------------:|----------------:|---------------:|
| SD1 | 30×10 | 5.15% | **5.10%** | 14.61% |
| SD1 | 40×10 | 5.08% | **3.65%** | 14.21% |
| SD2 | 30×10 | **13.45%** | 14.85% | 126.55% |
| SD2 | 40×10 | 0.68% | **0.52%** | 109.87% |

### BenchData / gap mean (%)

| 数据集 | 本项目（SBeam×10） | P1 (DRL-FJSP 2024) | P2 (DANIEL 2024) | P3 ([31] 2022) |
|-------|------------------:|------------------:|----------------:|---------------:|
| Brandimarte | 20.67% | 13.24% | **13.58%** | 28.52% |
| Hurink_edata | **14.47%** | 15.54% | 16.33% | 15.53% |
| Hurink_rdata | **7.83%** | 12.09% | 11.42% | 11.15% |
| Hurink_vdata | **2.59%** | 5.37% | 3.28% | 4.25% |
