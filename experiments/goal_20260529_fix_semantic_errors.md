# 目标：修复两处语义错误，验证对 makespan gap 的影响

> **配套手册**：[../AUTO_EXPERIMENT.md](../AUTO_EXPERIMENT.md)
> **基准参照**：[baseline.md](baseline.md)
> **修改范围**：`model/BiGraphNetwork.py`（Exp 001）、`model/ppo.py`（Exp 002）
> **禁区文件**：`env/FJSPEnv.py` 中发现的 Exp 003/004 问题需走独立零号基准流程，见第 6 节

---

## 1. Hypothesis（假设）

代码审查发现以下两处**可改区内**的语义错误，修复后预期改善全局图表示质量：

| # | 文件 | 问题摘要 | 预期影响 |
|---|------|---------|---------|
| 001 | `model/BiGraphNetwork.py:172` | `nonzero_averaging` 依赖"零输入→零输出"假设，但 MLP bias 导致完工 job 的 embedding 不为零，函数实际上对全部 J 个 job（含已完工）取均值，而非只对活跃 job | 全局 context `h_j_global` 质量下降，尤其在 episode 后期；修复后预期 gap 改善 |
| 002 | `model/ppo.py:88` | `get_gae_advantages` 存储了 `done_seq` 但未在 advantage 递推中乘 `(1-done)`；当前由于 `flag_same_opes=True` 所有 env 同步结束故不影响训练，但语义不完整 | 当前场景预期**无回归**，代码语义完整性提升；若未来支持不等长 batch 则必须修复 |

两个实验**独立进行、顺序执行**，Exp 002 在 Exp 001 结束后才开始。

---

## 2. 主指标（Primary metric）

- **名称**：`gap mean (%)` on SD2 测试集（30×10+mix 与 40×10+mix 的均值），越低越好
- **当前 baseline 值**（来源：baseline.md 第 1.1 节，2026-05-29）：

| 数据集 | makespan | gap mean | gap std |
|-------|---------|---------|---------|
| SD2 30×10+mix | 806.97 | **16.64%** | 4.84% |
| SD2 40×10+mix | 1012.28 | **3.52%** | 10.51% |
| SD1 30×10 | 330.98 | **20.44%** | 5.87% |
| SD1 40×10 | 541.40 | **47.87%** | 7.19% |
| Brandimarte | 254.70 | **37.88%** | 28.43% |
| Hurink_edata | 1346.65 | **27.11%** | 18.31% |
| Hurink_rdata | 1066.70 | **13.06%** | 7.24% |
| Hurink_vdata | 1044.30 | **12.44%** | 10.00% |

- **裁决阈值**：SD2 主指标（16.64% 与 3.52% 的均值 ≈ 10.08%）改善 ≥ 1 pp，且 SD1/BenchData 无明显回归（gap 上升 < 1 pp）

---

## 3. 辅助指标（仅观察，不参与裁决）

- 验证集 makespan 收敛曲线（TensorBoard `train/vali_makespan_mean`）
- 训练 episode reward 均值（`train/episode_reward_mean`）
- 单次测试推理时间（test_results 中 `time` 列的均值）

---

## 4. 前置准备

### 4.1 备份基准模型

```bash
# 确认基准模型存在
ls ./trained_network/

# 将其复制为独立备份（名称中含 baseline 标记，防止被覆盖）
cp ./trained_network/10x5.pth ./trained_network/10x5_baseline.pth
```

### 4.2 确认测试命令（首次执行时验证）

```bash
# 查看 data/ 下 SD1、SD2、BenchData 的实际子目录结构，确认 test_data 参数值
ls ./data/SD1/
ls ./data/SD2/
ls ./data/BenchData/

# 确认 solution CSV 存在
ls ./data/SD1/SD1Solution.csv
ls ./data/SD2/SD2Solution.csv
ls ./data/BenchData/BenchDataSolution.csv
```

标准测试命令：

```bash
python test.py \
  --test_model 10x5 \
  --test_data SD1 SD2 BenchData \
  --cover_flag True \
  --seed_test 50
```
---

## 5. Exp 001：修复 `nonzero_averaging`（model/BiGraphNetwork.py）

### 5.1 问题描述

`nonzero_averaging`（第 172–178 行）本意是只对"有非零特征的活跃 job"取均值：
```python
# 当前代码（有缺陷）
def nonzero_averaging(self, x):
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)   # 依赖零输入→零输出
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)
```

实际问题：完工 job 在 `fea_j` 中被置零，但经过带 bias 的 `job_mlp` 后输出不为零；
`count_nonzero` 对所有 job 均返回非零，`z` 始终等于 `J`，函数退化为普通均值（含完工 job）。

`dynamic_pair_mask`（已传入 `forward`）可直接推导活跃 job 集合：
完工 job 的所有 (job, machine) 对均被标记为不可行（全行为 True），
故 `~dynamic_pair_mask.all(dim=-1)` 可精确给出活跃 job 的 mask。

### 5.2 代码变更（`model/BiGraphNetwork.py`）

**Step A — 替换方法定义（第 172–178 行）：**

删除：
```python
def nonzero_averaging(self, x):
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)
```

替换为：
```python
def active_job_mean(self, x, job_valid):
    """Compute mean over active (non-completed) jobs only.
    job_valid: [B, J] bool tensor, True = job still has unscheduled operations.
    """
    count = job_valid.float().sum(dim=-1, keepdim=True).clamp_min(1)  # [B, 1]
    return (x * job_valid.unsqueeze(-1).float()).sum(dim=-2) / count   # [B, d]
```

**Step B — 更新 `forward()` 中的调用（第 123–124 行）：**

删除：
```python
h_j_global = self.nonzero_averaging(_h_j)
h_m_global = self.nonzero_averaging(_h_m)
```

替换为：
```python
job_valid = ~dynamic_pair_mask.all(dim=-1)       # [B, J]: True = active job
h_j_global = self.active_job_mean(_h_j, job_valid)
h_m_global = _h_m.mean(dim=-2)                   # machines are always all active
```

### 5.3 执行步骤

```bash
# Step 1: 确认工作区干净
git status   # 必须 clean

# Step 2: 应用代码变更（按 5.2 节手动编辑 model/BiGraphNetwork.py）

# Step 3: 训练（标准预算 500 updates）
python train.py --n_j 10 --n_m 5 \
    --max_updates 500 --seed_train 300 \
    --run_name exp_001_fix_nonzero_avg --log_dir ./runs

# Step 4: 测试
python test.py \
    --test_model 10x5 \
    --test_data SD1 SD2 BenchData \
    --cover_flag True --seed_test 50

# Step 5: 记录结果 → 填写第 9 节实验日志

# Step 6-A（若改善 ≥ 1 pp）: keep
git add model/BiGraphNetwork.py
git add results.tsv
git commit -m "[keep] exp#001: gap XX.XX→YY.YY | fix nonzero_averaging with active_job_mean"

# Step 6-B（若无改善或回归）: revert
git add results.tsv
git commit -m "[log] exp#001: revert (gap XX.XX→YY.YY, no improvement)"
git reset --hard HEAD~1   # 仅回滚代码，results.tsv 已先 commit
```

### 5.4 裁决标准

| 条件 | 决策 |
|------|------|
| SD2 gap mean 改善 ≥ 1 pp，且 SD1/BenchData 无 > 1 pp 回归 | **keep** |
| SD2 gap mean 无改善，但 SD1 或 BenchData 有改善 ≥ 1 pp | **keep**（注明分析） |
| 所有数据集均无改善或任意数据集回归 > 2 pp | **revert** |

---

## 6. Exp 002：修复 GAE 忽略 `done` 标志（model/ppo.py）

> **前置条件**：Exp 001 结束后（无论 keep/revert）再执行 Exp 002。

### 6.1 问题描述

`get_gae_advantages`（第 75–101 行）在 advantage 递推时未乘 `(1 - done_t)`，
标准 GAE 公式要求在终止步截断 bootstrap：

```
A_t = δ_t + (γλ) · A_{t+1} · (1 - done_t)
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
```

当前代码中 `done_seq` 已存入 memory 但未被使用。在 `flag_same_opes=True` 的默认设置下，
所有 env 在同一时刻结束（`done` 仅在最后一步变 True，已走特殊分支），故不影响当前训练。
修复的意义在于：代码语义完整、为支持不等长 batch 打好基础。

### 6.2 代码变更（`model/ppo.py`）

**在 `get_gae_advantages` 方法开头（第 76–78 行之后）添加 `done_arr`：**

定位到：
```python
def get_gae_advantages(self):
    reward_arr = torch.stack(self.reward_seq, dim=0)
    values = torch.stack(self.val_seq, dim=0)

    len_trajectory, len_envs = reward_arr.shape
```

在 `len_trajectory, len_envs = ...` 之后新增一行：
```python
    done_arr = torch.stack(self.done_seq, dim=0).float()  # [T, B], 1.0 = done
```

**替换 advantage 递推循环（第 83–88 行）：**

删除：
```python
    for i in reversed(range(len_trajectory)):
        if i == len_trajectory - 1:
            delta_t = reward_arr[i] - values[i]
        else:
            delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
        advantage = delta_t + self.gamma * self.gae_lambda * advantage
```

替换为：
```python
    for i in reversed(range(len_trajectory)):
        not_done = 1.0 - done_arr[i]                       # [B]: 0.0 if env finished at step i
        if i == len_trajectory - 1:
            delta_t = reward_arr[i] - values[i]
        else:
            delta_t = (reward_arr[i]
                       + self.gamma * values[i + 1] * not_done
                       - values[i])
        advantage = delta_t + self.gamma * self.gae_lambda * advantage * not_done
```

### 6.3 执行步骤

```bash
# Step 1: 确认工作区干净（Exp 001 已 keep 或 revert 完毕）
git status

# Step 2: 应用代码变更（按 6.2 节手动编辑 model/ppo.py）

# Step 3: 训练
python train.py --n_j 10 --n_m 5 \
    --max_updates 500 --seed_train 300 \
    --run_name exp_002_fix_gae_done --log_dir ./runs

# Step 4: 测试
python test.py \
    --test_model 10x5 \
    --test_data SD1 SD2 BenchData \
    --cover_flag True --seed_test 50

# Step 5: 记录结果 → 填写第 9 节实验日志

# Step 6-A（改善或无回归）: keep
git add model/ppo.py results.tsv
git commit -m "[keep] exp#002: gap XX.XX→YY.YY | fix GAE done mask in get_gae_advantages"

# Step 6-B（回归 > 1 pp）: revert
git add results.tsv
git commit -m "[log] exp#002: revert (gap regression)"
git reset --hard HEAD~1
```

### 6.4 裁决标准

| 条件 | 决策 |
|------|------|
| 所有数据集回归均 < 1 pp（等同于"无影响"） | **keep**（代码正确性改善，无副作用） |
| 任意数据集回归 > 1 pp | **revert**（当前设置下不适用，暂不引入） |

---

## 7. 禁区内的已知问题（需独立零号基准流程）

以下两处错误位于 `env/FJSPEnv.py`（AUTO_EXPERIMENT.md 第 2.1 节禁区），
**不在本实验范围内**，需按禁区修改流程单独处理：

### Exp 003（低优先级）：`rem_work` 对已完工 job 返回最后一道工序的均值而非 0

- **位置**：`env/FJSPEnv.py:296–298`
- **现状**：完工 job 的 `candidate` 指向 `last_op_id`，`rem_work` 切片长度为 1，返回最后工序的 `op_mean_pt`
- **为何无影响**：第 308 行 `np.where(mask, 0, fea_j)` 将完工 job 的所有特征置零，错误值被覆盖
- **修正方向**：在计算 `rem_work` 时，完工 job（`self.mask[env_idx, job_idx] == True`）直接 append `0`
- **禁区处理步骤**：修改 `env/FJSPEnv.py` → 重新跑 3 次 baseline → 更新 baseline.md → 后续实验基于新 baseline

### Exp 004（中优先级）：`step()` 中 `actions` 与 `active_idx` 长度潜在不匹配

- **位置**：`env/FJSPEnv.py:193–197`
- **现状**：`actions` 在训练时有 B 个元素（全量），`active_idx` 仅含未完成的 env（A 个）；`self.candidate[active_idx, active_job]` 中 `active_job` 长度为 B，而 `active_idx` 长度为 A，当 A < B 时会 IndexError
- **为何无影响**：`flag_same_opes=True` 保证同一 batch 内所有 env 操作总数相同，A 始终等于 B
- **修正方向**：训练主循环（`train.py`）中增加 `batch_idx = ~done` 过滤，与验证代码保持一致；或在 `step()` 内用 `active_job = actions[active_idx] // n_mch` 取活跃子集
- **禁区处理步骤**：同上

---

## 8. 停止条件

- ✅ Exp 001 + Exp 002 均完成（keep 或 revert），本目标结束
- ✅ 若 Exp 001 keep 且 SD2 gap 改善 ≥ 2 pp，进入**确认阶段**（500 → 1000 updates 完整训练），以 1000 updates 结果更新 baseline.md
- ❌ 不在本目标内无限迭代超参；超参调优请另开 goal 文件

---

## 9. 结论（目标结束后填写）

- **Exp 001 结论**：（假设成立 / 不成立，最优配置，commit SHA）
- **Exp 002 结论**：（假设成立 / 不成立）
- **是否需要更新 baseline.md**：（是 / 否，原因）
- **产生的新假设**：（如有）

---

## 10. 实验日志

| exp_id | 改动 | SD2 30×10 gap | SD2 40×10 gap | SD2 均值 | vs baseline | 决策 | commit |
|--------|------|--------------|--------------|---------|-------------|------|--------|
| 001 | fix nonzero_averaging → active_job_mean | | | | | keep/revert | |
| 002 | fix GAE done mask | | | | | keep/revert | |
