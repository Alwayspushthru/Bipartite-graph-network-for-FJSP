# 基于二部图网络的柔性作业车间调度（FJSP）

本项目使用**二部图注意力网络 + PPO 强化学习**求解柔性作业车间调度问题（Flexible Job Shop Problem, FJSP），训练数据由程序自动生成。

---

## 项目结构

```
├── env/FJSPEnv.py          # 调度环境（状态、特征提取、步进逻辑）
├── model/
│   ├── BiGraphNetwork.py   # 二部图注意力网络（工序↔机器双向消息传递）
│   ├── ppo.py              # PPO 训练器
│   └── sub_layers.py       # MLP、Actor、Critic 子层
├── utils/
│   ├── data_utils.py       # 数据生成（SD3CaseGenerator 等）
│   └── common_utils.py     # 公共工具函数
├── train.py                # 训练入口
├── test.py                 # 测试入口
└── params.py               # 所有超参数定义
```

---

## 训练数据生成（SD3）

训练数据由 `SD3CaseGenerator`（`utils/data_utils.py`）在线随机生成，无需提前准备数据文件。

### 默认规模

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_j` | 10 | 作业数 |
| `n_m` | 5 | 机器数 |
训练规模由参数进行变换，在本项目中会训练的规模为10x5,10x15,20x10

### 生成规则

**① 每个作业的工序数**

从 `[floor(0.8 × n_m), ceil(1.2 × n_m)]` 范围内随机取整数。默认（n_m=5）即每个作业 **4～6 道工序**。

同一批次（`num_envs=20` 个并行环境）共享同一组工序数配置，每次更新环境时重新采样。

**② 每道工序的可加工机器数（柔性度）**

从 `[1, n_m]` 中随机取整，再从所有机器中无重复随机抽取对应数量的机器分配给该工序。

**③ 各机器的加工时间**

先为每道工序随机生成平均加工时间 `mean = randint(1, 99)`，各兼容机器的实际加工时间在均值的 **±20%** 范围内随机取整：

```
low  = max(1,  round(mean × 0.8))
high = min(99, round(mean × 1.2))
actual = randint(low, high)
```

### 数据格式

每个实例第一行为 `作业数 机器数 平均柔性度`，后续每行描述一个作业的所有工序及其可加工机器与加工时间。

---

## 网络结构

`BiGraphLayer` 以工序节点（O）和机器节点（M）构成二部图，通过注意力机制双向传递消息：

- **O ← M**：工序节点聚合来自可加工机器的信息
- **M ← O**：机器节点聚合来自待加工工序的信息
- **配对特征（Pair）**：工序-机器对的边特征参与注意力计算并同步更新

堆叠 `num_bigraph_layers=2` 层后，Actor 输出动作概率，Critic 输出状态价值。

| 特征类型 | 维度 |
|----------|------|
| 工序特征 | 6 |
| 机器特征 | 4 |
| 工序-机器对特征 | 6 |
| 隐藏层维度 | 64 |

---

## 快速开始

### 训练

```bash
python train.py --train_size 20x10 --num_envs 20 --max_updates 1000
```

### 测试

```bash
python test.py --test_model 10x5_1 --test_data SD2
```

### 生成数据文件（可选）

```bash
python utils/data_utils.py   # 使用 seed=100 生成 100 个实例到 data/SD3/
```

---

## 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 20 | 并行训练环境数 |
| `max_updates` | 1000 | 训练更新轮数 |
| `lr` | 3e-4 | 学习率 |
| `k_epochs` | 4 | 每轮数据的 PPO 更新次数 |
| `eps_clip` | 0.2 | PPO 裁剪参数 |
| `gamma` | 1.0 | 折扣因子 |
| `gae_lambda` | 0.98 | GAE 参数 |
| `reset_env_timestep` | 20 | 环境重置（重新采样数据）间隔 |
| `validate_timestep` | 10 | 验证与日志记录间隔 |
