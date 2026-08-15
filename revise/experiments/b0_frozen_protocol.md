# Frozen B0 Protocol

Status: `COMPLETED_WITH_DOCUMENTED_CPU_DEVIATION`

## Source snapshot

- Branch: `main`
- Git commit: `d145d92f178ecff6438da0dbb90f9d3bc37cb72f`
- Commit time: `2026-08-14 20:53:14 +0800`
- Commit subject: `返修init`
- Baseline origin: commit `d145d92f178ecff6438da0dbb90f9d3bc37cb72f` plus the documented non-functional cleanup that removes an unused attempt to load `data/SD3/10x5` from `train.py`
- Working-tree state at freeze: `train.py` contains the documented cleanup; `revise/` contains revision records

### File fingerprints

| File | SHA-256 |
|---|---|
| `params.py` | `84755ba1c5586bbb56ad8b2e794037a3a926c70655d4e8ca43d37e348a55eea1` |
| `train.py` | `04775930a00f41390f15aefa7c7faebcaccd3300514c14f2392ad37cfb588504` |
| `test.py` | `9d47395395811e3e84f8a62b12d553375c705c3d17614bfe66c4e9e4f4c172f1` |
| `env/FJSPEnv.py` | `c700249306dfbf7457670fa320ffb9ae1b5104aa4260453244bb7b7b6f719562` |
| `model/BiGraphNetwork.py` | `5005a4cabee86cba82acc518b6196fb340a038c8f4f446a42130c07267f32fbb` |
| `model/ppo.py` | `d789c6f9b97a5f951166ee1413ba0ba67a9ec571f975311e8c61a5aa7834dab6` |
| `pyproject.toml` | `4e6a9ca21d1681fa2556152597d90bb4c1c4969e3f849ad504844f6cad7c223c` |
| `uv.lock` | `2cf5f77a5a5080b3dbcb6f69f320f38ea5234286475e591b8d0312ee145184df` |
| `data/BenchData/BenchDataSolution.csv` | `61177369e7b0bff70d0489f72d41acb6dbd4e00fb9b4732ff859c797696b8f03` |

## Frozen training configuration

| Item | Value |
|---|---|
| Configuration | B0, `ablation=none` |
| Training distribution | SD3 |
| Instance size | 10 jobs, 5 machines |
| Updates | 1,000 |
| Primary training seed | 300 |
| Environments | 20 |
| Resampling interval | 20 updates |
| Validation interval | 10 updates |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| PPO epochs | 4 |
| PPO clip | 0.2 |
| Discount factor | 1.0 |
| GAE lambda | 0.98 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Policy-loss coefficient | 1.0 |
| Checkpoint selection | lowest mean validation makespan |

The validation set is the 100 files under `data/data_train_vali/SD3/10x5`. Training instances are generated online by `SD3CaseGenerator`. The unused legacy attempt to load `data/SD3/10x5` has been removed; this cleanup does not change training, validation, checkpoint selection, or evaluation behavior.

## Frozen training command

Run from the repository root in the established CUDA environment:

```powershell
python train.py --device cuda --device_id 0 --data_source SD3 --n_j 10 --n_m 5 --max_updates 1000 --seed_train 300 --seed_test 50 --num_envs 20 --reset_env_timestep 20 --validate_timestep 10 --lr 3e-4 --gamma 1 --gae_lambda 0.98 --k_epochs 4 --eps_clip 0.2 --entloss_coef 0.01 --vloss_coef 0.5 --ploss_coef 1 --ablation none --model_name revise_b0_s300 --run_name revise_b0_s300
```

Expected checkpoint: `trained_network/revise_b0_s300.pth`.

The current `train.py` creates `trained_network/SD3/` but saves the checkpoint in the root `trained_network/` directory. The command and expected path above follow the actual save behavior.

## Frozen evaluation command

```powershell
python test.py --device cuda --device_id 0 --test_model revise_b0_s300 --test_data BenchData --seed_test 50 --beam_width 10 --beam_stochastic true --ablation none
```

`BenchData` contains 10 Brandimarte, 40 Hurink edata, 40 Hurink rdata, and 40 Hurink vdata instances. `test.py` evaluates them in one collection and reports each group separately.

Expected raw output pattern:

```text
test_results/BenchData/revise_b0_s300_Bgnn-SB10_<timestamp>.xlsx
```

Expected summary log: `test_results/test_log.txt`.

## Records required after execution

- actual start and end times;
- GPU and software environment;
- checkpoint SHA-256;
- TensorBoard run path;
- raw Excel result path and SHA-256;
- observed Brandimarte, edata, rdata, and vdata mean Gap;
- per-group Gap standard deviation;
- mean inference time;
- any deviation from this protocol.

## Freeze rule

Do not modify B0 source files after launching the B0 run. Implement each reviewer variant from the frozen commit and identify its own commit or source fingerprint. If a shared bug is found, record it explicitly, establish a new B0 snapshot, and rerun B0 before comparing any variant.

## Completed execution record

- Run ID: `B0-S300-CPU-20260814`
- Training window: 2026-08-14 22:10 to 22:28, Asia/Shanghai
- Training duration: 1,087.84 seconds
- Environment: Python 3.12.13, PyTorch 2.13.0, CPU
- Authorized deviation: CUDA and MPS were unavailable; the author approved CPU execution
- Best mean validation makespan: 508.66 at update 1,000
- Checkpoint: `trained_network/revise_b0_s300.pth`
- Checkpoint SHA-256: `3710abc86c61a745d517ba3fcc7891410752cf63722a3c5e4b087de8623f30a8`
- TensorBoard record: `runs/revise_b0_s300/2026-08-14_2210/events.out.tfevents.1786716620.KyrstindeAir.44829.0`
- Evaluation window: 2026-08-14 22:28:47 to 22:29:55, Asia/Shanghai
- Raw evaluation result: `test_results/BenchData/revise_b0_s300_Bgnn-SB10_20260814_222847.xlsx`
- Raw result SHA-256: `e266b757667e1fc9f4faea8d9bf5b5c7d1401af71fae0ffc674903ae0dc0d5fa`
- Evaluation completeness: 130 rows, zero missing values

All revision variants compared with this B0 must use the same CPU environment unless B0 is rerun in the replacement environment.
