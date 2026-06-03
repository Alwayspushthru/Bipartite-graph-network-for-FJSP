import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from params import configs
from utils.common_utils import setup_seed, strToSuffix
from utils.data_utils import load_data_from_files
from env.FJSPEnv import FJSPEnv
from model.ppo import PPO_initialize

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device if torch.cuda.is_available() else "cpu")
configs.device = device.type
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
   torch.set_default_device('cuda')
else:
   torch.set_default_device('cpu')

ppo = PPO_initialize()

_SOLUTION_CSV = {
    'SD1': 'SD1Solution.csv',
    'SD2': 'SD2Solution.csv',
    'BenchData': 'BenchDataSolution.csv',
}

def load_baseline(data_name):
    csv_name = _SOLUTION_CSV.get(data_name)
    if csv_name is None:
        return None
    csv_path = f'./data/{data_name}/{csv_name}'
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

def test_greedy_strategy(data_set, model_path, seed, batch_size=0):
    """
    Greedy (argmax) testing. Greedy is deterministic and each instance's forward
    pass is independent, so running a whole batch of instances together yields
    the exact same per-instance makespan as running them one by one — only much
    faster (one large-batch forward instead of many batch-1 forwards). Finished
    instances are skipped via the active-subset masking, mirroring validate_envs.

    Per-instance wall-clock time is no longer meaningful under batching, so the
    returned `time` column is the batch wall-clock amortized evenly per instance.
    """
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    job_lengths, op_pts = data_set[0], data_set[1]
    n_total = len(job_lengths)
    if batch_size is None or batch_size <= 0:
        batch_size = n_total

    makespans = np.empty(n_total, dtype=np.float64)
    times = np.empty(n_total, dtype=np.float64)

    # The env assumes a uniform (n_jobs, n_machines) across the batch (it infers
    # both from the first instance), so group instances by shape before batching.
    # SD1/SD2 collapse to a single group; BenchData splits into its size classes.
    # Results are scattered back to the original index to keep baseline alignment.
    shape_groups = {}
    for idx in range(n_total):
        key = (job_lengths[idx].shape[0], op_pts[idx].shape[1])
        shape_groups.setdefault(key, []).append(idx)

    chunks = [grp[s:s + batch_size]
              for grp in shape_groups.values()
              for s in range(0, len(grp), batch_size)]

    for chunk in tqdm(chunks, file=sys.stdout, desc="progress", colour='blue'):
        b = len(chunk)
        state = env.set_initial_data([job_lengths[i] for i in chunk],
                                     [op_pts[i] for i in chunk])
        done = env.env_done.copy()
        h_hist = torch.zeros(b, ppo.policy.hist_dim, device=device)

        t1 = time.time()
        while not done.all():
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(done)  # only feed unfinished instances
                pi, _, h_new = ppo.policy(
                    state.fea_j_tensor[batch_idx],
                    state.fea_m_tensor[batch_idx],
                    state.fea_pairs_tensor[batch_idx],
                    state.dynamic_pair_mask_tensor[batch_idx],
                    h_hist[batch_idx],
                )
                action_envs = torch.argmax(pi, dim=-1)
            h_hist[batch_idx] = h_new
            state, _, done = env.step(actions=action_envs.cpu().numpy())
        t2 = time.time()

        per_inst_time = (t2 - t1) / b  # amortized — see docstring
        for k, orig_idx in enumerate(chunk):
            makespans[orig_idx] = env.current_makespan[k]
            times[orig_idx] = per_inst_time

    return np.column_stack([makespans, times])


def test_sampling_strategy(data_set, model_path, seed, n_samples):
    """
    Run n_samples stochastic rollouts per instance; keep the best makespan.

    The n_samples rollouts of one instance are run as a single parallel batch
    (the instance is replicated n_samples times) instead of a Python loop, which
    is far faster on GPU. Because the per-step sampling now draws for the whole
    batch at once, the RNG stream differs from the old sequential version, so
    individual makespans are not bit-identical to the previous seed — but the
    procedure (n_samples i.i.d. rollouts, keep best) is statistically the same.
    """
    test_result_list = []
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        state = env.set_initial_data([data_set[0][i]] * n_samples,
                                     [data_set[1][i]] * n_samples)
        done = env.env_done.copy()
        h_hist = torch.zeros(n_samples, ppo.policy.hist_dim, device=device)

        t1 = time.time()
        while not done.all():
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(done)
                pi, _, h_new = ppo.policy(
                    state.fea_j_tensor[batch_idx],
                    state.fea_m_tensor[batch_idx],
                    state.fea_pairs_tensor[batch_idx],
                    state.dynamic_pair_mask_tensor[batch_idx],
                    h_hist[batch_idx],
                )
                dist = torch.distributions.Categorical(pi)
                action_envs = dist.sample()
            h_hist[batch_idx] = h_new
            state, _, done = env.step(actions=action_envs.cpu().numpy())
        t2 = time.time()

        best_makespan = env.current_makespan.min()
        test_result_list.append([best_makespan, t2 - t1])

    return np.array(test_result_list)

def main(config):
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{model_name}.pth', model_name))

    # collect the test data: each entry in test_data is a data source folder (SD1/SD2/BenchData)
    test_data = [(load_data_from_files(f'./data/{name}'), name) for name in config.test_data]
    model_prefix = "Bgnn-G"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            n_samples = config.n_samples
            use_sampling = n_samples > 1
            mode_str = f"Sampling×{n_samples}" if use_sampling else "Greedy"
            model_prefix = f"Bgnn-S{n_samples}" if use_sampling else "Bgnn-G"

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = save_direc + f'/{model[1]}_{model_prefix}_{timestamp}.xlsx'

            if True:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{data[1]}")
                print(f"Test mode: {mode_str}")
                if use_sampling:
                    save_result = test_sampling_strategy(data[0], model[0], config.seed_test, n_samples)
                else:
                    # Greedy strategy is deterministic (argmax), so one run suffices.
                    # Synthetic sets (SD1/SD2) are size-uniform within each size class
                    # and get batched; BenchData is a heterogeneous real benchmark whose
                    # per-instance solve time is the reported metric, so keep it batch=1.
                    greedy_bs = 1 if data[1] == 'BenchData' else config.test_batch_size
                    save_result = test_greedy_strategy(data[0], model[0], config.seed_test,
                                                       greedy_bs)

                baseline_df = load_baseline(data[1])

                print(f"time: {save_result[:, 1].mean():.4f}s")
                print("testing results:")

                log_prefix = f'{timestamp}    model: {model[1]}    data: {data[1]}    mode: {mode_str}    '
                log_indent = ' ' * len(log_prefix)
                log_lines = []

                if baseline_df is not None and len(baseline_df) == len(save_result):
                    baseline = baseline_df['ub'].values
                    gaps = (save_result[:, 0] - baseline) / baseline * 100

                    if data[1] == 'BenchData':
                        group_labels = baseline_df['benchname'].values
                    else:
                        group_labels = baseline_df['dataname'].str.replace(r'_\d+$', '', regex=True).values
                    for idx, g_name in enumerate(sorted(np.unique(group_labels))):
                        g_mask = group_labels == g_name
                        g_gaps = gaps[g_mask]
                        g_makespan = save_result[:, 0][g_mask]
                        print(f"  [{g_name}]  makespan={g_makespan.mean():.2f}  gap: mean={g_gaps.mean():.2f}%  std={g_gaps.std():.2f}%")
                        group_str = f'[{g_name}]  makespan={g_makespan.mean():.2f}  gap: mean={g_gaps.mean():.2f}%  std={g_gaps.std():.2f}%'
                        log_lines.append((log_prefix if idx == 0 else log_indent) + group_str)

                    result_df = pd.DataFrame({
                        'makespan': save_result[:, 0],
                        'time': save_result[:, 1],
                        'ref_makespan': baseline,
                        'gap(%)': gaps.round(2),
                    })
                else:
                    result_df = pd.DataFrame(save_result, columns=["makespan", "time"])
                    log_lines.append(log_prefix + f'makespan={save_result[:, 0].mean():.2f}  time={save_result[:, 1].mean():.4f}s')

                result_df.to_excel(save_path, index=False)

                with open('./test_results/test_log.txt', 'a', encoding='utf-8') as f:
                    f.write('\n'.join(log_lines) + '\n')


if __name__ == "__main__":
    main(configs)
