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

def test_greedy_strategy(data_set, model_path, seed):
    test_result_list = []
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        h_hist = torch.zeros(1, ppo.policy.hist_dim, device=device)
        t1 = time.time()
        while True:
            with torch.no_grad():
                pi, _, h_hist = ppo.policy(
                    state.fea_j_tensor,
                    state.fea_m_tensor,
                    state.fea_pairs_tensor,
                    state.dynamic_pair_mask_tensor,
                    h_hist,
                )
                action_envs = torch.argmax(pi, dim=-1)
                state, reward, done = env.step(actions=action_envs.cpu().numpy())
                if done:
                    break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

    return np.array(test_result_list)


def test_sampling_strategy(data_set, model_path, seed, n_samples):
    """Run n_samples stochastic rollouts per instance; keep the best makespan."""
    test_result_list = []
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()
        best_makespan = float('inf')

        for _ in range(n_samples):
            state = env.reset()
            h_hist = torch.zeros(1, ppo.policy.hist_dim, device=device)
            while True:
                with torch.no_grad():
                    pi, _, h_hist = ppo.policy(
                        state.fea_j_tensor,
                        state.fea_m_tensor,
                        state.fea_pairs_tensor,
                        state.dynamic_pair_mask_tensor,
                        h_hist,
                    )
                    dist = torch.distributions.Categorical(pi)
                    action_envs = dist.sample()
                    state, _, done = env.step(actions=action_envs.cpu().numpy())
                    if done:
                        break
            best_makespan = min(best_makespan, env.current_makespan[0])

        t2 = time.time()
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
                    save_result = test_greedy_strategy(data[0], model[0], config.seed_test)

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
