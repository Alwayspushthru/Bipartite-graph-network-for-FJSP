import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import configs
from utils.common_utils import setup_seed, strToSuffix
from utils.data_utils import load_data_from_files, pack_data_from_config
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

def sample_action_from_logits(logits, dynamic_pair_mask):
    valid_mask = ~dynamic_pair_mask.bool()
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    batch_size, num_jobs, num_machines = masked_logits.shape
    flat_logits = masked_logits.view(batch_size, num_jobs * num_machines)
    dist = torch.distributions.Categorical(logits=flat_logits)
    action = dist.sample()
    return action, dist.log_prob(action)

def test_sampling_strategy(data_set, model_path, sample_times, seed):
    setup_seed(seed)
    test_result_list = []
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        job_length_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
        op_pt_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

        state = env.set_initial_data(job_length_dataset, op_pt_dataset)
        t1 = time.time()
        while True:
            with torch.no_grad():
                logits, _ = ppo.policy_old(
                    state.fea_j_tensor,
                    state.fea_m_tensor,
                    state.fea_pairs_tensor,
                    state.candidate_tensor,
                    state.dynamic_pair_mask_tensor,
                )
            action_envs, _ = sample_action_from_logits(logits, state.dynamic_pair_mask_tensor)
            state, _, done = env.step(action_envs.cpu().numpy())
            if done.all():
                break

        t2 = time.time()
        best_makespan = np.min(env.current_makespan)
        test_result_list.append([best_makespan, t2 - t1])

    return np.array(test_result_list)

def test_greedy_strategy(data_set, model_path, seed):
    test_result_list = []
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()
        while True:
            with torch.no_grad():
                pi, _ = ppo.policy(
                    state.fea_j_tensor,
                    state.fea_m_tensor,
                    state.fea_pairs_tensor,
                    state.dynamic_pair_mask_tensor,
                )
                action_envs = torch.argmax(pi, dim=-1)
                state, reward, done = env.step(actions=action_envs.cpu().numpy())
                if done:
                    break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

    return np.array(test_result_list)

def main(config, flag_sample):
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        # test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))
        test_model.append((f'./trained_network/{model_name}.pth', model_name))

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)

    if not flag_sample:
        model_prefix = "Bi2-G"
    else:
        model_prefix = "Bi-Graph_S"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f'/{model_prefix}+{model[1]}_{data[1]}.xlsx'

            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(1):
                        result = test_greedy_strategy(data[0], model[0], config.seed_test)
                        result_5_times.append(result)
                    result_5_times = np.array(result_5_times)

                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print(f"makespan(greedy): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())

                else:
                    # Sample mode, test once.
                    print("Test mode: Sample")
                    save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
                    print("testing results:")
                    print(f"makespan(sampling): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())

                result_df = pd.DataFrame(save_result, columns=["makespan", "time"])
                result_df.to_excel(save_path, index=False)

if __name__ == "__main__":
    main(configs, configs.test_mode)
