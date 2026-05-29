import sys
import random
import torch
import os
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from params import configs
from utils.common_utils import setup_seed,strToSuffix
from utils.data_utils import load_data_from_files, SD2_instance_generator
from utils.logger_utils import TBLogger

from env.FJSPEnv import FJSPEnv
from model.ppo import PPO_initialize
from model.ppo import Memory

from utils.data_utils import CaseGenerator,SD3CaseGenerator

str_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device)

class Trainer:
    def __init__(self, config):
        self.n_j = config.n_j
        self.n_m = config.n_m

        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)

        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        #log
        self.log_interval = config.log_interval

        # 创建保存model和log的文件夹
        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')

        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')

        if self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'
        else:
            self.data_name = f'{self.n_j}x{self.n_m}'

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)
        setup_seed(self.seed_test)

        self.env = FJSPEnv(device) # 训练用的环境
        # validation data set
        self.test_data = load_data_from_files(self.test_data_path) # 测试用的数据
        vali_data = load_data_from_files(self.vali_data_path) # 验证用的数据
        self.vali_env = FJSPEnv(device) # 验证用的环境
        self.vali_env.set_initial_data(vali_data[0], vali_data[1]) # 在环境中初始化数据

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

        #log
        self.tb_logger = TBLogger(
            log_dir=config.log_dir,
            run_name=config.run_name or self.model_name,
            enabled=config.use_tensorboard,
        )

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)
            # h is maintained only for rollout sampling; update recomputes from h0=zeros
            h = torch.zeros(self.num_envs, self.ppo.policy.hist_dim, device=device)

            while True:
                # state store (no h stored — update recomputes from scratch)
                self.memory.push(state)
                with torch.no_grad():
                    action, log_prob, value, h = self.ppo.policy_old.act(
                        state.fea_j_tensor,
                        state.fea_m_tensor,
                        state.fea_pairs_tensor,
                        state.candidate_tensor,
                        state.dynamic_pair_mask_tensor,
                        h,
                    )

                # state transition
                state, reward, done = self.env.step(actions=action.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)
                done_tensor = torch.from_numpy(done).to(device)

                # Reset hidden for envs that finished this step
                h = h * (~done_tensor).float().unsqueeze(-1)

                self.memory.action_seq.append(action)
                self.memory.log_probs.append(log_prob)
                self.memory.val_seq.append(value)
                self.memory.reward_seq.append(reward)
                self.memory.done_seq.append(done_tensor)

                if done.all():
                    break

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            loss_info = self.ppo.update(self.memory)
            self.memory.clear_memory()

            if isinstance(loss_info, dict):
                if "policy_loss" in loss_info:
                    self.tb_logger.add_scalar("ppo/policy_loss", loss_info["policy_loss"], update_step=i_update)
                if "entropy" in loss_info:
                    self.tb_logger.add_scalar("ppo/entropy",loss_info["entropy"],update_step=i_update)
                if "value_loss" in loss_info:
                    self.tb_logger.add_scalar("ppo/value_loss",loss_info["value_loss"],update_step=i_update)
                if "total_loss" in loss_info:
                    self.tb_logger.add_scalar("ppo/total_loss",loss_info["total_loss"],update_step=i_update)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            ep_et = time.time()

            if (i_update + 1) % self.log_interval == 0:
                self.tb_logger.add_scalar("train/episode_reward_mean",mean_rewards_all_env,global_step=i_update)
                self.tb_logger.add_scalar("train/makespan_mean",mean_makespan_all_env,global_step=i_update)

            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env,
                    loss_info.get("total_loss", 0.0) if isinstance(loss_info, dict) else loss_info,
                    ep_et - ep_st))

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                vali_result = self.validate_envs().mean() #####
                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
                self.tb_logger.add_scalar("train/vali_makespan_mean",vali_result,global_step=i_update)

        self.train_et = time.time()

        # log results
        self.save_training_log()
        self.tb_logger.flush()
        self.tb_logger.close()

    def sample_training_instances(self):
        """
            sample training instances following the config,
        :return: new training instances
        """
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        dataset_JobLength = []
        dataset_OpPT = []
        for i in range(self.num_envs):
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)
            elif self.data_source == 'SD2':
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            else:
                case = SD3CaseGenerator(self.n_j, self.n_m, nums_ope=prepare_JobLength)  # 新的生成数据的方式
                JobLength, OpPT, _ = case.get_case()

            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def validate_envs(self):
        self.ppo.policy.eval()
        state = self.vali_env.reset()
        done = self.vali_env.env_done
        n_vali = state.fea_j_tensor.shape[0]
        h_hist = torch.zeros(n_vali, self.ppo.policy.hist_dim, device=device)
        while not done.all():
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(done)
                action_envs, _, _, h_hist_new = self.ppo.policy.act(
                    state.fea_j_tensor[batch_idx],
                    state.fea_m_tensor[batch_idx],
                    state.fea_pairs_tensor[batch_idx],
                    state.candidate_tensor[batch_idx],
                    state.dynamic_pair_mask_tensor[batch_idx],
                    h_hist[batch_idx],
                )
            h_hist[batch_idx] = h_hist_new
            state, _, done = self.vali_env.step(actions=action_envs.cpu().numpy())
        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_model(self):
        model_dir = f'./trained_network'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.ppo.policy.state_dict(), f'{model_dir}/{self.model_name}.pth')

def main():
    trainer = Trainer(configs)
    trainer.train()

if __name__ == '__main__':
    main()
