import torch.nn as nn
import torch
from copy import deepcopy

from model.BiGraphNetwork import BiGraphNetwork
from params import configs
import numpy as np

class Memory:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 6]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 3]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.job_mask_seq = []  # [N, tensor[sz_b, J]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]

    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        del self.fea_j_seq[:]
        del self.fea_m_seq[:]
        del self.fea_pairs_seq[:]
        del self.candidate_seq[:]
        del self.job_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]

    def push(self, state):
        self.fea_j_seq.append(state.fea_j_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.job_mask_seq.append(state.job_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)

    def transpose_data(self):
        """
            transpose the first and second dimension of collected variables
        """
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_job_mask_seq = torch.stack(self.job_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)

        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)

        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        return (t_Fea_j_seq, t_Fea_m_seq, t_pairMessage_seq,
                t_candidate_seq, t_job_mask_seq, t_dynamicMask_seq,
                t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq)

    def get_gae_advantages(self):
        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = torch.stack(self.val_seq, dim=0)

        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # [sz_b, N]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # [sz_b, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        # normalization
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) \
                          / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq

class PPO:
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau

        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size

        self.policy = BiGraphNetwork(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def update(self, memory):
        '''
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        '''

        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        (t_Fea_j_seq, t_Fea_m_seq, t_pairMessage_seq,
         t_candidate_seq, t_job_mask_seq, t_dynamicMask_seq,
         t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq) = t_data

        num_samples = t_action_seq.size(0)
        total_loss = 0.0 # policy,value,ent
        value_loss = 0.0
        policy_loss_total = 0.0
        entropy_total = 0.0
        num_batches = 0 # 总步数

        for _ in range(self.k_epochs):
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                batch_slice = slice(start, end)

                logp_new, entropy, value_new = self.policy.evaluate_actions(
                    t_Fea_j_seq[batch_slice],
                    t_Fea_m_seq[batch_slice],
                    t_pairMessage_seq[batch_slice],
                    t_candidate_seq[batch_slice],
                    t_dynamicMask_seq[batch_slice],
                    t_action_seq[batch_slice],
                )

                ratio = torch.exp(logp_new - t_logprobs_seq[batch_slice].detach())
                surr1 = ratio * t_advantage_seq[batch_slice]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * t_advantage_seq[batch_slice]

                policy_loss = -torch.min(surr1, surr2)
                critic_loss = self.V_loss_2(value_new, v_target_seq[batch_slice])
                entropy_loss = -entropy

                loss =  self.ploss_coef * policy_loss + self.vloss_coef * critic_loss + self.entloss_coef * entropy_loss
                self.optimizer.zero_grad()

                total_loss += loss.mean().detach()
                value_loss += critic_loss.mean().detach()
                policy_loss_total += policy_loss.mean().detach()
                entropy_total += entropy_loss.mean().detach()
                num_batches += 1

                loss.mean().backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return {
            "total_loss": (total_loss.item() / num_batches),
            "value_loss": (value_loss.item() / num_batches),
            "policy_loss": (policy_loss_total.item() / num_batches),
            "entropy": (entropy_total.item() / num_batches),
        }


def PPO_initialize():
    ppo = PPO(config=configs)
    return ppo