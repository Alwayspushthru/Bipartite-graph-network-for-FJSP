import torch
import torch.nn as nn
from copy import deepcopy

from model.BiGraphNetwork import BiGraphNetwork
from params import configs


class Memory:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # state sequences — each element is [B, ...]
        self.fea_j_seq = []
        self.fea_m_seq = []
        self.fea_pairs_seq = []
        self.fea_waiting_seq = []
        self.candidate_seq = []
        self.job_mask_seq = []
        self.dynamic_pair_mask_seq = []
        # transition sequences
        self.action_seq = []
        self.reward_seq = []
        self.val_seq = []
        self.done_seq = []
        self.log_probs = []

    def push(self, state):
        self.fea_j_seq.append(state.fea_j_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)
        self.fea_waiting_seq.append(state.fea_waiting_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.job_mask_seq.append(state.job_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)

    def clear_memory(self):
        del self.fea_j_seq[:]
        del self.fea_m_seq[:]
        del self.fea_pairs_seq[:]
        del self.fea_waiting_seq[:]
        del self.candidate_seq[:]
        del self.job_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def get_sequence_data(self):
        """
        Stack all collected tensors into [T, B, ...] without flattening.
        Returns tensors suitable for forward_sequence.
        """
        fea_j    = torch.stack(self.fea_j_seq, dim=0)             # [T, B, J, Fj]
        fea_m    = torch.stack(self.fea_m_seq, dim=0)             # [T, B, M, Fm]
        fea_pairs = torch.stack(self.fea_pairs_seq, dim=0)        # [T, B, J, M, Fp]
        fea_waiting = torch.stack(self.fea_waiting_seq, dim=0)    # [T, B, J, 4]
        mask     = torch.stack(self.dynamic_pair_mask_seq, dim=0) # [T, B, J, M]
        action   = torch.stack(self.action_seq, dim=0)            # [T, B]
        reward   = torch.stack(self.reward_seq, dim=0)            # [T, B]
        old_val  = torch.stack(self.val_seq, dim=0)               # [T, B]
        done     = torch.stack(self.done_seq, dim=0)              # [T, B]
        old_logp = torch.stack(self.log_probs, dim=0)             # [T, B]
        return fea_j, fea_m, fea_pairs, fea_waiting, mask, action, reward, old_val, done, old_logp

    def get_gae_advantages(self, last_values=None):
        """
        Compute GAE advantages with done masking.

        The bootstrap value and the done flag are decoupled:
          - `i == T - 1` only selects which "next value" to use
            (last_values = V(s_T) for the final step, values[i+1] otherwise).
          - `not_done` independently decides whether to bootstrap at all.
        This makes truncated (non-terminal) rollouts correct: when an env has
        not finished at step T-1, its return bootstraps from V(s_T). When every
        env has terminated, last_values defaults to zeros (masked by not_done
        anyway), recovering the previous no-bootstrap behaviour.

        Normalizes per-env (over T). This is a deliberate, load-bearing design:
        the reward is a makespan-LB decrement whose magnitude scales with the
        instance's makespan, so per-env standardization gives the policy
        gradient scale-invariance that is critical for OOD generalization to
        benchmark data of very different scales (see exp_log 20260603). Global
        normalization over T*B regressed all BenchData groups (Hurink_vdata
        +12.92 pp) and was reverted.
        """
        reward_arr = torch.stack(self.reward_seq, dim=0)  # [T, B]
        values     = torch.stack(self.val_seq,    dim=0)  # [T, B]
        done_arr   = torch.stack(self.done_seq,   dim=0)  # [T, B]

        T, B = reward_arr.shape
        if last_values is None:
            last_values = torch.zeros(B, device=values.device)

        advantage = torch.zeros(B, device=values.device)
        advantage_seq = []

        for i in reversed(range(T)):
            not_done = (~done_arr[i]).float()  # [B]
            next_value = last_values if i == T - 1 else values[i + 1]
            delta = reward_arr[i] + self.gamma * next_value * not_done - values[i]
            advantage = delta + self.gamma * self.gae_lambda * not_done * advantage
            advantage_seq.insert(0, advantage)

        t_adv = torch.stack(advantage_seq, dim=0)  # [T, B]
        v_target = t_adv + values                  # [T, B] (raw advantages — keep
                                                   # value targets unnormalized)

        # Normalize per-env (over T dimension) — preserves scale-invariance for
        # OOD generalization. Do NOT switch to global (T*B) normalization: it
        # ties the gradient to the training reward scale and badly hurts OOD
        # benchmark transfer (reverted, exp_log 20260603).
        t_adv = (t_adv - t_adv.mean(dim=0, keepdim=True)) / \
                (t_adv.std(dim=0, keepdim=True) + 1e-8)

        return t_adv, v_target


class PPO:
    def __init__(self, config):
        self.lr          = config.lr
        self.gamma       = config.gamma
        self.gae_lambda  = config.gae_lambda
        self.eps_clip    = config.eps_clip
        self.k_epochs    = config.k_epochs
        self.tau         = config.tau

        self.ploss_coef  = config.ploss_coef
        self.vloss_coef  = config.vloss_coef
        self.entloss_coef = config.entloss_coef

        self.policy     = BiGraphNetwork(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2  = nn.MSELoss()

    def update(self, memory, last_values=None):
        (fea_j_seq, fea_m_seq, fea_pairs_seq, fea_waiting_seq, mask_seq,
         action_seq, reward_seq, old_val_seq, done_seq, old_logp_seq) = memory.get_sequence_data()

        t_adv, v_target = memory.get_gae_advantages(last_values)

        T, B = action_seq.shape

        total_loss = 0.0
        value_loss = 0.0
        policy_loss_total = 0.0
        entropy_total = 0.0

        for _ in range(self.k_epochs):
            h0 = torch.zeros(B, self.policy.hist_dim, device=fea_j_seq.device)

            # Recompute full sequence with current policy parameters (true BPTT)
            pi_seq, value_seq, _ = self.policy.forward_sequence(
                fea_j_seq, fea_m_seq, fea_pairs_seq, fea_waiting_seq, mask_seq, h0, done_seq
            )  # [T, B, J*M], [T, B]

            dist    = torch.distributions.Categorical(pi_seq)
            logp_new = dist.log_prob(action_seq)  # [T, B]
            entropy  = dist.entropy()             # [T, B]

            ratio = torch.exp(logp_new - old_logp_seq.detach())
            surr1 = ratio * t_adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * t_adv

            policy_loss  = -torch.min(surr1, surr2)
            critic_loss  = self.V_loss_2(value_seq, v_target.detach())
            entropy_loss = -entropy

            loss = (self.ploss_coef  * policy_loss
                  + self.vloss_coef  * critic_loss
                  + self.entloss_coef * entropy_loss)

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss       += loss.mean().detach()
            value_loss       += critic_loss.mean().detach()
            policy_loss_total += policy_loss.mean().detach()
            entropy_total    += entropy_loss.mean().detach()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return {
            "total_loss":   total_loss.item()        / self.k_epochs,
            "value_loss":   value_loss.item()        / self.k_epochs,
            "policy_loss":  policy_loss_total.item() / self.k_epochs,
            "entropy":      entropy_total.item()     / self.k_epochs,
        }


def PPO_initialize():
    return PPO(config=configs)
