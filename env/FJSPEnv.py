from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
import copy

import sys
import torch

@dataclass
class EnvState:
    fea_j_tensor: torch.Tensor = None
    fea_m_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None

    candidate_tensor: torch.Tensor = None
    job_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None

    def update(self, fea_j, fea_m, fea_pairs, candidate, job_mask_tensor, dynamic_pair_mask, device):
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device) # operation feature
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.job_mask_tensor = torch.from_numpy(np.copy(job_mask_tensor)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

class FJSPEnv:
    """
        Environment that builds the local scheduling state from synthetic data.
        let E/N/J/M denote the number of envs/operations/jobs/machines
    """
    def __init__(self, device):
        self.old_state = EnvState()

        # the dimension of operation raw features
        self.op_fea_dim = 7
        # the dimension of machine raw features
        self.mch_fea_dim = 4
        # the dimension of edge raw features
        self.edge_fea_dim = 6

        self.device = device

    def set_initial_data(self, job_length_list, op_pt_list):
        self.number_of_envs = len(job_length_list)  # batch_size
        self.job_length = np.array(job_length_list)  # 将数组列表结构转换成二维规则矩阵的形式
        self.number_of_jobs = job_length_list[0].shape[0]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.env_idxs = np.arange(self.number_of_envs)
        self.number_of_ops_per_env = np.sum(self.job_length,axis=1).astype(int) # 每个环境的操作数
        self.number_of_ops = np.max(self.number_of_ops_per_env) # 最大的操作数 O_max
        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]

        self.op_pt = np.zeros((self.number_of_envs, self.number_of_ops, self.number_of_machines)) # (B,O_max,M)
        self.op_valid_mask = np.zeros((self.number_of_envs, self.number_of_ops), dtype=bool)

        for env_idx, op_pt_env in enumerate(op_pt_list):
            env_ops = self.number_of_ops_per_env[env_idx]
            self.op_pt[env_idx,:env_ops,:] = op_pt_env[:env_ops]
            self.op_valid_mask[env_idx,:env_ops] = True

        head_op_id = np.zeros((self.number_of_envs, 1), dtype=int)
        self.job_first_op_id = np.concatenate(
            [head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1
        ).astype(int)
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        # Masks and relations
        self.process_relation = (self.op_pt != 0 )& self.op_valid_mask[:,:,None]    # [B, O, M]
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = np.sum(self.process_relation, axis=2) # [B,O_max] 每个操作能被多少台机器加工
        self.compatible_mch = np.sum(self.process_relation, axis=1) # [B,M] 每台机器能加工多少操作

        self.initial_vars()

        # Normalize processing times for feature stability
        pt =self.op_pt
        mask = pt > 0   # 因为pt=0表示不可加工，所以等于0的数据不应该统一进行normalize
        # 用 +inf/-inf 做掩码，避免 flatten 到全局
        pt_pos = np.where(mask, pt, np.inf)
        pt_neg = np.where(mask, pt, -np.inf)

        lb = pt_pos.min(axis=(1,2)) # pre_env 按每个实例归一化 # (E, )
        ub = pt_neg.max(axis=(1,2)) # (E, )

        self.pt_lower_bound = lb
        self.pt_upper_bound = ub

        # 广播到 (E,1,1)
        lb_b = lb[:, None, None]
        den_b = (ub - lb)[:, None, None]
        pt_norm = np.where(mask, (pt - lb_b) / (den_b + 1e-8), 0.0).astype(np.float32)

        self.true_op_pt = pt.copy() # 真实的值
        self.unmasked_op_pt = pt_norm.copy()

        op_pt_ma = ma.array(self.unmasked_op_pt, mask=self.reverse_process_relation)
        self.op_pt = op_pt_ma
        self.op_mean_pt = op_pt_ma.mean(axis=2).filled(0)
        self.op_min_pt = op_pt_ma.min(axis=2).filled(0)
        self.op_max_pt = op_pt_ma.max(axis=2).filled(0)
        self.pt_span = self.op_max_pt - self.op_min_pt

        # the estimated lower bound of complete time of operations
        self.op_ct_lb = copy.deepcopy(self.op_min_pt) # 每个操作的估计完工时间
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])

        # Candidate processing times and masks
        self.candidate_pt = np.array([self.op_pt[k, self.candidate[k]] for k in range(self.number_of_envs)])  ### 非法记得置为-1
        dynamic_pair = np.array([self.process_relation[k, self.candidate[k]] for k in range(self.number_of_envs)])
        self.candidate_process_relation = np.copy(~dynamic_pair)

        # Construct features
        self.construct_candidate_features() # [B,J,8]
        self.construct_mch_features()
        self.construct_pair_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = self.init_quality

        self.old_state.update(self.fea_j,self.fea_m,self.fea_pairs,self.candidate,self.mask,
                              self.candidate_process_relation, self.device)

        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_compatible_op = np.copy(self.compatible_op)
        self.old_compatible_mch = np.copy(self.compatible_mch)

        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):

        self.initial_vars()

        # copy the old data
        self.op_ct_lb = np.copy(self.old_op_ct_lb)

        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.compatible_op = np.copy(self.old_compatible_op)
        self.compatible_mch = np.copy(self.old_compatible_mch)

        # copy the old state
        self.state = copy.deepcopy(self.old_state)

        return self.state

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.step_count = 0
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        # the complete time of operations ([E,N])
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        self.unscheduled_op_mask = np.zeros((self.number_of_envs, self.number_of_ops), dtype=bool)
        self.unscheduled_op_mask[self.op_valid_mask] = True
        self.idle_acc = np.zeros((self.number_of_envs, self.number_of_machines))  # 机器m的空闲时间累计

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)
        self.env_done = np.zeros(self.number_of_envs,dtype=bool)


    def step(self, actions):
        """
            perform the state transition & return the next state and reward
            :param actions: the action list with shape [E]
            :return: the next state, reward and the done flag
        """
        active_idx = np.where(~self.env_done)[0]

        active_job = actions // self.number_of_machines
        active_machine = actions % self.number_of_machines
        chosen_op = self.candidate[active_idx, active_job]

        # 合法性检查
        if (self.reverse_process_relation[active_idx, chosen_op, active_machine]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {active_machine}')
            sys.exit()

        # 候选工序推进
        active_has_next = (chosen_op != self.job_last_op_id[active_idx, active_job])   # 是否还有下一道工序
        self.candidate[active_idx[active_has_next], active_job[active_has_next]] += 1  # 那么对应的工件就+1
        self.mask[active_idx[~active_has_next], active_job[~active_has_next]] = True   # (B,J)

        next_env = active_idx[active_has_next]
        next_job = active_job[active_has_next]
        next_op = chosen_op[active_has_next] + 1

        self.candidate_pt[next_env, next_job] = self.unmasked_op_pt[next_env, next_op]  # 更新候选操作的加工时间
        self.candidate_process_relation[next_env, next_job] = self.reverse_process_relation[next_env, next_op] # 更新可行性关系
        self.candidate_process_relation[active_idx[~active_has_next], active_job[~active_has_next]] = True

        # 对活跃env做时间更新
        chosen_op_st = np.maximum(self.candidate_free_time[active_idx, active_job],
                                  self.mch_free_time[active_idx, active_machine])       # 本步操作的开始时间

        chosen_op_pt = self.unmasked_op_pt[active_idx, chosen_op, active_machine]
        self.op_ct[active_idx, chosen_op] = chosen_op_st + chosen_op_pt

        self.unscheduled_op_mask[active_idx, chosen_op] = False

        idle_inc = np.maximum(0.0, chosen_op_st - self.mch_free_time[active_idx, active_machine])
        self.idle_acc[active_idx, active_machine] += idle_inc

        self.candidate_free_time[active_idx, active_job] = self.op_ct[active_idx, chosen_op]
        self.mch_free_time[active_idx, active_machine] = self.op_ct[active_idx, chosen_op]

        # 真实时间更新
        true_chosen_op_st = np.maximum(self.true_candidate_free_time[active_idx, active_job],
                                       self.true_mch_free_time[active_idx, active_machine])
        self.true_op_ct[active_idx, chosen_op] = true_chosen_op_st + self.true_op_pt[active_idx, chosen_op, active_machine]

        self.true_candidate_free_time[active_idx, active_job] = self.true_op_ct[active_idx, chosen_op]
        self.true_mch_free_time[active_idx, active_machine] = self.true_op_ct[active_idx, chosen_op]

        self.current_makespan[active_idx] = np.maximum(self.current_makespan[active_idx],
                                                       self.true_op_ct[active_idx, chosen_op])

        diff = self.op_ct[active_idx, chosen_op] - self.op_ct_lb[active_idx, chosen_op]
        # 对每个活跃env, 把该job从当前op到该job最后一个op的下界整体后移diff
        for k in range(len(active_idx)):
            e = active_idx[k]
            j = active_job[k]
            op = chosen_op[k]

            start = op
            end = self.job_last_op_id[e, j] + 1
            self.op_ct_lb[e, start:end] += diff[k]

        self.construct_candidate_features()
        self.construct_mch_features()
        self.construct_pair_features()

        self.step_count += 1

        for i in range(self.number_of_envs):
            self.env_done[i] = (self.step_count >= self.number_of_ops_per_env[i])

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        op_ct_lb_visible = np.where(self.op_valid_mask, self.op_ct_lb, 0)
        reward = self.max_endTime - np.max(op_ct_lb_visible, axis=1)
        self.max_endTime = np.max(op_ct_lb_visible, axis=1)

        true_candidate = np.where(self.mask, -1, self.candidate)

        self.state.update(self.fea_j, self.fea_m, self.fea_pairs, true_candidate, self.mask,
                          self.candidate_process_relation, self.device)

        return self.state, np.array(reward), self.env_done


    def construct_candidate_features(self):
        """
            [1] feasible_mas_ratio : 可用机器数/总机器数
            [2] job_ready : 对应工件的释放时间(norm)
            [3] rem_ops :  工件剩余操作数 / 工件总操作数    (加工进度)
            [4] rem_work : 工件剩余工作量(未调度操作的平均加工时间之和)
            [5] p_mean : 平均加工时间
            [6] p_span : 加工时间跨度
            [7] job_ct_lb : 工件最后一道工序的完工时间下界
        :return: fea_j[B,J,7] 若其中有工件已完工那么用mask将对应特征置为0
        """
        feasible_mas = self.compatible_op[self.env_idxs[:, None], self.candidate] # 操作能被多少台机器加工
        feasible_mas_ratio = feasible_mas / self.number_of_machines
        job_ready = self.candidate_free_time
        rem_ops = (self.job_length - self.candidate + self.job_first_op_id) / self.job_length

        rem_work = []
        for env_idx in range(self.number_of_envs):
            job_work = []
            for job_idx in range(self.number_of_jobs):
                start = self.candidate[env_idx, job_idx]
                end = self.job_last_op_id[env_idx, job_idx] + 1
                job_work.append(np.sum(self.op_mean_pt[env_idx, start:end]))
            rem_work.append(job_work)
        rem_work = np.array(rem_work)

        p_mean = self.op_mean_pt[self.env_idxs[:, None], self.candidate]
        p_span = self.pt_span[self.env_idxs[:, None], self.candidate]
        job_ct_lb = self.op_ct_lb[self.env_idxs[:, None], self.job_last_op_id]

        self.fea_j = np.stack(
            (feasible_mas_ratio, job_ready, rem_ops, rem_work, p_mean, p_span, job_ct_lb),
            axis=2,
        )

        mask = self.mask[:,:,None]
        self.fea_j = np.where(mask, 0, self.fea_j)

        # 每个env还有多少个有效节点
        num_left_nodes = np.sum(~self.mask, axis=1, keepdims=True) # [B, 1]
        valid_env = (num_left_nodes.squeeze(-1) > 0)               # [B]

        if valid_env.any():
            fea_j_valid = self.fea_j[valid_env]  # [Bv, J, 7]
            mask_valid = mask[valid_env]         # [Bv, J, 1]
            num_left_valid = num_left_nodes[valid_env]  # [Bv, 1]

            mean_fea_j = np.sum(fea_j_valid, axis=1) / num_left_valid
            temp = np.where(mask_valid, mean_fea_j[:, np.newaxis, :], fea_j_valid)

            var_fea_j = np.var(temp, axis=1)
            std_fea_j = np.sqrt(var_fea_j * self.number_of_jobs / num_left_valid)

            self.fea_j[valid_env] = (temp - mean_fea_j[:, np.newaxis, :]) / \
                         (std_fea_j[:, np.newaxis, :] + 1e-8)

    def construct_mch_features(self):
        """
            [1] feasible_ops_norm: 所有可加工操作数 / 所有还存在的工件数
            [2] mach_ready: 机器的空闲时间
            [3] expect_workload: 按责任权重分摊后的机器潜在负载
            [4] idle: 空闲时间累计
        :return:fea_m[B,M,4]
        """
        feasible_ops = np.sum(~self.candidate_process_relation, axis=1)
        num_alive = np.sum(~self.mask, axis=1, keepdims=True)
        feasible_ops_norm = feasible_ops / (num_alive + 1e-8)
        mach_ready = self.mch_free_time

        unscheduled_relation = self.process_relation & self.unscheduled_op_mask[:, :, None]
        # 还未调度的工序中可加工的部分(B,O,M)
        n_compatible = np.sum(unscheduled_relation, axis=2, keepdims=True)  # (B,O,1)
        uniform_resp = np.where(unscheduled_relation, 1.0 / (n_compatible + 1e-8), 0.0)
        expect_workload = np.sum(uniform_resp * self.true_op_pt, axis=1)
        
        idle = self.idle_acc
        self.fea_m = np.stack((feasible_ops_norm, mach_ready, expect_workload, idle), axis=2)

        # 没有删除节点的normalize
        mean_fea_m = np.sum(self.fea_m, axis=1) / self.number_of_machines
        var_fea_m = np.var(self.fea_m, axis=1)
        std_fea_m = np.sqrt(var_fea_m)

        self.fea_m = (self.fea_m - mean_fea_m[:, np.newaxis, :]) / \
                     (std_fea_m[:, np.newaxis, :] + 1e-8)

    def construct_pair_features(self):
        """
            [1] pt_out: 边上的加工时间
            [2] start: 最早开始时间
            [3] wait_job: 工件就绪时间-机器就绪时间
            [4] wait_mach: 机器就绪时间-共建就绪时间
            [5] ratio_mach: p / p_mean(m)
            [6] ratio_op: p / p_mean(o)
        :return: fea_pairs[B,J,M,6]
        """
        pt = self.candidate_pt
        mask = ~self.candidate_process_relation
        ES = np.maximum(self.candidate_free_time[..., None], self.mch_free_time[:, None, :])  # (1,3,3)
        start = np.where(mask.astype(bool), ES, 0)

        wait_job = np.maximum(0, self.candidate_free_time[:, :, None] - self.mch_free_time[:, None, :])  # [3]等待工件
        wait_mach = np.maximum(0, self.mch_free_time[:, None, :] - self.candidate_free_time[:, :, None])  # [4]等待机器

        feasible_ops = np.sum(mask, axis=1)
        pt_masked = np.where(mask,pt,0.0)  # 把pt中的非法动作加工时间置为0，主要考虑有工件已经完成的情况
        sum = np.sum(pt_masked, axis=1, keepdims=True)
        mch_mean_candidate_pt = sum / (feasible_ops[:,None,:] + 1e-8)
        ratio_m = pt_masked / (mch_mean_candidate_pt + 1e-8)

        op_mean_p = np.expand_dims(self.op_mean_pt[self.env_idxs[:, None], self.candidate], axis=-1)
        ratio_op = pt_masked / (op_mean_p + 1e-8)  # [6]加工时间/同操作平均

        pt_out = np.where(mask,pt,0)

        self.fea_pairs = np.stack((pt_out, start, wait_job,wait_mach, ratio_m, ratio_op),axis=3,)
