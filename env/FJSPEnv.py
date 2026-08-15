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
    fea_waiting_tensor: torch.Tensor = None

    candidate_tensor: torch.Tensor = None
    job_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None

    def update(self, fea_j, fea_m, fea_pairs, fea_waiting, candidate, job_mask_tensor, dynamic_pair_mask, device):
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device) # operation feature
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)
        self.fea_waiting_tensor = torch.from_numpy(np.copy(fea_waiting)).float().to(device)

        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.job_mask_tensor = torch.from_numpy(np.copy(job_mask_tensor)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

class FJSPEnv:
    """
        Environment that builds the local scheduling state from synthetic data.
        let E/N/J/M denote the number of envs/operations/jobs/machines
    """
    def __init__(self, device, revision_variant='b0', reward_shaping_beta=0.1):
        self.old_state = EnvState()

        # the dimension of operation raw features
        self.op_fea_dim = 8
        # the dimension of machine raw features
        self.mch_fea_dim = 5
        # the dimension of edge raw features
        self.edge_fea_dim = 6

        self.device = device
        self.revision_variant = revision_variant
        self.reward_shaping_beta = reward_shaping_beta

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

        # Precomputed prefix-sum of op_mean_pt for O(1) range-sum queries in construct_candidate_features.
        # Shape [B, O_max+1]: col 0 is zero, col i+1 = sum of op_mean_pt[:, 0:i+1].
        self.op_mean_pt_cumsum = np.concatenate(
            [np.zeros((self.number_of_envs, 1)), np.cumsum(self.op_mean_pt, axis=1)],
            axis=1,
        )  # [B, O_max+1]
        flexibility = self.compatible_op / self.number_of_machines
        self.op_flexibility_cumsum = np.concatenate(
            [np.zeros((self.number_of_envs, 1)), np.cumsum(flexibility, axis=1)],
            axis=1,
        )
        self.job_total_expected_work = (
            self.op_mean_pt_cumsum[self.env_idxs[:, None], self.job_last_op_id + 1]
            - self.op_mean_pt_cumsum[self.env_idxs[:, None], self.job_first_op_id]
        )
        self.total_expected_work = np.sum(
            np.where(self.op_valid_mask, self.op_mean_pt, 0.0), axis=1
        )

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
        self.construct_waiting_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = self.init_quality

        self.old_state.update(self.fea_j,self.fea_m,self.fea_pairs,self.fea_waiting,self.candidate,self.mask,
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
        self.machine_assigned_count = np.zeros((self.number_of_envs, self.number_of_machines))
        self.shaping_potential = np.zeros(self.number_of_envs)

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
        self.machine_assigned_count[active_idx, active_machine] += 1

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
        self.construct_waiting_features()

        self.step_count += 1

        self.env_done = self.step_count >= self.number_of_ops_per_env

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        op_ct_lb_visible = np.where(self.op_valid_mask, self.op_ct_lb, 0)
        reward = self.max_endTime - np.max(op_ct_lb_visible, axis=1)
        self.max_endTime = np.max(op_ct_lb_visible, axis=1)

        if self.revision_variant == 'r':
            progress = np.minimum(self.step_count / self.number_of_ops_per_env, 1.0)
            processing_load = np.maximum(0.0, self.mch_free_time - self.idle_acc)
            load_fraction = processing_load / (processing_load.sum(axis=1, keepdims=True) + 1e-8)
            imbalance = np.std(load_fraction, axis=1)
            new_potential = -4.0 * progress * (1.0 - progress) * imbalance
            reward = reward + self.reward_shaping_beta * (new_potential - self.shaping_potential)
            self.shaping_potential = new_potential

        true_candidate = np.where(self.mask, -1, self.candidate)

        self.state.update(self.fea_j, self.fea_m, self.fea_pairs, self.fea_waiting, true_candidate, self.mask,
                          self.candidate_process_relation, self.device)

        return self.state, np.array(reward), self.env_done


    def reorder(self, perm):
        """
            Reindex all per-env *dynamic* state along the batch axis by `perm`.

            Used by beam search: after scoring K beams and selecting the K best
            children, each child continues from one parent beam, so every mutable
            per-env array must be gathered by the parent indices before `step`.

            `perm` is an int array of length number_of_envs with values in
            [0, number_of_envs). Static per-instance arrays (op_pt, process_relation,
            compatible_op/mch, op_mean_pt*, job_* ids, op_valid_mask, ...) are
            identical across beams of one instance and are deliberately left
            untouched. The EnvState tensors are rebuilt by the following `step`.
        """
        perm = np.asarray(perm)
        self.current_makespan = self.current_makespan[perm]
        self.op_ct = self.op_ct[perm]
        self.mch_free_time = self.mch_free_time[perm]
        self.candidate_free_time = self.candidate_free_time[perm]
        self.true_op_ct = self.true_op_ct[perm]
        self.true_candidate_free_time = self.true_candidate_free_time[perm]
        self.true_mch_free_time = self.true_mch_free_time[perm]
        self.candidate = self.candidate[perm]
        self.unscheduled_op_mask = self.unscheduled_op_mask[perm]
        self.idle_acc = self.idle_acc[perm]
        self.machine_assigned_count = self.machine_assigned_count[perm]
        self.shaping_potential = self.shaping_potential[perm]
        self.mask = self.mask[perm]
        self.env_done = self.env_done[perm]
        self.op_ct_lb = self.op_ct_lb[perm]
        self.max_endTime = self.max_endTime[perm]
        self.candidate_pt = self.candidate_pt[perm]
        self.candidate_process_relation = self.candidate_process_relation[perm]

    def construct_candidate_features(self):
        """
            [1] feasible_mas_ratio : 可用机器数/总机器数
            [2] job_ready : 对应工件的释放时间(norm)
            [3] rem_ops :  工件剩余操作数 / 工件总操作数    (加工进度)
            [4] rem_work : 工件剩余工作量(未调度操作的平均加工时间之和)
            [5] p_mean : 平均加工时间
            [6] p_span : 加工时间跨度
            [7] criticality : 关键路径松弛(1=在关键路径上, slack越小越关键)
        :return: fea_j[B,J,8] 若其中有工件已完工那么用mask将对应特征置为0
        """
        feasible_mas = self.compatible_op[self.env_idxs[:, None], self.candidate] # 操作能被多少台机器加工
        feasible_mas_ratio = feasible_mas / self.number_of_machines
        job_ready = self.candidate_free_time
        rem_ops = (self.job_length - self.candidate + self.job_first_op_id) / self.job_length

        rem_work = np.maximum(0,
            self.op_mean_pt_cumsum[self.env_idxs[:, None], self.job_last_op_id + 1] -
            self.op_mean_pt_cumsum[self.env_idxs[:, None], self.candidate]
        )  # [B, J]
        rem_work = np.where(self.mask, 0.0, rem_work)

        p_mean = self.op_mean_pt[self.env_idxs[:, None], self.candidate]
        p_span = self.pt_span[self.env_idxs[:, None], self.candidate]

        op_ct_lb_cand = self.op_ct_lb[self.env_idxs[:, None], self.candidate]  # [B, J]
        raw_delay_ratio = np.log1p(
            np.maximum(0.0, self.candidate_free_time - op_ct_lb_cand) / (op_ct_lb_cand + 1e-8)
        )
        delay_ratio = raw_delay_ratio * (1.0 - feasible_mas_ratio)

        # 关键路径松弛：工件预计完工(头+尾) 与全局 makespan 下界的接近度
        job_finish_lb = self.op_ct_lb[self.env_idxs[:, None], self.job_last_op_id]  # [B,J] 头+尾
        op_ct_lb_visible = np.where(self.op_valid_mask, self.op_ct_lb, 0.0)
        makespan_lb = np.max(op_ct_lb_visible, axis=1, keepdims=True)  # [B,1]
        slack = makespan_lb - job_finish_lb  # >=0, 越小越关键
        criticality = 1.0 - slack / (makespan_lb + 1e-8)  # 1=在关键路径上

        self.fea_j = np.stack((feasible_mas_ratio, job_ready, rem_ops, rem_work, p_mean, p_span, delay_ratio, criticality), axis=2)

        mask = self.mask[:,:,None]
        self.fea_j = np.where(mask, 0, self.fea_j)

        # 每个env还有多少个有效节点
        num_left_nodes = np.sum(~self.mask, axis=1, keepdims=True) # [B, 1]
        valid_env = (num_left_nodes.squeeze(-1) > 0)               # [B]

        if valid_env.any():
            fea_j_valid = self.fea_j[valid_env]  # [Bv, J, 6]
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
        utilization = (self.mch_free_time - self.idle_acc) / (self.mch_free_time + 1e-8)
        self.fea_m = np.stack((feasible_ops_norm, mach_ready, expect_workload, idle, utilization), axis=2)

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

        pair_features = [pt_out, start, wait_job, wait_mach, ratio_m, ratio_op]
        if self.revision_variant == 'l':
            queue_length = self.machine_assigned_count / self.number_of_ops_per_env[:, None]
            processing_load = np.maximum(0.0, self.mch_free_time - self.idle_acc)
            cumulative_load = processing_load / (self.total_expected_work[:, None] + 1e-8)
            pair_shape = (self.number_of_envs, self.number_of_jobs, self.number_of_machines)
            pair_features.extend([
                np.broadcast_to(queue_length[:, None, :], pair_shape),
                np.broadcast_to(cumulative_load[:, None, :], pair_shape),
            ])
        self.fea_pairs = np.stack(pair_features, axis=3)

    def construct_waiting_features(self):
        """Fixed-size summaries of operations after the current candidate."""
        waiting_count = np.maximum(0, self.job_last_op_id - self.candidate)
        waiting_start = np.minimum(self.candidate + 1, self.job_last_op_id + 1)
        waiting_work = (
            self.op_mean_pt_cumsum[self.env_idxs[:, None], self.job_last_op_id + 1]
            - self.op_mean_pt_cumsum[self.env_idxs[:, None], waiting_start]
        )
        waiting_flex_sum = (
            self.op_flexibility_cumsum[self.env_idxs[:, None], self.job_last_op_id + 1]
            - self.op_flexibility_cumsum[self.env_idxs[:, None], waiting_start]
        )
        count_ratio = waiting_count / self.job_length
        workload_ratio = waiting_work / (self.job_total_expected_work + 1e-8)
        mean_flexibility = waiting_flex_sum / (waiting_count + 1e-8)
        mean_processing_time = waiting_work / (waiting_count + 1e-8)
        self.fea_waiting = np.stack(
            (count_ratio, workload_ratio, mean_flexibility, mean_processing_time), axis=2
        )
        self.fea_waiting = np.where(self.mask[:, :, None], 0.0, self.fea_waiting)
