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
        self.op_fea_dim = 6
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
        self.dynamic_pair_mask = np.copy(~dynamic_pair)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)

        # Construct features
        self.construct_candidate_features() # [B,J,8]
        self.construct_mch_features()
        self.construct_pair_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = self.init_quality

        self.old_state.update(self.fea_j,self.fea_m,self.fea_pairs,self.candidate,self.mask,
                              self.dynamic_pair_mask,self.device)

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
        self.compatible_op = np.copy(self.compatible_op)
        self.compatible_mch = np.copy(self.compatible_mch)

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

        self.mch_cum_load = np.zeros((self.number_of_envs, self.number_of_machines)) #机器m到目前为止被分配过的总加工时间之和
        self.idle_acc = np.zeros((self.number_of_envs, self.number_of_machines))  # 机器m的空闲时间累计

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)
        self.env_done = np.zeros(self.number_of_envs,dtype=bool)


    def step(self,actions):
        """
            perform the state transition & return the next state and reward
            :param actions: the action list with shape [E]
            :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]

        self.step_count += 1

        for i in range(self.number_of_envs):
            self.env_done[i] = (self.step_count >= self.number_of_ops_per_env[i])  ####

        active_envs = ~ self.env_done # (B,) 还没调度完的环境是 True

        if (self.reverse_process_relation[self.env_idxs[active_envs], chosen_op[active_envs], chosen_mch[active_envs]]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        # update candidate and message
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job]) & active_envs # 候选动作推进flag，只要选择的动作不是工件最后一个操作
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag # 那么对应的工件就+1
        self.mask[self.env_idxs, chosen_job] = np.where(active_envs, ~candidate_add_flag, self.mask[self.env_idxs, chosen_job])
        # 仅对 “活跃环境（active_envs=True）” 的位置，用 ~candidate_add_flag 覆盖原有值；对 “非活跃环境（active_envs=False）” 的位置，保留 self.mask 原有值不变
        pt = self.unmasked_op_pt[self.env_idxs, chosen_op, chosen_mch] * active_envs  # (B,) 每个环境中选择的操作的加工时间
        self.mch_cum_load[self.env_idxs, chosen_mch] += pt

        mask_temp = candidate_add_flag # 用来作为布尔掩码把env分成两组
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1] # 更新候选操作的加工时间
        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1] # 更新可行性关系

        finished_job_mask = (~mask_temp) & active_envs
        self.candidate_process_relation[finished_job_mask, chosen_job[finished_job_mask]] = 1 # 对于已经完成加工的工件，全部置为1
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)  ### 这个变量可以不要

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],self.mch_free_time[self.env_idxs, chosen_mch]) # 候选工件释放时间与目标机器释放时间，选大的那个时间
        self.op_ct[self.env_idxs, chosen_op] = np.where(active_envs, chosen_op_st + self.op_pt[self.env_idxs, chosen_op, chosen_mch], self.op_ct[self.env_idxs, chosen_op])

        idle_inc = np.maximum(0.0, chosen_op_st - self.mch_free_time[self.env_idxs, chosen_mch])
        self.idle_acc[self.env_idxs, chosen_mch] += idle_inc * active_envs

        self.candidate_free_time[self.env_idxs, chosen_job] = np.where(active_envs, self.op_ct[self.env_idxs, chosen_op], self.candidate_free_time[self.env_idxs, chosen_job])
        self.mch_free_time[self.env_idxs, chosen_mch] = np.where(active_envs, self.op_ct[self.env_idxs, chosen_op], self.mch_free_time[self.env_idxs, chosen_mch])

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],self.true_mch_free_time[self.env_idxs, chosen_mch])
        self.true_op_ct[self.env_idxs, chosen_op] = np.where(active_envs, true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch], self.true_op_ct[self.env_idxs, chosen_op])
        self.true_candidate_free_time[self.env_idxs, chosen_job] = np.where(active_envs,
                                                                            self.true_op_ct[self.env_idxs, chosen_op],
                                                                            self.true_candidate_free_time[
                                                                                self.env_idxs, chosen_job])
        self.true_mch_free_time[self.env_idxs, chosen_mch] = np.where(active_envs,
                                                                      self.true_op_ct[self.env_idxs, chosen_op],
                                                                      self.true_mch_free_time[
                                                                          self.env_idxs, chosen_mch])

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[self.env_idxs, chosen_op])

        self.construct_candidate_features()

        self.construct_mch_features()

        self.construct_pair_features()

        diff = (self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]) * active_envs

        mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,np.newaxis])
        self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        op_ct_lb_visible = np.where(self.op_valid_mask, self.op_ct_lb, 0)
        reward = self.max_endTime - np.max(op_ct_lb_visible, axis=1)
        self.max_endTime = np.max(op_ct_lb_visible, axis=1)

        true_candidate = np.where(self.mask, -1, self.candidate)

        self.state.update(self.fea_j, self.fea_m, self.fea_pairs,true_candidate, self.mask,
                          self.dynamic_pair_mask,self.device)

        return self.state, np.array(reward), self.env_done

    def construct_candidate_features(self):
        """
            [1] feasible_mas_ratio : 可用机器数/总机器数
            [2] job_ready : 对应工件的释放时间(norm)
            [3] rem_ops :  工件剩余操作数 / 工件总操作数    (加工进度)
            [4] rem_work : 工件剩余工作量(未调度操作的平均加工时间之和)
            [5] p_mean : 平均加工时间
            [6] p_span : 加工时间跨度
        :return: fea_j[B,J,6] 若其中有工件已完工那么用mask将对应特征置为0
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

        self.fea_j = np.stack((feasible_mas_ratio,job_ready,rem_ops, rem_work, p_mean,p_span),axis=2,)

        mask = self.mask[:,:,None]
        self.fea_j = np.where(mask, 0, self.fea_j)

        # 针对已完工的工件进行归一化
        if (~self.env_done).any():
            num_left_nodes = np.sum(~self.mask, axis=1, keepdims=True) # 还剩多少工件
            mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes
            temp = np.where(mask, mean_fea_j[:, np.newaxis, :], self.fea_j)
            var_fea_j = np.var(temp, axis=1)
            std_fea_j = np.sqrt(var_fea_j * self.number_of_jobs / num_left_nodes)
            self.fea_j = (temp - mean_fea_j[:, np.newaxis, :]) / \
                         (std_fea_j[:, np.newaxis, :] + 1e-8)

    def construct_mch_features(self):
        """
            [1] feasible_ops_norm: 所有可加工操作数 / 所有还存在的工件数
            [2] mach_ready: 机器的空闲时间
            [3] workload: 机器总负载
            [4] idle: 空闲时间累计
        :return:fea_m[B,M,4]
        """
        feasible_ops = np.sum(~self.candidate_process_relation, axis=1)
        num_alive = np.sum(~self.mask, axis=1, keepdims=True)
        feasible_ops_norm = feasible_ops / (num_alive + 1e-8)
        mach_ready = self.mch_free_time
        workload = self.mch_cum_load
        idle = self.idle_acc
        self.fea_m = np.stack((feasible_ops_norm, mach_ready, workload, idle), axis=2)

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
