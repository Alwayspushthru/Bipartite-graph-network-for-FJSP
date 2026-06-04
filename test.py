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

def _log1mexp(a):
    """
    Numerically stable log(1 - exp(a)) for a <= 0 (a == 0 -> -inf).
    Two-branch trick (Mächler 2012): use log(-expm1) when a is close to 0,
    log1p(-exp) when a is very negative.
    """
    return torch.where(a > -0.6931471805599453,
                       torch.log(-torch.expm1(a)),
                       torch.log1p(-torch.exp(a)))


def _conditional_gumbel_shift(g_phi, g_parent):
    """
    Shift independent child Gumbels `g_phi` (one Gumbel(phi_child) per action of a
    parent) so that their per-parent max equals the parent's perturbed value
    `g_parent` — the core of Kool et al. (2019) "Stochastic Beams and Where to
    Find Them". This makes top-K over the whole perturbed frontier equivalent to
    sampling K complete schedules *without replacement* from the policy.

    g_phi:    [K, A]  independent Gumbel(phi_child)
    g_parent: [K, 1]  parent perturbed value (the max the children must hit)
    returns:  [K, A]  conditioned Gumbels with max_a == g_parent per row
    """
    z = g_phi.max(dim=1, keepdim=True).values                      # [K,1] per-parent max
    v = g_parent - g_phi + _log1mexp(g_phi - z)                    # [K,A]
    return g_parent - v.clamp(min=0) - torch.log1p(torch.exp(-v.abs()))


def test_beam_strategy(data_set, model_path, seed, beam_width, stochastic=False):
    """
    Beam search over the construction process: keep `beam_width` partial schedules
    at every step instead of one (greedy) or `n_samples` i.i.d. rollouts (sampling).

    Each step:
      1. forward all K beams -> pi [K, J*M];
      2. score every child = parent_cum_logprob + log pi  -> [K, J*M];
      3. keep the global top-K children (top-K over K*J*M);
      4. reorder the env's dynamic state + GRU history by each child's parent,
         then apply the chosen actions.

    With `stochastic=False` this is deterministic beam: step 2 ranks by raw
    cumulative log-prob, so the K beams collapse onto the policy's mode (good when
    the policy is reliable/sharp, e.g. SD1; weak under OOD/high-flexibility data
    where exploration matters).

    With `stochastic=True` it is *stochastic beam search* (Gumbel-top-k): the
    cumulative log-probs are perturbed with conditional Gumbel noise so that the
    top-K frontier is exactly K schedules sampled WITHOUT REPLACEMENT. This keeps
    beam's structured pruning but restores sampling's diversity (no duplicate
    trajectories), and in expectation dominates i.i.d. Sampling×K.

    Every beam of one instance schedules the same number of operations, so all
    sequences share length — cumulative log-prob is a fair, length-unbiased rank
    and no normalization is needed. The returned makespan is the *best* among the
    K final complete schedules (the perturbation only steers the search; the
    objective decides the winner).

    Cost is the same order as Sampling×K: one batch-K forward per step.
    """
    test_result_list = []
    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ppo.policy.eval()

    env = FJSPEnv(device)
    K = beam_width

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        state = env.set_initial_data([data_set[0][i]] * K, [data_set[1][i]] * K)
        h_hist = torch.zeros(K, ppo.policy.hist_dim, device=device)
        # Only beam 0 expands on the first step (all beams are identical copies),
        # so the first selection yields K distinct children rather than K copies.
        cum_logprob = torch.full((K,), float('-inf'), device=device)
        cum_logprob[0] = 0.0
        # Perturbed value G per beam (only used when stochastic). Root G is 0 WLOG:
        # a single common ancestor's value shifts all leaves monotonically and so
        # cannot change the top-K ordering.
        g_beam = torch.full((K,), float('-inf'), device=device)
        g_beam[0] = 0.0

        t1 = time.time()
        done = env.env_done.copy()
        while not done.all():
            with torch.no_grad():
                pi, _, h_new = ppo.policy(
                    state.fea_j_tensor,
                    state.fea_m_tensor,
                    state.fea_pairs_tensor,
                    state.dynamic_pair_mask_tensor,
                    h_hist,
                )
                A = pi.shape[1]
                log_pi = torch.log(pi)  # masked actions -> 0 -> -inf, never selected
                child_logprob = cum_logprob.unsqueeze(1) + log_pi  # [K, A] phi_child

                if stochastic:
                    # Independent Gumbel(phi_child), then condition so each parent's
                    # child-max equals the parent's own perturbed value g_beam.
                    u = torch.rand_like(child_logprob).clamp_min(1e-12)
                    gumbel = -torch.log(-torch.log(u))
                    g_phi = child_logprob + gumbel                  # [K, A]
                    rank = _conditional_gumbel_shift(g_phi, g_beam.unsqueeze(1))
                    # Masked actions and dead parents (all-children -inf) make the
                    # shift produce NaN (from -inf − -inf); NaN poisons topk
                    # ordering. Force every invalid child back to -inf.
                    rank = torch.where(torch.isfinite(child_logprob), rank,
                                       torch.full_like(rank, float('-inf')))
                else:
                    rank = child_logprob                            # rank by raw logprob

                top_rank, top_flat = torch.topk(rank.reshape(-1), K)
                parent = top_flat // A
                action = top_flat % A

                # Safety: if fewer than K valid children exist (tiny instances),
                # topk may return -inf slots whose action is masked/illegal. Fill
                # them with the best (always-finite) child to keep actions legal.
                bad = ~torch.isfinite(top_rank)
                if bad.any():
                    parent = torch.where(bad, parent[0], parent)
                    action = torch.where(bad, action[0], action)

                # Carry forward the selected children's true cumulative log-prob
                # (and perturbed value), gathered by the chosen (parent, action).
                flat_logprob = child_logprob.reshape(-1)
                cum_logprob = flat_logprob[parent * A + action]
                if stochastic:
                    g_beam = top_rank
                    if bad.any():
                        g_beam = torch.where(bad, top_rank[0], g_beam)

            h_hist = h_new[parent]
            env.reorder(parent.cpu().numpy())
            state, _, done = env.step(actions=action.cpu().numpy())
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
            beam_width = config.beam_width
            use_beam = beam_width > 1
            use_sampling = (not use_beam) and n_samples > 1
            if use_beam:
                mode_str = f"{'SBeam' if config.beam_stochastic else 'Beam'}×{beam_width}"
                model_prefix = f"Bgnn-{'SB' if config.beam_stochastic else 'B'}{beam_width}"
            elif use_sampling:
                mode_str = f"Sampling×{n_samples}"
                model_prefix = f"Bgnn-S{n_samples}"
            else:
                mode_str = "Greedy"
                model_prefix = "Bgnn-G"

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = save_direc + f'/{model[1]}_{model_prefix}_{timestamp}.xlsx'

            if True:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{data[1]}")
                print(f"Test mode: {mode_str}")
                if use_beam:
                    save_result = test_beam_strategy(data[0], model[0], config.seed_test,
                                                     beam_width, config.beam_stochastic)
                elif use_sampling:
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
