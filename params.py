import argparse

def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='Arguments for FJSP')

# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')

# args for file_name
parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')

# args for AutoExperiment
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')

# args for seed
parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
# args for data load
parser.add_argument('--data_source', type=str, default='SD3', help='Suffix of test data')

# args for network
parser.add_argument('--fea_j_input_dim', type=int, default=8, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=5, help='Dimension of machine raw feature vectors')
parser.add_argument('--fea_pair_input_dim', type=int, default=6, help='Dimension of pair raw feature vectors')

parser.add_argument('--num_bigraph_layers', type=int, default=2)

# args for ablation study (single switch; one variant per run, 'none' = full model)
parser.add_argument('--ablation', type=str, default='none',
                    choices=['none', 'mean_agg', 'no_pair_bias', 'no_gru'],
                    help='Ablate one core design component. '
                         'mean_agg: replace bipartite attention with uniform mean aggregation; '
                         'no_pair_bias: drop edge-feature modulation of attention scores; '
                         'no_gru: zero out the cross-step GRU history (Markovian policy).')

parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Actor MLP layers')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension for actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Critic MLP layers')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension for critic')

# args for PPO Algorithm
parser.add_argument('--num_envs', type=int, default=20, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')

# args for training
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')

# args for training log
parser.add_argument('--log_dir', type=str, default='./runs', help='Root directory for TensorBoard logs')
parser.add_argument('--run_name', type=str, default='exp', help='Run name for TensorBoard logs')
parser.add_argument('--model_name', type=str, default='',
                    help='Checkpoint filename (without .pth) saved under trained_network/. '
                         'Empty = derive from size, e.g. 10x5 (overwrites the canonical file). '
                         'Set a unique name to avoid clobbering, e.g. 10x5_baseline.')
parser.add_argument('--log_interval', type=int, default=10, help='Interval (updates) for train/env logging')
parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='Whether to enable TensorBoard logging')

# args for test
parser.add_argument('--test_data', nargs='+', default=['SD2'], help='Data source folders to test (SD1/SD2/BenchData)')
parser.add_argument('--test_model', nargs='+', default=['10x5_1'], help='Model names under trained_network/ (without .pth)')


# args for testData to excel
parser.add_argument('--sort_flag', type=str2bool, default=True,
                    help='Whether sorting the printed results by the makespan')

# args for inference mode
parser.add_argument('--n_samples', type=int, default=1,
                    help='Number of stochastic rollouts per instance at test time. 1 = greedy (argmax).')
parser.add_argument('--beam_width', type=int, default=1,
                    help='Beam search width at test time. >1 keeps the K most probable partial '
                         'schedules each step and reports the best final makespan. Takes precedence '
                         'over n_samples. 1 (default) = disabled.')
parser.add_argument('--beam_stochastic', type=str2bool, default=False,
                    help='When beam_width>1, use stochastic beam search (Gumbel-top-k, i.e. '
                         'sampling K schedules without replacement) instead of deterministic '
                         'top-K. Adds sampling-style diversity to beam; helps on OOD/high-'
                         'flexibility data (SD2/BenchData) where deterministic beam collapses.')
parser.add_argument('--test_batch_size', type=int, default=0,
                    help='How many same-shape instances to run through the network at once during '
                         'greedy testing. 0 (default) = pack each size class in one batch (e.g. all '
                         '100 of SD2 30x10 together). Set a positive number to cap the batch if GPU '
                         'memory is tight. BenchData is always run single-instance regardless.')

# args for selfplay
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')


configs = parser.parse_args()
