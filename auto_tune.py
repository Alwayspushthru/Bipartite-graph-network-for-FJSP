"""
Auto-tune harness for the FJSP bipartite-graph policy.

Implements the standard operating procedure of experiments/AUTO_EXPERIMENT.md for
the goal in experiments/goal_20260604_indist_composite.md: search params.py one
dimension at a time and keep only configs that *significantly* beat the greedy
in-distribution composite baseline.

Two-stage budget (manual 3.3 — same budget within a stage):
  Stage A (screen)  : 1 seed,  --updates-a updates, greedy. Culls candidates that
                      are clearly worse than the base config at the SAME short
                      budget (not vs the full-budget baseline — that would be an
                      apples-to-oranges comparison).
  Stage B (confirm) : --seeds-b seeds, --updates-b updates, greedy. A candidate is
                      kept only if its N-seed composite mean beats the baseline by
                      more than the significance threshold 2*sigma/sqrt(N).

Config is passed to train/eval as CLI flags; params.py is never edited. The
current best (coordinate-ascent) config lives in experiments/best_config.json.
"keep" updates that file (+ optional git commit); "revert" only appends a
results.tsv row. Training AND eval run as subprocesses so structural candidates
(hidden_dim, layers) build the matching architecture in both.

Examples
--------
  # plumbing smoke test (tiny budgets, 2 seeds, no git):
  python auto_tune.py --smoke

  # real run over the lr and eps_clip dimensions, auto-commit keeps:
  python auto_tune.py --dims lr eps_clip --git
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import date

PY          = sys.executable
SRC_CKPT    = './trained_network/10x5.pth'        # train.py SD3 10x5 output
BEST_CONFIG = './experiments/best_config.json'
RESULTS_TSV = './results.tsv'
GOAL_ID     = 'indist_composite'
METRIC_NAME = 'indist_composite_gap%'

DATA_SOURCE = 'SD3'
N_J, N_M    = 10, 5

# greedy composite baseline + run-to-run noise floor (sigma) from
# test_results/noise_floor_raw.csv, 4 in-distribution groups, seeds 300-309.
BASELINE0   = 9.245
SIGMA0      = 2.598

# params.py defaults for every searchable flag (used to skip no-op candidates).
DEFAULTS = {
    'lr': '3e-4', 'eps_clip': '0.2', 'entloss_coef': '0.01',
    'gae_lambda': '0.98', 'k_epochs': '4',
    'hidden_dim_actor': '64', 'hidden_dim_critic': '64',
    'num_bigraph_layers': '2', 'minibatch_size': '1024',
}

# Search space, ordered by change/payoff ratio (sheet section 4). Each entry maps
# a dimension name -> candidate values. 'hidden_dim' is a meta-dim expanding to
# both actor and critic widths; all others map to a single flag of the same name.
SEARCH = [
    ('lr',                 ['1e-4', '1e-3']),
    ('eps_clip',           ['0.1', '0.3']),
    ('entloss_coef',       ['0.001', '0.05']),
    ('gae_lambda',         ['0.90', '0.95']),
    ('k_epochs',           ['2', '8']),
    ('hidden_dim',         ['32', '128']),
    ('num_bigraph_layers', ['1', '3']),
    ('minibatch_size',     ['512', '2048']),
]


def expand(dim, value):
    """Turn a (dim, value) into the concrete {flag: value} override(s)."""
    if dim == 'hidden_dim':
        return {'hidden_dim_actor': value, 'hidden_dim_critic': value}
    return {dim: value}


def overrides_to_args(ov):
    args = []
    for k, v in ov.items():
        args += [f'--{k}', str(v)]
    return args


def overrides_str(ov):
    return ' '.join(f'{k}={v}' for k, v in sorted(ov.items())) or '(defaults)'


# ----------------------------------------------------------------------------- IO
def load_best():
    if os.path.exists(BEST_CONFIG):
        with open(BEST_CONFIG, encoding='utf-8') as f:
            d = json.load(f)
        return d.get('overrides', {}), float(d.get('baseline', BASELINE0))
    return {}, BASELINE0


def save_best(overrides, baseline):
    os.makedirs(os.path.dirname(BEST_CONFIG), exist_ok=True)
    with open(BEST_CONFIG, 'w', encoding='utf-8') as f:
        json.dump({'overrides': overrides, 'baseline': baseline,
                   'sigma': SIGMA0, 'updated': str(date.today())}, f, indent=2)


def next_exp_id():
    mx = 0
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, encoding='utf-8') as f:
            for line in f:
                tok = line.split('\t', 1)[0].strip()
                if tok.startswith('exp_') and tok[4:].isdigit():
                    mx = max(mx, int(tok[4:]))
    return mx + 1


def append_row(exp_id, hypo, value, baseline, changes, budget, status, sha=''):
    delta = value - baseline if value is not None else float('nan')
    row = [f'exp_{exp_id:03d}', str(date.today()), GOAL_ID, hypo, METRIC_NAME,
           f'{value:.3f}' if value is not None else 'NA', f'{baseline:.3f}',
           f'{delta:+.3f}' if value is not None else 'NA',
           changes, budget, status, sha]
    with open(RESULTS_TSV, 'a', encoding='utf-8') as f:
        f.write('\t'.join(row) + '\n')


# -------------------------------------------------------------------- subprocess
def run(cmd):
    print('  $ ' + ' '.join(cmd))
    r = subprocess.run(cmd)
    return r.returncode == 0


def train_one(seed, updates, overrides, tag):
    """Train one seed, copy the checkpoint to ./trained_network/<tag>.pth."""
    before = os.path.getmtime(SRC_CKPT) if os.path.exists(SRC_CKPT) else -1
    cmd = [PY, 'train.py', '--seed_train', str(seed), '--data_source', DATA_SOURCE,
           '--n_j', str(N_J), '--n_m', str(N_M), '--max_updates', str(updates),
           '--use_tensorboard', 'false'] + overrides_to_args(overrides)
    if not run(cmd):
        print(f'  ! train failed (seed {seed})'); return None
    if not os.path.exists(SRC_CKPT) or os.path.getmtime(SRC_CKPT) <= before:
        print(f'  ! checkpoint not updated (no val improvement?) seed {seed}')
        return None
    dst = f'./trained_network/{tag}.pth'
    shutil.copyfile(SRC_CKPT, dst)
    return tag


def eval_models(tags, overrides, out_json):
    # EVAL_OUT goes via env (not a CLI flag): params.py parse_args() is strict.
    cmd = [PY, 'eval_composite.py', '--test_model', *tags] + overrides_to_args(overrides)
    env = dict(os.environ, EVAL_OUT=out_json)
    print('  $ EVAL_OUT=' + out_json + ' ' + ' '.join(cmd))
    if subprocess.run(cmd, env=env).returncode != 0:
        print('  ! eval failed'); return None
    with open(out_json, encoding='utf-8') as f:
        return json.load(f)


def git_commit(paths, msg):
    subprocess.run(['git', 'add', *paths])
    subprocess.run(['git', 'commit', '-m', msg])
    r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                       capture_output=True, text=True)
    return r.stdout.strip()


def cleanup(tags):
    for t in tags:
        p = f'./trained_network/{t}.pth'
        if os.path.exists(p):
            os.remove(p)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', nargs='+', default=[d for d, _ in SEARCH],
                    help='which search dimensions to run (default: all)')
    ap.add_argument('--updates-a', type=int, default=200, help='Stage A budget')
    ap.add_argument('--updates-b', type=int, default=1000, help='Stage B budget')
    ap.add_argument('--seeds-b', type=int, default=5, help='Stage B seed count')
    ap.add_argument('--seed-start', type=int, default=300)
    ap.add_argument('--a-margin', type=float, default=1.0,
                    help='Stage A cull margin (pp worse than base @ short budget)')
    ap.add_argument('--max-candidates', type=int, default=0,
                    help='0 = no cap')
    ap.add_argument('--git', action='store_true', help='auto-commit keep/revert')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny budgets/seeds for a plumbing test (overrides above)')
    args = ap.parse_args()

    if args.smoke:
        # Must exceed validate_timestep (default 10) or train.py never validates
        # and never calls save_model() -> the mtime guard reports no checkpoint.
        args.updates_a, args.updates_b, args.seeds_b = 12, 12, 2
        print('[smoke] updates_a=12 updates_b=12 seeds_b=2')

    os.makedirs('./test_results', exist_ok=True)
    base, baseline = load_best()
    thr = 2 * SIGMA0 / (args.seeds_b ** 0.5)
    print(f'baseline={baseline:.3f}%  sigma={SIGMA0}  N={args.seeds_b}  '
          f'significance threshold=2sigma/sqrt(N)={thr:.3f} pp')
    print(f'base overrides: {overrides_str(base)}\n')

    seeds_b = list(range(args.seed_start, args.seed_start + args.seeds_b))
    stageA_ref = None          # composite of `base` at short budget
    stageA_ref_sig = None      # signature of base when ref was measured
    no_improve = 0
    n_done = 0

    def base_sig(b):
        return json.dumps(b, sort_keys=True)

    for dim in args.dims:
        values = dict(SEARCH).get(dim)
        if values is None:
            print(f'[skip] unknown dim {dim}'); continue

        for value in values:
            flags = expand(dim, value)
            # skip no-op (value already the current base/default)
            if all(str(base.get(k, DEFAULTS[k])) == str(v) for k, v in flags.items()):
                continue

            if args.max_candidates and n_done >= args.max_candidates:
                print(f'\n[stop] reached --max-candidates={args.max_candidates}')
                return
            if no_improve >= 3:
                print('\n[stop] 3 consecutive non-significant candidates '
                      '(sheet 7) — revisit the hypothesis.')
                return

            cand = dict(base); cand.update(flags)
            exp_id = next_exp_id()
            change = overrides_str(flags)
            print('=' * 70)
            print(f'exp_{exp_id:03d} | dim={dim} -> {change} | base={overrides_str(base)}')
            print('=' * 70)
            n_done += 1

            # -- Stage A reference (base @ short budget), (re)measured if base moved
            if stageA_ref is None or stageA_ref_sig != base_sig(base):
                print('-- measuring Stage A reference (base @ short budget) --')
                rtag = train_one(args.seed_start, args.updates_a, base, '_ref_A')
                if rtag is None:
                    print('  ! could not measure Stage A ref; aborting'); return
                ref = eval_models([rtag], base, './test_results/_ref_A.json')
                cleanup([rtag])
                if ref is None:
                    return
                stageA_ref = ref['composite_mean']
                stageA_ref_sig = base_sig(base)
                print(f'  Stage A ref composite = {stageA_ref:.3f}%')

            # -- Stage A: candidate @ short budget, 1 seed
            print('-- Stage A (screen) --')
            atag = train_one(args.seed_start, args.updates_a, cand,
                             f'auto_e{exp_id:03d}_A_s{args.seed_start}')
            if atag is None:
                append_row(exp_id, change, None, baseline, change,
                           f'{args.updates_a}upd x1 greedy', 'fail_train')
                no_improve += 1; continue
            a = eval_models([atag], cand, f'./test_results/auto_e{exp_id:03d}_A.json')
            cleanup([atag])
            if a is None:
                append_row(exp_id, change, None, baseline, change,
                           f'{args.updates_a}upd x1 greedy', 'fail_eval')
                no_improve += 1; continue
            a_comp = a['composite_mean']
            if a_comp > stageA_ref + args.a_margin:
                print(f'  culled: Stage A {a_comp:.3f}% > ref {stageA_ref:.3f}% '
                      f'+ {args.a_margin}')
                append_row(exp_id, change, a_comp, stageA_ref, change,
                           f'{args.updates_a}upd x1 greedy', 'revert_A')
                no_improve += 1
                if args.git:
                    git_commit([RESULTS_TSV], f'[log] exp#{exp_id:03d}: revert_A '
                               f'({a_comp:.2f} vs ref {stageA_ref:.2f})')
                continue
            print(f'  survives Stage A ({a_comp:.3f}% <= ref+margin)')

            # -- Stage B: candidate @ full budget, N seeds
            print(f'-- Stage B (confirm, {args.seeds_b} seeds) --')
            btags = []
            for s in seeds_b:
                t = train_one(s, args.updates_b, cand,
                              f'auto_e{exp_id:03d}_B_s{s}')
                if t: btags.append(t)
            if len(btags) < 2:
                print('  ! <2 Stage B seeds trained; logging fail')
                append_row(exp_id, change, None, baseline, change,
                           f'{args.updates_b}upd x{args.seeds_b} greedy', 'fail_train')
                cleanup(btags); no_improve += 1; continue
            b = eval_models(btags, cand, f'./test_results/auto_e{exp_id:03d}_B.json')
            if b is None:
                append_row(exp_id, change, None, baseline, change,
                           f'{args.updates_b}upd x{args.seeds_b} greedy', 'fail_eval')
                cleanup(btags); no_improve += 1; continue
            m = b['composite_mean']
            improvement = baseline - m
            budget = f'{args.updates_b}upd x{len(btags)} greedy'
            print(f'  Stage B composite={m:.3f}%  baseline={baseline:.3f}%  '
                  f'improvement={improvement:+.3f} pp  (need >{thr:.3f})')

            if improvement > thr:
                print('  >>> KEEP (significant)')
                base = cand
                baseline = m
                save_best(base, baseline)
                sha = ''
                if args.git:
                    sha = git_commit([RESULTS_TSV, BEST_CONFIG],
                                     f'[keep] exp#{exp_id:03d}: '
                                     f'{baseline + improvement:.2f}->{m:.2f} | {change}')
                append_row(exp_id, change, m, baseline + improvement, change,
                           budget, 'keep', sha)
                cleanup(btags)            # winning config recorded in best_config
                no_improve = 0
            else:
                print('  >>> revert (not significant)')
                append_row(exp_id, change, m, baseline, change, budget, 'revert')
                cleanup(btags)
                no_improve += 1
                if args.git:
                    git_commit([RESULTS_TSV], f'[log] exp#{exp_id:03d}: revert '
                               f'({baseline:.2f}->{m:.2f})')

    print('\n[done] search space exhausted. '
          f'best={overrides_str(base)}  baseline={baseline:.3f}%')


if __name__ == '__main__':
    main()
