"""
Composite-metric evaluator for the auto-tune harness.

Computes the *primary metric* defined in
experiments/goal_20260604_indist_composite.md:

    indist_composite = unweighted mean of the GREEDY mean gap of the 4 SD3
    in-distribution groups:  SD1 30x10, SD1 40x10, SD2 30x10+mix, SD2 40x10+mix.

Unlike noise_floor_eval.py this works for N >= 1 model and dumps a small JSON the
orchestrator (auto_tune.py) reads back. It MUST be run as a subprocess carrying
the candidate's structural flags (--hidden_dim_*, --num_bigraph_layers, ...): the
global `ppo` in test.py is built from `configs` at import time, so the network
architecture has to match the checkpoint being loaded.

The output JSON path is taken from the EVAL_OUT environment variable (not a CLI
flag) because params.py calls strict parse_args() at import — an unknown flag
would crash. Only flags params.py defines may appear on argv.

    EVAL_OUT=test_results/c.json python eval_composite.py \
        --test_model auto_e001_B_s300 ... --hidden_dim_actor 128
"""

import os
import json
import numpy as np

from params import configs
# Importing test.py builds the global ppo from `configs` (with our CLI flags) and
# sets the default device — exactly the architecture our checkpoints expect.
from test import test_greedy_strategy, load_baseline
from utils.data_utils import load_data_from_files

# The 4 in-distribution groups, in (data_source, group_label) form. Group labels
# mirror test.py's grouping: SD1/SD2 strip the trailing _<idx> from `dataname`.
TARGET_SETS = ('SD1', 'SD2')


def group_labels(data_name, bdf):
    if data_name == 'BenchData':
        return bdf['benchname'].values
    return bdf['dataname'].str.replace(r'_\d+$', '', regex=True).values


def composite_for_model(model_name, seed_test, batch_size):
    """Return {group_key: mean_gap} for the 4 in-distribution groups."""
    model_path = f'./trained_network/{model_name}.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    per_group = {}
    for name in TARGET_SETS:
        bdf = load_baseline(name)
        if bdf is None:
            raise SystemExit(f"no baseline csv for {name}")
        data = load_data_from_files(f'./data/{name}')
        res = test_greedy_strategy(data, model_path, seed_test, batch_size)
        base = bdf['ub'].values
        gaps = (res[:, 0] - base) / base * 100
        gl = group_labels(name, bdf)
        for g in sorted(np.unique(gl)):
            per_group[f'{name}:{g}'] = float(gaps[gl == g].mean())
    return per_group


def main(config):
    models = config.test_model
    bs = config.test_batch_size

    per_model_composite = []
    per_model_groups = []
    for m in models:
        pg = composite_for_model(m, config.seed_test, bs)
        comp = float(np.mean(list(pg.values())))  # equal-weight over the 4 groups
        per_model_composite.append(comp)
        per_model_groups.append(pg)
        print(f"  {m}: composite={comp:.3f}%  " +
              "  ".join(f"{k}={v:.2f}" for k, v in pg.items()))

    arr = np.array(per_model_composite)
    # group means averaged across models (for diagnostics)
    group_keys = per_model_groups[0].keys()
    group_mean = {k: float(np.mean([pg[k] for pg in per_model_groups]))
                  for k in group_keys}

    out = {
        'n': len(models),
        'models': list(models),
        'composite_mean': float(arr.mean()),
        'composite_std': float(arr.std(ddof=1)) if len(arr) > 1 else None,
        'composite_min': float(arr.min()),
        'composite_max': float(arr.max()),
        'per_model_composite': per_model_composite,
        'per_group_mean': group_mean,
    }

    print(f"\ncomposite_mean={out['composite_mean']:.3f}%  "
          f"std={out['composite_std'] if out['composite_std'] is None else round(out['composite_std'],3)}  "
          f"n={out['n']}")

    out_path = os.environ.get('EVAL_OUT', '')
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {out_path}")
    return out


if __name__ == '__main__':
    main(configs)
