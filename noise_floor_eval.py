"""
Training-time noise floor evaluator.

Given K models that were trained with K different `--seed_train` values (and are
otherwise identical), evaluate each one GREEDILY on every requested eval set and
report, per dataset group:

    mean gap   = average of each model's per-group mean gap   (the metric itself)
    σ (noise)  = std ACROSS the K models of that per-group mean gap  (the floor)

σ is the run-to-run measurement noise: another config's improvement is only
credible if it clearly exceeds ~2σ. Grouping mirrors test.py exactly:
  - BenchData            -> baseline 'benchname'
  - SD1 / SD2 (synthetic)-> baseline 'dataname' with trailing _<idx> stripped

Models and eval sets come from the standard params:
    python noise_floor_eval.py --test_data SD1 SD2 BenchData \
                               --test_model nf_seed300 nf_seed301 ...
"""

import os
import numpy as np
import pandas as pd

from params import configs
from utils.data_utils import load_data_from_files
from test import test_greedy_strategy, load_baseline


def group_labels(data_name, bdf):
    """Same grouping test.py uses for its per-group gap breakdown."""
    if data_name == 'BenchData':
        return bdf['benchname'].values
    return bdf['dataname'].str.replace(r'_\d+$', '', regex=True).values


def main(config):
    models = config.test_model
    if len(models) < 2:
        raise SystemExit("Need >= 2 models (seeds) to measure a noise floor; "
                         f"got {len(models)}: {models}")

    os.makedirs('./test_results', exist_ok=True)

    # Preload each eval set's data + baseline + group layout once.
    sets = {}
    for name in config.test_data:
        bdf = load_baseline(name)
        if bdf is None:
            print(f"[skip] no *Solution.csv baseline for {name}")
            continue
        data = load_data_from_files(f'./data/{name}')
        gl = group_labels(name, bdf)
        sets[name] = {
            'data':   data,
            'base':   bdf['ub'].values,
            'glabel': gl,
            'groups': sorted(np.unique(gl)),
        }

    if not sets:
        raise SystemExit("No eval set had a baseline csv; nothing to evaluate.")

    # results[set][group] = list of per-model mean gaps (one entry per seed)
    results = {name: {g: [] for g in sets[name]['groups']} for name in sets}
    raw_rows = []  # per-model per-group, for inspection

    for mi, model in enumerate(models):
        model_path = f'./trained_network/{model}.pth'
        if not os.path.exists(model_path):
            print(f"[warn] missing model, skipped: {model_path}")
            continue
        for name, s in sets.items():
            # BenchData is heterogeneous real data -> single-instance like test.py;
            # synthetic sets get the configured (possibly packed) batch size.
            bs = 1 if name == 'BenchData' else config.test_batch_size
            res = test_greedy_strategy(s['data'], model_path, config.seed_test, bs)
            gaps = (res[:, 0] - s['base']) / s['base'] * 100
            for g in s['groups']:
                gmean = gaps[s['glabel'] == g].mean()
                results[name][g].append(gmean)
                raw_rows.append({'model': model, 'eval_set': name,
                                 'group': g, 'mean_gap(%)': round(gmean, 3)})
        print(f"  done {mi + 1}/{len(models)}: {model}")

    # ---- report ----
    print("\n" + "=" * 72)
    print(f"Training-time noise floor | {len(models)} seeds | greedy")
    print("=" * 72)

    summary_rows = []
    for name, s in sets.items():
        print(f"\n[{name}]")
        print(f"  {'group':16s} {'mean gap':>10s} {'σ (noise)':>12s} "
              f"{'min':>9s} {'max':>9s} {'n':>4s}")
        for g in s['groups']:
            v = np.array(results[name][g])
            sd = v.std(ddof=1) if len(v) > 1 else float('nan')
            print(f"  {g:16s} {v.mean():9.3f}% {sd:11.3f}% "
                  f"{v.min():8.2f}% {v.max():8.2f}% {len(v):4d}")
            summary_rows.append({
                'eval_set': name, 'group': g, 'n_seeds': len(v),
                'mean_gap(%)': round(v.mean(), 3),
                'noise_std(%)': round(sd, 3),
                'min(%)': round(float(v.min()), 2),
                'max(%)': round(float(v.max()), 2),
            })

    pd.DataFrame(summary_rows).to_csv('./test_results/noise_floor_summary.csv', index=False)
    pd.DataFrame(raw_rows).to_csv('./test_results/noise_floor_raw.csv', index=False)
    print("\nsaved -> ./test_results/noise_floor_summary.csv  (mean & σ per group)")
    print("saved -> ./test_results/noise_floor_raw.csv      (per-model per-group gaps)")


if __name__ == "__main__":
    main(configs)
