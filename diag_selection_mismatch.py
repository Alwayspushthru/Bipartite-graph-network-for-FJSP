"""
Diagnostic: is the cross-seed instability a TRAINING divergence or a model
SELECTION / transfer mismatch?

train.py keeps the checkpoint with the best SD3-validation makespan, but our
primary metric is the SD1/SD2 greedy composite gap. If a seed with a BAD
composite still has a GOOD (low) SD3-val makespan, then SD3-val does not predict
composite -> the apparent "instability" is a selection/transfer mismatch
(hypothesis A), not training divergence (hypothesis B).

Prints, per default-config seed checkpoint:
  SD3-val mean makespan (the selection metric)  vs  known SD1/SD2 composite gap.
Then reports the rank correlation between them.
"""
import numpy as np
from scipy.stats import spearmanr
from params import configs
from test import test_greedy_strategy
from utils.data_utils import load_data_from_files

# Known SD1/SD2 greedy composite per seed (from test_results/noise_floor_raw.csv).
COMPOSITE = {
    'nf_seed300': 8.546, 'nf_seed301': 6.596, 'nf_seed302': 7.468,
    'nf_seed303': 10.692, 'nf_seed304': 15.151, 'nf_seed305': 7.520,
    'nf_seed306': 10.764, 'nf_seed307': 7.042, 'nf_seed308': 10.554,
    'nf_seed309': 8.118,
}

vali = load_data_from_files('./data/data_train_vali/SD3/10x5')

rows = []
for name, comp in COMPOSITE.items():
    res = test_greedy_strategy(vali, f'./trained_network/{name}.pth',
                               configs.seed_test, 0)
    sd3_mk = float(res[:, 0].mean())
    rows.append((name, sd3_mk, comp))
    print(f'{name}: SD3val_makespan={sd3_mk:8.2f}   SD1/SD2_composite={comp:6.2f}%')

sd3 = np.array([r[1] for r in rows])
comp = np.array([r[2] for r in rows])
rho, p = spearmanr(sd3, comp)

print('\n' + '=' * 60)
print('Spearman rank corr (SD3-val makespan vs SD1/SD2 composite):')
print(f'  rho = {rho:.3f}   p = {p:.3f}')
print('  high positive rho  -> SD3-val PREDICTS composite (selection OK; '
      'instability is real training noise, hypothesis B)')
print('  low/zero/neg rho   -> SD3-val does NOT predict composite '
      '(selection/transfer MISMATCH, hypothesis A)')
# Spotlight the worst-composite seeds: do they also have bad SD3-val?
order_comp = sorted(rows, key=lambda r: r[2])
order_sd3 = sorted(rows, key=lambda r: r[1])
print(f"\nworst composite seeds: {[r[0] for r in order_comp[-3:]]}")
print(f"worst SD3-val seeds:   {[r[0] for r in order_sd3[-3:]]}")
print("  (overlap small -> mismatch; overlap large -> same bad seeds)")
