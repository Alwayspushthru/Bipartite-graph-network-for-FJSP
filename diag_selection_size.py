"""
Diagnostic round 2: which SD3 validation SIZE best predicts SD1/SD2 transfer?

diag_selection_mismatch.py showed SD3 10x5-val makespan is uncorrelated
(rho=-0.19) with the SD1/SD2 composite. Hypothesis: validating on a LARGER SD3
size (closer to the 30x10/40x10 test sizes) tracks transfer better, so switching
train.py's selection criterion to a larger size would keep better-transferring
checkpoints and shrink the cross-seed noise.

Evaluates each default seed checkpoint on SD3 validation sets of several sizes
and reports the rank correlation of each vs the known SD1/SD2 composite. No
training, no data leakage (test set stays SD1/SD2).
"""
import numpy as np
from scipy.stats import spearmanr
from params import configs
from test import test_greedy_strategy
from utils.data_utils import load_data_from_files

COMPOSITE = {
    'nf_seed300': 8.546, 'nf_seed301': 6.596, 'nf_seed302': 7.468,
    'nf_seed303': 10.692, 'nf_seed304': 15.151, 'nf_seed305': 7.520,
    'nf_seed306': 10.764, 'nf_seed307': 7.042, 'nf_seed308': 10.554,
    'nf_seed309': 8.118,
}
SIZES = ['10x5', '15x10', '20x10', '20x5']
seeds = list(COMPOSITE.keys())
comp = np.array([COMPOSITE[s] for s in seeds])

print(f"{'size':8s} " + " ".join(f"{s[-3:]:>6s}" for s in seeds) + "   rho_vs_composite")
for size in SIZES:
    try:
        vali = load_data_from_files(f'./data/data_train_vali/SD3/{size}')
    except Exception as e:
        print(f"{size:8s} (load failed: {e})"); continue
    mks = []
    for s in seeds:
        res = test_greedy_strategy(vali, f'./trained_network/{s}.pth',
                                   configs.seed_test, 0)
        mks.append(float(res[:, 0].mean()))
    mks = np.array(mks)
    rho, p = spearmanr(mks, comp)
    print(f"{size:8s} " + " ".join(f"{m:6.1f}" for m in mks) +
          f"   rho={rho:+.3f} (p={p:.3f})")

print("\nInterpretation: the size with the most POSITIVE rho is the best "
      "selection criterion (its val makespan ranks seeds like the composite "
      "does). If a larger size beats 10x5's rho=-0.19, switch train.py "
      "validation to that size.")
