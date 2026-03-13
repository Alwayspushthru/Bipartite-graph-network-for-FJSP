import torch
import numpy as np
import random
from torch.distributions.categorical import Categorical
from env.fjsp_env import EnvState
from typing import Dict

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str

def to_numpy(array) -> np.ndarray:
    """Convert tensors/arrays to NumPy without importing torch here."""

    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach") and hasattr(array, "cpu"):
        return array.detach().cpu().numpy()
    return np.asarray(array)

def clone_state_tensors(state: EnvState) -> Dict[str, torch.Tensor]:
    """Create a CPU clone of the tensors required to rebuild ``EnvState``."""

    tensor_fields = [
        "fea_j_tensor", "fea_m_tensor", "fea_pairs_tensor",
        "candidate_tensor", "job_mask_tensor", "dynamic_pair_mask_tensor"
    ]
    snapshot: Dict[str, torch.Tensor] = {}
    for field in tensor_fields:
        value = getattr(state, field, None)
        if value is None:
            continue
        snapshot[field] = value.detach().cpu().clone()
    return snapshot
