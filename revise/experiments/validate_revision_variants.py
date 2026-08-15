from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from env.FJSPEnv import FJSPEnv
from model.BiGraphNetwork import BiGraphNetwork
from utils.data_utils import load_data_from_files


def config(variant):
    return SimpleNamespace(
        fea_j_input_dim=8,
        fea_m_input_dim=5,
        fea_pair_input_dim=6,
        num_bigraph_layers=2,
        ablation="none",
        revision_variant=variant,
        num_mlp_layers_actor=3,
        hidden_dim_actor=64,
        num_mlp_layers_critic=3,
        hidden_dim_critic=64,
    )


def first_legal_actions(state):
    valid = ~state.dynamic_pair_mask_tensor.cpu().numpy()
    return valid.reshape(valid.shape[0], -1).argmax(axis=1)


def main():
    torch.manual_seed(7)
    device = torch.device("cpu")
    jobs, pts = load_data_from_files("data/data_train_vali/SD3/10x5")
    jobs, pts = jobs[:2], pts[:2]

    parameter_counts = {}
    for variant in ("b0", "q", "l", "g", "r"):
        env = FJSPEnv(device, variant, 0.1)
        state = env.set_initial_data(jobs, pts)
        expected_pair_dim = 8 if variant == "l" else 6
        assert state.fea_pairs_tensor.shape[-1] == expected_pair_dim
        assert state.fea_waiting_tensor.shape == (2, 10, 4)
        assert torch.isfinite(state.fea_waiting_tensor).all()

        model = BiGraphNetwork(config(variant))
        parameter_counts[variant] = sum(p.numel() for p in model.parameters())
        h0 = torch.zeros(2, model.hist_dim)
        with torch.no_grad():
            pi, value, h1 = model(
                state.fea_j_tensor,
                state.fea_m_tensor,
                state.fea_pairs_tensor,
                state.fea_waiting_tensor,
                state.dynamic_pair_mask_tensor,
                h0,
            )
            pi_seq, value_seq, _ = model.forward_sequence(
                state.fea_j_tensor.unsqueeze(0),
                state.fea_m_tensor.unsqueeze(0),
                state.fea_pairs_tensor.unsqueeze(0),
                state.fea_waiting_tensor.unsqueeze(0),
                state.dynamic_pair_mask_tensor.unsqueeze(0),
                h0,
                torch.zeros(1, 2, dtype=torch.bool),
            )
        assert torch.isfinite(pi).all() and torch.isfinite(value).all() and torch.isfinite(h1).all()
        assert torch.allclose(pi, pi_seq[0], atol=1e-6)
        assert torch.allclose(value, value_seq[0], atol=1e-6)

        if variant == "l":
            assert torch.count_nonzero(state.fea_pairs_tensor[..., 6:]) == 0
            state, _, _ = env.step(first_legal_actions(state))
            assert torch.count_nonzero(state.fea_pairs_tensor[..., 6:]) > 0

    b0_model = BiGraphNetwork(config("b0"))
    b0_model.load_state_dict(
        torch.load("trained_network/revise_b0_s300.pth", map_location="cpu", weights_only=True),
        strict=True,
    )

    base_env = FJSPEnv(device, "b0", 0.1)
    shaped_env = FJSPEnv(device, "r", 0.1)
    base_state = base_env.set_initial_data(jobs[:1], pts[:1])
    shaped_state = shaped_env.set_initial_data(jobs[:1], pts[:1])
    base_return = 0.0
    shaped_return = 0.0
    done = np.array([False])
    while not done.all():
        actions = first_legal_actions(base_state)
        base_state, base_reward, done = base_env.step(actions)
        shaped_state, shaped_reward, shaped_done = shaped_env.step(actions)
        assert np.array_equal(done, shaped_done)
        base_return += float(base_reward[0])
        shaped_return += float(shaped_reward[0])
    assert abs(base_return - shaped_return) < 1e-7
    assert abs(float(shaped_env.shaping_potential[0])) < 1e-12
    assert np.array_equal(base_env.current_makespan, shaped_env.current_makespan)

    print("variant validation passed")
    print("parameter_counts", parameter_counts)
    print("reward_return_difference", shaped_return - base_return)
    print("checkpoint_compatibility", Path("trained_network/revise_b0_s300.pth").is_file())


if __name__ == "__main__":
    main()
