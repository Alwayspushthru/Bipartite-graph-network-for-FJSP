import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model.sub_layers import MLP, Actor, Critic

class BiGraphLayer(nn.Module):
    def __init__(self, d):
        super(BiGraphLayer, self).__init__()
        # O <- M
        self.Wq_o = nn.Linear(d, d, bias=False)
        self.Wk_m = nn.Linear(d, d, bias=False)
        self.Wv_m = nn.Linear(d, d, bias=False)
        self.wa = nn.Linear(d, 1, bias=False)
        self.Wg = nn.Linear(d, d, bias=True)
        self.ln_j = nn.LayerNorm(d)

        # M <- O
        self.Wq_m = nn.Linear(d, d, bias=False)
        self.Wk_o = nn.Linear(d, d, bias=False)
        self.Wv_o = nn.Linear(d, d, bias=False)
        self.wa_mo = nn.Linear(d, 1, bias=False)
        self.Wg_mo = nn.Linear(d, d, bias=True)
        self.ln_m = nn.LayerNorm(d)

        # P <- (O, M, P)
        self.Wp = nn.Linear(3 * d, d, bias=True)
        self.ln_p = nn.LayerNorm(d)

    def forward(self, h_j, h_m, h_pair, dynamic_pair_mask):
        B, J, d = h_j.shape
        M = h_m.size(1)

        # ===================== O <- M (aggregate only) =====================
        q = self.Wq_o(h_j).unsqueeze(2)  # [B,J,1,d]
        k = self.Wk_m(h_m).unsqueeze(1)  # [B,1,M,d]
        v = self.Wv_m(h_m).unsqueeze(1)  # [B,1,M,d]

        score = (q * k).sum(-1) / math.sqrt(d)  # [B,J,M]
        score = score + self.wa(h_pair).squeeze(-1)
        score = score.masked_fill(dynamic_pair_mask, -1e9)
        alpha = F.softmax(score, dim=2)  # over M
        gate = 1.0 + torch.tanh(self.Wg(h_pair))  # [B,J,M,d]
        agg_j = (alpha.unsqueeze(-1) * v * gate).sum(dim=2)  # [B,J,d]

        # ===================== M <- O (aggregate only, uses original h_j) =====================
        q2 = self.Wq_m(h_m).unsqueeze(1)  # [B,1,M,d]
        k2 = self.Wk_o(h_j).unsqueeze(2)  # [B,J,1,d]  ← original h_j
        v2 = self.Wv_o(h_j).unsqueeze(2)  # [B,J,1,d]  ← original h_j

        score2 = (q2 * k2).sum(-1) / math.sqrt(d)  # [B,J,M]
        score2 = score2 + self.wa_mo(h_pair).squeeze(-1)
        score2 = score2.masked_fill(dynamic_pair_mask, -1e9)

        all_invalid_m = dynamic_pair_mask.all(dim=1, keepdim=True)  # [B,1,M]
        score2 = score2.masked_fill(all_invalid_m, 0.0)

        alpha2 = F.softmax(score2, dim=1)  # over J
        alpha2 = alpha2.masked_fill(all_invalid_m, 0.0)

        gate2 = 1.0 + torch.tanh(self.Wg_mo(h_pair))  # [B,J,M,d]
        agg_m = (alpha2.unsqueeze(-1) * v2 * gate2).sum(dim=1)  # [B,M,d]

        # ===================== Apply both updates in parallel =====================
        h_j = self.ln_j(h_j + agg_j)
        h_m = self.ln_m(h_m + agg_m)

        # ===================== P <- (O, M, P) =====================
        h_j_pair = h_j.unsqueeze(2).expand(-1, -1, M, -1)
        h_m_pair = h_m.unsqueeze(1).expand(-1, J, -1, -1)
        pair_input = torch.cat([h_j_pair, h_m_pair, h_pair], dim=-1)
        pair_delta = torch.tanh(self.Wp(pair_input))
        h_pair = self.ln_p(h_pair + pair_delta)

        return h_j, h_m, h_pair


class BiGraphNetwork(nn.Module):
    def __init__(self, config):
        super(BiGraphNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim  # 7
        self.fea_m_input_dim = config.fea_m_input_dim  # 5
        self.fea_pairs_input_dim = config.fea_pair_input_dim  # 6

        self.fea_embed_dim = 128
        self.mes_dim = 128

        self.num_BiG_layers = config.num_bigraph_layers
        self.BiG_layers = nn.ModuleList(
            [BiGraphLayer(d=self.mes_dim) for _ in range(self.num_BiG_layers)]
        )

        self.job_mlp = MLP(2, self.fea_j_input_dim, self.fea_embed_dim, self.mes_dim)
        self.mach_mlp = MLP(2, self.fea_m_input_dim, self.fea_embed_dim, self.mes_dim)
        self.pair_mlp = MLP(2, self.fea_pairs_input_dim, self.fea_embed_dim, self.mes_dim)

        # Actor path: high-resolution local features for per-(job,machine) discrimination
        self.actor_dim = 32
        self.actor_j_proj    = nn.Linear(self.mes_dim, self.actor_dim)
        self.actor_m_proj    = nn.Linear(self.mes_dim, self.actor_dim)
        self.actor_pair_proj = nn.Linear(self.mes_dim, self.actor_dim)

        # Global path: compact distributional features for Critic and GRU history
        self.global_dim = 16
        self.global_j_proj    = nn.Linear(self.mes_dim, self.global_dim)
        self.global_m_proj    = nn.Linear(self.mes_dim, self.global_dim)
        self.global_pair_proj = nn.Linear(self.mes_dim, self.global_dim)

        # GRU input: 3 global means (each global_dim) → [B, 3*global_dim]
        self.hist_dim = 64
        self.gru_cell = nn.GRUCell(3 * self.global_dim, self.hist_dim)

        # Actor: local_j + local_m + global_j + global_m + local_pair + h_hist
        actor_input_dim = 3 * self.actor_dim + 2 * self.global_dim + self.hist_dim
        self.actor = Actor(config.num_mlp_layers_actor, actor_input_dim,
            config.hidden_dim_actor, 1)
        # Critic: 3 global means + h_hist
        critic_input_dim = 3 * self.global_dim + self.hist_dim
        self.critic = Critic(config.num_mlp_layers_critic, critic_input_dim,
            config.hidden_dim_critic, 1)

    def forward(self, fea_j, fea_m, fea_pairs, dynamic_pair_mask, h_hist=None):
        """Single-step forward pass. Used during rollout and inference."""
        B, J, M = dynamic_pair_mask.shape

        h_j = self.job_mlp(fea_j)
        h_m = self.mach_mlp(fea_m)
        h_pair = self.pair_mlp(fea_pairs)

        for layer in self.BiG_layers:
            h_j, h_m, h_pair = layer(h_j, h_m, h_pair, dynamic_pair_mask)

        # Actor projections — local, high-resolution
        a_j    = self.actor_j_proj(h_j)       # [B, J, actor_dim]
        a_m    = self.actor_m_proj(h_m)        # [B, M, actor_dim]
        a_pair = self.actor_pair_proj(h_pair)  # [B, J, M, actor_dim]

        # Global projections — compact, suited for mean pooling
        g_j    = self.global_j_proj(h_j)       # [B, J, global_dim]
        g_m    = self.global_m_proj(h_m)        # [B, M, global_dim]
        g_pair = self.global_pair_proj(h_pair)  # [B, J, M, global_dim]

        active_job_mask  = ~dynamic_pair_mask.all(dim=-1)  # [B, J]
        active_mach_mask = ~dynamic_pair_mask.all(dim=1)   # [B, M]
        g_j_global = self.nonzero_averaging(g_j, active_job_mask)   # [B, global_dim]
        g_m_global = self.nonzero_averaging(g_m, active_mach_mask)  # [B, global_dim]

        valid_pair_mask = ~dynamic_pair_mask
        g_pair_global = (g_pair * valid_pair_mask.unsqueeze(-1)).sum(dim=(1, 2)) \
                        / valid_pair_mask.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1)  # [B, global_dim]

        h_graph = torch.cat([g_j_global, g_m_global, g_pair_global], dim=-1)  # [B, 3*global_dim]

        if h_hist is None:
            h_hist = torch.zeros(B, self.hist_dim, device=fea_j.device)
        h_hist_new = self.gru_cell(h_graph, h_hist)

        # Broadcast for per-(job,machine) actor input
        a_j_exp        = a_j.unsqueeze(2).expand(-1, -1, M, -1)
        a_m_exp        = a_m.unsqueeze(1).expand(-1, J, -1, -1)
        g_j_global_exp = g_j_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)
        g_m_global_exp = g_m_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)
        h_hist_exp     = h_hist_new.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)

        candidate_feature = torch.cat([
            a_j_exp.reshape(B, J * M, self.actor_dim),
            a_m_exp.reshape(B, J * M, self.actor_dim),
            g_j_global_exp.reshape(B, J * M, self.global_dim),
            g_m_global_exp.reshape(B, J * M, self.global_dim),
            a_pair.reshape(B, J * M, self.actor_dim),
            h_hist_exp.reshape(B, J * M, self.hist_dim),
        ], dim=-1)  # [B, J*M, 3*actor_dim + 2*global_dim + hist_dim]

        logits = self.actor(candidate_feature).squeeze(-1)
        logits[dynamic_pair_mask.reshape(B, -1)] = float('-inf')
        pi = F.softmax(logits, dim=1)

        value = self.critic(torch.cat([h_graph, h_hist_new], dim=-1)).squeeze(-1)

        return pi, value, h_hist_new

    def forward_sequence(self, fea_j_seq, fea_m_seq, fea_pairs_seq, mask_seq, h0, done_seq=None):
        """
        Full-sequence forward with BPTT through the GRU.
        Used in PPO update to recompute pi/value under current policy parameters.

        Args:
            fea_j_seq:     [T, B, J, Fj]
            fea_m_seq:     [T, B, M, Fm]
            fea_pairs_seq: [T, B, J, M, Fp]
            mask_seq:      [T, B, J, M]  dynamic_pair_mask
            h0:            [B, hist_dim]  initial hidden (zeros at episode start)
            done_seq:      [T, B] bool, done flag after each step

        Returns:
            pi_seq:    [T, B, J*M]
            value_seq: [T, B]
            h_last:    [B, hist_dim]
        """
        T = fea_j_seq.size(0)
        h = h0
        pi_list, value_list = [], []

        for t in range(T):
            pi_t, value_t, h = self.forward(
                fea_j_seq[t], fea_m_seq[t], fea_pairs_seq[t], mask_seq[t], h
            )
            pi_list.append(pi_t)
            value_list.append(value_t)

            # Reset hidden for envs that finished this step
            if done_seq is not None:
                h = h * (~done_seq[t]).float().unsqueeze(-1)

        return torch.stack(pi_list, dim=0), torch.stack(value_list, dim=0), h

    def act(self, fea_j, fea_m, fea_pairs, candidate, dynamic_pair_mask, h_hist=None):
        pi, value, h_hist_new = self.forward(fea_j, fea_m, fea_pairs, dynamic_pair_mask, h_hist)
        dist = Categorical(pi)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, h_hist_new

    def nonzero_averaging(self, x, mask):
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1)
        return (x * mask_f).sum(dim=1) / count
