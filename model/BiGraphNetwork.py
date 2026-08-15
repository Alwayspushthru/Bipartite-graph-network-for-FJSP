import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model.sub_layers import MLP, Actor, Critic


class BiGraphAttention(nn.Module):
    """
    Multi-head cross-attention adapted for FJSP bipartite graph.
    Restores the two FJSP-specific mechanisms from the original single-head design:
      1. pair-wise attention bias  — wa(h_pair) projected to H scalars per pair/head
      2. pair-wise gate            — Wg applied to attention-weighted pair features
    """
    def __init__(self, d, num_heads=2, ablation='none'):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.ablation  = ablation

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

        # pair-wise attention bias: d → num_heads scalars per (query, key) pair
        self.wa = nn.Linear(d, num_heads, bias=False)
        # pair-wise gate using attention-weighted pair features
        self.Wg = nn.Linear(d, d, bias=True)

    def forward(self, query_feat, key_feat, h_pair, mask):
        """
        query_feat: [B, Nq, d]
        key_feat:   [B, Nk, d]
        h_pair:     [B, Nq, Nk, d]  — query-indexed pair features
        mask:       [B, Nq, Nk]     — True = invalid pair
        """
        B, Nq, d = query_feat.shape
        Nk = key_feat.size(1)
        H, Dh = self.num_heads, self.head_dim

        # Project and split into heads: [B, H, N, Dh]
        q = self.Wq(query_feat).reshape(B, Nq, H, Dh).transpose(1, 2)
        k = self.Wk(key_feat).reshape(B, Nk, H, Dh).transpose(1, 2)
        v = self.Wv(key_feat).reshape(B, Nk, H, Dh).transpose(1, 2)

        if self.ablation == 'mean_agg':
            # Ablation: replace learned attention with uniform mean over valid
            # neighbors (drops Q.K scores and pair-bias entirely).
            valid = (~mask).float().unsqueeze(1)                  # [B, 1, Nq, Nk]
            alpha = valid / valid.sum(-1, keepdim=True).clamp_min(1.0)
            alpha = alpha.expand(B, H, Nq, Nk)                    # shared across heads
        else:
            # Scaled dot-product scores: [B, H, Nq, Nk]
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            # Pair-wise attention bias: [B, Nq, Nk, H] -> [B, H, Nq, Nk]
            if self.ablation != 'no_pair_bias':
                pair_bias = self.wa(h_pair).permute(0, 3, 1, 2)
                scores = scores + pair_bias

            # Mask invalid pairs then softmax
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
            alpha = F.softmax(scores, dim=-1)  # [B, H, Nq, Nk]

        # Aggregate values: [B, H, Nq, Dh] -> [B, Nq, d]
        agg = torch.matmul(alpha, v).transpose(1, 2).reshape(B, Nq, d)
        agg = self.Wo(agg)

        # Pair-wise gate: weight h_pair by mean attention across heads
        alpha_mean = alpha.mean(dim=1)                              # [B, Nq, Nk]
        h_pair_ctx = (alpha_mean.unsqueeze(-1) * h_pair).sum(dim=2) # [B, Nq, d]
        gate = 1.0 + torch.tanh(self.Wg(h_pair_ctx))               # [B, Nq, d]

        return agg * gate


class BiGraphLayer(nn.Module):
    def __init__(self, d, num_heads=2, ablation='none'):
        super(BiGraphLayer, self).__init__()
        self.attn_om = BiGraphAttention(d, num_heads, ablation)  # O <- M
        self.ln_j    = nn.LayerNorm(d)

        self.attn_mo = BiGraphAttention(d, num_heads, ablation)  # M <- O
        self.ln_m    = nn.LayerNorm(d)

        self.Wp   = nn.Linear(3 * d, d, bias=True)    # P <- (O, M, P)
        self.ln_p = nn.LayerNorm(d)

    def forward(self, h_j, h_m, h_pair, dynamic_pair_mask):
        _, J, _ = h_j.shape
        M = h_m.size(1)

        # ===================== O <- M =====================
        agg_j = self.attn_om(h_j, h_m, h_pair, dynamic_pair_mask)

        # ===================== M <- O =====================
        # h_pair transposed so machines are the query dimension
        agg_m = self.attn_mo(h_m, h_j,
                              h_pair.transpose(1, 2),
                              dynamic_pair_mask.transpose(1, 2))

        # ===================== Apply both updates in parallel =====================
        h_j = self.ln_j(h_j + agg_j)
        h_m = self.ln_m(h_m + agg_m)

        # ===================== P <- (O, M, P) =====================
        h_j_pair = h_j.unsqueeze(2).expand(-1, -1, M, -1)
        h_m_pair = h_m.unsqueeze(1).expand(-1, J, -1, -1)
        pair_input = torch.cat([h_j_pair, h_m_pair, h_pair], dim=-1)
        h_pair = self.ln_p(h_pair + torch.tanh(self.Wp(pair_input)))

        return h_j, h_m, h_pair


class BiGraphNetwork(nn.Module):
    def __init__(self, config):
        super(BiGraphNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim  # 7
        self.fea_m_input_dim = config.fea_m_input_dim  # 5
        self.revision_variant = getattr(config, 'revision_variant', 'b0')
        self.fea_pairs_input_dim = 8 if self.revision_variant == 'l' else config.fea_pair_input_dim

        self.fea_embed_dim = 128
        self.mes_dim = 128

        self.ablation = getattr(config, 'ablation', 'none')

        self.num_BiG_layers = config.num_bigraph_layers
        self.BiG_layers = nn.ModuleList(
            [BiGraphLayer(d=self.mes_dim, ablation=self.ablation)
             for _ in range(self.num_BiG_layers)]
        )

        self.job_mlp = MLP(2, self.fea_j_input_dim, self.fea_embed_dim, self.mes_dim)
        self.mach_mlp = MLP(2, self.fea_m_input_dim, self.fea_embed_dim, self.mes_dim)
        self.pair_mlp = MLP(2, self.fea_pairs_input_dim, self.fea_embed_dim, self.mes_dim)
        if self.revision_variant == 'q':
            self.waiting_mlp = MLP(2, 4, self.fea_embed_dim, self.mes_dim)
            self.waiting_gate = nn.Linear(2 * self.mes_dim, self.mes_dim)
            self.waiting_ln = nn.LayerNorm(self.mes_dim)

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
        if self.revision_variant == 'g':
            self.urgency_gate = nn.Sequential(
                nn.Linear(3 * self.global_dim, self.hist_dim),
                nn.GELU(),
                nn.Linear(self.hist_dim, self.hist_dim),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.urgency_gate[2].bias, 2.0)

        # Actor: local_j + local_m + global_j + global_m + local_pair + h_hist
        actor_input_dim = 3 * self.actor_dim + 2 * self.global_dim + self.hist_dim
        self.actor = Actor(config.num_mlp_layers_actor, actor_input_dim,
            config.hidden_dim_actor, 1)
        # Critic: 3 global means + h_hist
        critic_input_dim = 3 * self.global_dim + self.hist_dim
        self.critic = Critic(config.num_mlp_layers_critic, critic_input_dim,
            config.hidden_dim_critic, 1)

    def _embed_jobs(self, fea_j, fea_waiting):
        h_j = self.job_mlp(fea_j)
        if self.revision_variant == 'q':
            h_wait = self.waiting_mlp(fea_waiting)
            gate = torch.sigmoid(self.waiting_gate(torch.cat([h_j, h_wait], dim=-1)))
            h_j = self.waiting_ln(h_j + gate * h_wait)
        return h_j

    def _update_history(self, h_graph, h_hist):
        h_candidate = self.gru_cell(h_graph, h_hist)
        if self.revision_variant == 'g':
            gate = self.urgency_gate(h_graph)
            return h_hist + gate * (h_candidate - h_hist)
        return h_candidate

    def encode_no_recurrence(self, fea_j, fea_m, fea_pairs, fea_waiting, dynamic_pair_mask):
        h_j = self._embed_jobs(fea_j, fea_waiting)
        h_m = self.mach_mlp(fea_m)
        h_pair = self.pair_mlp(fea_pairs)

        for layer in self.BiG_layers:
            h_j, h_m, h_pair = layer(h_j, h_m, h_pair, dynamic_pair_mask)

        a_j = self.actor_j_proj(h_j)
        a_m = self.actor_m_proj(h_m)
        a_pair = self.actor_pair_proj(h_pair)

        g_j = self.global_j_proj(h_j)
        g_m = self.global_m_proj(h_m)
        g_pair = self.global_pair_proj(h_pair)

        active_job_mask = ~dynamic_pair_mask.all(dim=-1)
        active_mach_mask = ~dynamic_pair_mask.all(dim=1)
        g_j_global = self.nonzero_averaging(g_j, active_job_mask)
        g_m_global = self.nonzero_averaging(g_m, active_mach_mask)

        valid_pair_mask = ~dynamic_pair_mask
        g_pair_global = (g_pair * valid_pair_mask.unsqueeze(-1)).sum(dim=(1, 2)) \
                        / valid_pair_mask.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1)

        h_graph = torch.cat([g_j_global, g_m_global, g_pair_global], dim=-1)
        return a_j, a_m, a_pair, g_j_global, g_m_global, h_graph

    def decode_from_encoded(self, a_j, a_m, a_pair, g_j_global, g_m_global,
                            h_graph, h_hist, dynamic_pair_mask):
        N, J, M = dynamic_pair_mask.shape

        a_j_exp = a_j.unsqueeze(2).expand(-1, -1, M, -1)
        a_m_exp = a_m.unsqueeze(1).expand(-1, J, -1, -1)
        g_j_global_exp = g_j_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)
        g_m_global_exp = g_m_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)
        h_hist_exp = h_hist.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)

        candidate_feature = torch.cat([
            a_j_exp.reshape(N, J * M, self.actor_dim),
            a_m_exp.reshape(N, J * M, self.actor_dim),
            g_j_global_exp.reshape(N, J * M, self.global_dim),
            g_m_global_exp.reshape(N, J * M, self.global_dim),
            a_pair.reshape(N, J * M, self.actor_dim),
            h_hist_exp.reshape(N, J * M, self.hist_dim),
        ], dim=-1)

        logits = self.actor(candidate_feature).squeeze(-1)
        logits = logits.masked_fill(dynamic_pair_mask.reshape(N, -1), float('-inf'))
        pi = F.softmax(logits, dim=1)

        value = self.critic(torch.cat([h_graph, h_hist], dim=-1)).squeeze(-1)
        return pi, value

    def forward(self, fea_j, fea_m, fea_pairs, fea_waiting, dynamic_pair_mask, h_hist=None):
        """Single-step forward pass. Used during rollout and inference."""
        B, J, M = dynamic_pair_mask.shape

        h_j = self._embed_jobs(fea_j, fea_waiting)
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
        if self.ablation == 'no_gru':
            # Ablation: Markovian policy, no cross-step history.
            h_hist_new = torch.zeros(B, self.hist_dim, device=fea_j.device)
        else:
            h_hist_new = self._update_history(h_graph, h_hist)

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
        logits = logits.masked_fill(dynamic_pair_mask.reshape(B, -1), float('-inf'))
        pi = F.softmax(logits, dim=1)

        value = self.critic(torch.cat([h_graph, h_hist_new], dim=-1)).squeeze(-1)

        return pi, value, h_hist_new

    def forward_sequence(self, fea_j_seq, fea_m_seq, fea_pairs_seq, fea_waiting_seq, mask_seq, h0, done_seq=None):
        """
        Full-sequence forward with BPTT through the GRU.
        Used in PPO update to recompute pi/value under current policy parameters.

        Args:
            fea_j_seq:     [T, B, J, Fj]
            fea_m_seq:     [T, B, M, Fm]
            fea_pairs_seq: [T, B, J, M, Fp]
            fea_waiting_seq: [T, B, J, 4]
            mask_seq:      [T, B, J, M]  dynamic_pair_mask
            h0:            [B, hist_dim]  initial hidden (zeros at episode start)
            done_seq:      [T, B] bool, done flag after each step

        Returns:
            pi_seq:    [T, B, J*M]
            value_seq: [T, B]
            h_last:    [B, hist_dim]
        """
        T, B, J, M = mask_seq.shape

        def _flat(x):
            return x.reshape(T * B, *x.shape[2:])

        a_j, a_m, a_pair, g_j_global, g_m_global, h_graph = self.encode_no_recurrence(
            _flat(fea_j_seq), _flat(fea_m_seq), _flat(fea_pairs_seq),
            _flat(fea_waiting_seq), _flat(mask_seq)
        )
        h_graph_seq = h_graph.reshape(T, B, -1)

        if self.ablation == 'no_gru':
            # Ablation: Markovian policy, no cross-step history.
            h_hist_seq = torch.zeros(T, B, self.hist_dim, device=h_graph.device)
            h_last = torch.zeros(B, self.hist_dim, device=h_graph.device)
        else:
            h = h0
            h_hist_list = []
            for t in range(T):
                h = self._update_history(h_graph_seq[t], h)
                h_hist_list.append(h)
                if done_seq is not None:
                    h = h * (~done_seq[t]).float().unsqueeze(-1)

            h_hist_seq = torch.stack(h_hist_list, dim=0)
            h_last = h

        pi, value = self.decode_from_encoded(
            a_j, a_m, a_pair, g_j_global, g_m_global,
            h_graph, _flat(h_hist_seq), _flat(mask_seq)
        )

        return pi.reshape(T, B, J * M), value.reshape(T, B), h_last

    def act(self, fea_j, fea_m, fea_pairs, fea_waiting, candidate, dynamic_pair_mask, h_hist=None):
        pi, value, h_hist_new = self.forward(
            fea_j, fea_m, fea_pairs, fea_waiting, dynamic_pair_mask, h_hist
        )
        dist = Categorical(pi)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, h_hist_new

    def nonzero_averaging(self, x, mask):
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1)
        return (x * mask_f).sum(dim=1) / count
