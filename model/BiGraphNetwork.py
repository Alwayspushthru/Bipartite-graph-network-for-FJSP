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

    def forward(self, h_j, h_m, h_pair, dynamic_pair_mask):
        B, J, d = h_j.shape

        # ===================== O <- M =====================
        q = self.Wq_o(h_j).unsqueeze(2)  # [B,J,1,d]
        k = self.Wk_m(h_m).unsqueeze(1)  # [B,1,M,d]
        v = self.Wv_m(h_m).unsqueeze(1)  # [B,1,M,d]

        score = (q * k).sum(-1) / math.sqrt(d)  # [B,J,M]
        score = score + self.wa(h_pair).squeeze(-1)
        score = score.masked_fill(dynamic_pair_mask, -1e9)
        alpha = F.softmax(score, dim=2)  # over M
        gate = 1.0 + torch.tanh(self.Wg(h_pair))  # [B,J,M,d]
        msg = v * gate  # [B,J,M,d]
        agg = (alpha.unsqueeze(-1) * msg).sum(dim=2)  # [B,J,d]
        h_j = self.ln_j(h_j + agg)

        # ===================== M <- O =====================
        q2 = self.Wq_m(h_m).unsqueeze(1)  # [B,1,M,d]
        k2 = self.Wk_o(h_j).unsqueeze(2)  # [B,J,1,d]
        v2 = self.Wv_o(h_j).unsqueeze(2)  # [B,J,1,d]

        score2 = (q2 * k2).sum(-1) / math.sqrt(d)  # [B,J,M]
        score2 = score2 + self.wa_mo(h_pair).squeeze(-1)
        score2 = score2.masked_fill(dynamic_pair_mask, -1e9)

        all_invalid_m = dynamic_pair_mask.all(dim=1, keepdim=True)  # [B,1,M]
        score2 = score2.masked_fill(all_invalid_m, 0.0)

        alpha2 = F.softmax(score2, dim=1)  # over J
        alpha2 = alpha2.masked_fill(all_invalid_m, 0.0)

        gate2 = 1.0 + torch.tanh(self.Wg_mo(h_pair))  # [B,J,M,d]
        msg2 = v2 * gate2
        agg_m = (alpha2.unsqueeze(-1) * msg2).sum(dim=1)  # [B,M,d]

        h_m = self.ln_m(h_m + agg_m)

        return h_j, h_m


class BiGraphNetwork(nn.Module):
    def __init__(self, config):
        super(BiGraphNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim  # 6
        self.fea_m_input_dim = config.fea_m_input_dim  # 4
        self.fea_pairs_input_dim = config.fea_pair_input_dim  # 6

        self.fea_embed_dim = 8
        self.mes_dim = 32

        self.num_BiG_layers = config.num_bigraph_layers
        self.BiG_layers = [BiGraphLayer(d = self.mes_dim) for _ in range(self.num_BiG_layers)]

        self.job_mlp = MLP(2, self.fea_j_input_dim, self.fea_embed_dim, self.mes_dim)
        self.mach_mlp = MLP(2, self.fea_m_input_dim, self.fea_embed_dim,self.mes_dim)
        self.pair_mlp = MLP(2, self.fea_pairs_input_dim, self.fea_embed_dim,self.mes_dim)

        self.linear_layer = nn.Linear(self.mes_dim, self.fea_embed_dim)

        self.actor = Actor(config.num_mlp_layers_actor, 5 * self.fea_embed_dim,
            config.hidden_dim_actor,1,)
        self.critic = Critic(config.num_mlp_layers_critic, 3 * self.fea_embed_dim,
            config.hidden_dim_critic,1,)

    def forward(self, fea_j, fea_m, fea_pairs, dynamic_pair_mask):
        B,J,M = dynamic_pair_mask.shape

        h_j = self.job_mlp(fea_j) # 6 → 8 → 32
        h_m = self.mach_mlp(fea_m) # 4 → 8 → 32
        h_pair = self.pair_mlp(fea_pairs) # 6 → 8 → 32

        for layer in self.BiG_layers:
                h_j, h_m = layer(h_j, h_m, h_pair, dynamic_pair_mask)

        _h_j = self.linear_layer(h_j)
        _h_m = self.linear_layer(h_m)

        layer = nn.Linear(6,8)
        _h_pair = layer(fea_pairs)

        h_j_global = self.nonzero_averaging(_h_j)
        h_m_global = self.nonzero_averaging(_h_m)

        h_j_pair = _h_j.unsqueeze(2).expand(-1, -1, M, -1)  # (B,J,M,8)
        h_m_pair = _h_m.unsqueeze(1).expand(-1, J, -1, -1)  # (B,J,M,8)
        h_j_global_pair = h_j_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)  # (B,J,M,8)
        h_m_global_pair = h_m_global.unsqueeze(1).unsqueeze(2).expand(-1, J, M, -1)  # (B,J,M,8)

        h_j_flat = h_j_pair.reshape(B, J * M, 8)
        h_m_flat = h_m_pair.reshape(B, J * M, 8)
        h_j_global_flat = h_j_global_pair.reshape(B, J * M, 8)
        h_m_global_flat = h_m_global_pair.reshape(B, J * M, 8)
        h_pair_flat = _h_pair.reshape(B, J * M, 8)

        candidate_feature = torch.cat([h_j_flat, h_m_flat, h_j_global_flat,h_m_global_flat, h_pair_flat], dim=-1)
        logits = self.actor(candidate_feature).squeeze(-1)

        logits[dynamic_pair_mask.reshape(B, -1)] = float('-inf')
        pi = F.softmax(logits, dim=1)

        mask = ~dynamic_pair_mask
        mask = mask.unsqueeze(-1)
        h_pair_sum = (_h_pair * mask).sum(dim=(1,2))
        num_valid = mask.sum(dim=(1, 2)).clamp_min(1)  # (B,1)
        h_pair_global = h_pair_sum / num_valid  # (B,8)

        h_graph = torch.cat([h_j_global,h_m_global,h_pair_global], dim=-1)
        value = self.critic(h_graph).squeeze(-1)

        return pi, value

    def act(self, fea_j, fea_m, fea_pairs, candidate, dynamic_pair_mask):
        pi, value = self.forward(fea_j, fea_m, fea_pairs, dynamic_pair_mask)
        dist = Categorical(pi)

        action = dist.sample() # 采样动作
        log_prob = dist.log_prob(action) #计算对数概率

        return action, log_prob, value

    def evaluate_actions(self, fea_j, fea_m, fea_pairs, candidate, dynamic_pair_mask, action):
        pi, value = self.forward(fea_j, fea_m, fea_pairs, dynamic_pair_mask)

        dist = Categorical(pi)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def nonzero_averaging(self, x):
        b = x.sum(dim=-2)
        y = torch.count_nonzero(x, dim=-1)
        z = (y != 0).sum(dim=-1, keepdim=True)
        p = 1 / z
        p[z == 0] = 0
        return torch.mul(p, b)
