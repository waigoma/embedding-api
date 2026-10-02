import {test, afterEach} from 'node:test';
import assert from 'node:assert/strict';
import {request, snapshot} from '../src/webui/api.js';
const original = globalThis.fetch;
afterEach(() => { globalThis.fetch = original; });

test('API requests preserve POST contract, credentials and relative prefix resolution', async () => {
  let observed;
  globalThis.fetch = async (url, init) => { observed = {url, init}; return new Response('{"status":"loaded"}'); };
  assert.deepEqual(await request('v1/models/load', {body: {model_id: 'synthetic', model_type: 'embedding'}}), {status: 'loaded'});
  assert.ok(observed.url.pathname.endsWith('/embedding-api/v1/models/load'));
  assert.equal(observed.init.credentials, 'same-origin');
  assert.equal(observed.init.method, 'POST');
  assert.equal(JSON.parse(observed.init.body).model_id, 'synthetic');
});

test('API errors retain server detail without rendering HTML', async () => {
  globalThis.fetch = async () => new Response('{"detail":"synthetic conflict"}', {status: 409});
  await assert.rejects(request('v1/models/download', {body: {}}), /409.*synthetic conflict/);
  globalThis.fetch = async () => new Response('{"detail":[{"msg":"field required"}]}', {status: 422});
  await assert.rejects(request('v1/embeddings', {body: {}}), /field required/);
  globalThis.fetch = async () => new Response('<script>bad</script>', {status: 502});
  await assert.rejects(request('health'), /HTTP 502/);
  globalThis.fetch = async () => new Response('not json');
  await assert.rejects(request('health'), /JSON/);
});

test('partial read failures are explicit while successful sections remain inspectable', async () => {
  globalThis.fetch = async url => url.pathname.endsWith('/health')
    ? new Response('{"detail":"synthetic outage"}', {status: 503}) : new Response('{"data":[]}');
  const result = await snapshot();
  assert.equal(result.failures.length, 1);
  assert.match(result.failures[0], /health.*503/);
  assert.deepEqual(result.values.models.data, []);
  assert.equal(result.values.health, undefined);
});

test('aborted reads are ignored instead of producing a stale snapshot', async () => {
  const controller = new AbortController(); controller.abort();
  globalThis.fetch = async (_url, init) => { if (init.signal.aborted) throw new DOMException('aborted', 'AbortError'); };
  await assert.rejects(snapshot(controller.signal), {name: 'AbortError'});
});
