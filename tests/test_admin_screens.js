// Pure helpers of the embedding-api service screens and the screen registry (node --test).
import {test, afterEach} from 'node:test';
import assert from 'node:assert/strict';
import screens, {feedLoaders, tabAliases, fallbackTab} from '../src/static/admin/screens/index.js';
import {catalogLocalName, catalogState, jobSignature} from '../src/static/admin/screens/catalog.js';
import {testPayload, cosineSimilarity, loadedEmbeddingModels} from '../src/static/admin/screens/playground.js';
import {v1Client} from '../src/static/admin/service/v1.js';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

test('registry order, sections and the health feed loader', async () => {
  assert.deepEqual(screens.map(s => s.id), ['models', 'catalog', 'playground', 'recent', 'health']);
  assert.equal(screens.find(s => s.id === 'recent').label, '📊 Inference Logs');
  assert.equal(screens.find(s => s.id === 'catalog').label, '☆ カタログ');
  assert.deepEqual(screens.map(s => s.section), ['Models', 'Models', 'Playground', 'Observability', 'Server']);
  const calls = [];
  await feedLoaders.health(path => { calls.push(path); return Promise.resolve({}); });
  assert.deepEqual(calls, ['health']);
  assert.equal(fallbackTab, 'models');
  for (const target of Object.values(tabAliases)) assert.ok(screens.some(s => s.id === target), target);
});

test('catalog local names follow the previous UI policy', () => {
  assert.equal(catalogLocalName({repo_id: 'Qwen/Qwen3-Embedding-0.6B'}), 'Qwen3-Embedding-0.6B');
  assert.equal(catalogLocalName({repo_id: 'cl-nagoya/ruri-v3-310m', local_name: 'embedding/ruri'}), 'embedding/ruri');
});

test('catalog state: newest job wins, failed offers a forced retry, installed has no action', () => {
  const entry = {repo_id: 'o/model'};
  const installed = [{local_name: 'model', status: 'installed'}];
  assert.deepEqual(catalogState(entry, [], []).action, {label: '取得', force: false});
  assert.equal(catalogState(entry, [], []).state, '未取得');
  assert.equal(catalogState(entry, installed, []).state, 'installed');
  assert.equal(catalogState(entry, installed, []).action, null);
  for (const status of ['queued', 'downloading']) {
    const state = catalogState(entry, [], [{local_name: 'model', status, created_at: 1}]);
    assert.equal(state.state, status);
    assert.equal(state.action, null);
  }
  const failed = catalogState(entry, installed, [{local_name: 'model', status: 'failed', created_at: 1}]);
  assert.equal(failed.state, 'failed');
  assert.deepEqual(failed.action, {label: '再試行', force: true});
  const retried = catalogState(entry, installed, [
    {local_name: 'model', status: 'failed', created_at: 1},
    {local_name: 'model', status: 'completed', created_at: 2},
  ]);
  assert.equal(retried.state, 'installed', 'a successful retry clears the failed state');
  assert.equal(catalogState(entry, [], [{local_name: 'model', status: 'completed', created_at: 1}]).state, 'completed');
  assert.equal(catalogState(entry, [], [{local_name: 'other', status: 'failed', created_at: 1}]).state, '未取得');
});

test('job signature ignores progress-only changes', () => {
  const a = {items: [{local_name: 'm', status: 'downloading', progress_percent: 1}, {local_name: 'x', status: 'installed'}]};
  const b = {items: [{local_name: 'm', status: 'downloading', progress_percent: 50}, {local_name: 'x', status: 'installed'}]};
  const c = {items: [{local_name: 'm', status: 'failed'}, {local_name: 'x', status: 'installed'}]};
  assert.equal(jobSignature(a), jobSignature(b));
  assert.notEqual(jobSignature(a), jobSignature(c));
});

test('playground validates boundary values and preserves the OpenAI payload shape', () => {
  assert.deepEqual(testPayload('m', ' A ', ' B ', ''), {model: 'm', input: ['A', 'B'], encoding_format: 'float'});
  assert.equal(testPayload('m', 'a'.repeat(2000), 'B', '1').dimensions, 1);
  for (const dims of ['0', '-1', '1.5', 'NaN', 'Infinity', '9007199254740992']) assert.throws(() => testPayload('m', 'A', 'B', dims));
  assert.throws(() => testPayload('', 'A', 'B', ''));
  assert.throws(() => testPayload('m', ' ', 'B', ''));
  assert.throws(() => testPayload('m', 'a'.repeat(2001), 'B', ''));
});

test('cosine similarity is bounded and rejects malformed vectors', () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  assert.equal(cosineSimilarity([1, 0], [-1, 0]), -1);
  assert.equal(cosineSimilarity([0, 0], [1, 0]), null);
  for (const bad of [[], [1], [NaN, 0], [Infinity, 0], 'base64']) assert.throws(() => cosineSimilarity([1, 0], bad));
});

test('playground offers only loaded embedding models', () => {
  assert.deepEqual(loadedEmbeddingModels({loaded: {b: 'embedding', r: 'reranker', a: 'embedding'}}), ['a', 'b']);
  assert.deepEqual(loadedEmbeddingModels({}), []);
});

test('the /v1 client keeps a reverse-proxy prefix and stays on that origin', async () => {
  let observed;
  globalThis.fetch = async (url, init) => { observed = {url, init}; return new Response('{"object":"list","data":[]}', {headers: {'content-type': 'application/json'}}); };
  const v1 = v1Client({url: path => new URL(path, 'https://host.test/prefix/admin/').href});
  assert.deepEqual(await v1('v1/models/catalog'), {object: 'list', data: []});
  assert.equal(observed.url.href, 'https://host.test/prefix/v1/models/catalog');
  assert.equal(observed.init.credentials, 'same-origin');
  await assert.rejects(v1('https://elsewhere.test/v1/models'), /within the configured admin origin/);
});
