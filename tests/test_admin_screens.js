// Pure helpers of the embedding-api service screens and the screen registry (node --test).
import {test, afterEach} from 'node:test';
import assert from 'node:assert/strict';
import screens, {feedLoaders, tabAliases, fallbackTab} from '../src/static/admin/screens/index.js';
import {primaryAction, stageLabel} from '../src/static/admin/screens/model_overview.js';
import {testPayload, cosineSimilarity, loadedEmbeddingModels} from '../src/static/admin/screens/playground.js';
import {v1Client} from '../src/static/admin/service/v1.js';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

test('registry order, sections and the health feed loader', async () => {
  assert.deepEqual(screens.map(s => s.id), ['models', 'playground', 'recent', 'health']);
  assert.equal(screens.find(s => s.id === 'recent').label, '📊 Inference Logs');
  assert.equal(screens.find(s => s.id === 'models').label, '▦ モデル管理');
  assert.deepEqual(screens.map(s => s.section), ['Models', 'Playground', 'Observability', 'Server']);
  const calls = [];
  await feedLoaders.health(path => { calls.push(path); return Promise.resolve({}); });
  assert.deepEqual(calls, ['health']);
  await feedLoaders.overview(path => { calls.push(path); return Promise.resolve({}); });
  assert.deepEqual(calls, ['health', 'models/overview']);
  for (const old of ['catalog', 'downloads']) assert.equal(tabAliases[old], 'models');
  assert.equal(tabAliases.activity, 'recent');
  assert.equal(tabAliases.settings, 'health');
  assert.equal(fallbackTab, 'models');
  for (const target of Object.values(tabAliases)) assert.ok(screens.some(s => s.id === target), target);
});

test('overview rows show the first enabled primary action and the load/unload state', () => {
  const steps = [{key: 'downloaded', done_label: '取得済み'}, {key: 'loaded', done_label: 'ロード中'}];
  const actions = (downloaded, loaded) => [
    {id: 'download', enabled: !downloaded, primary: !downloaded},
    {id: 'load', enabled: downloaded && !loaded, primary: downloaded && !loaded},
    {id: 'unload', enabled: loaded, primary: loaded},
  ];
  const item = (downloaded, loaded) => ({support: 'adapter', lifecycle: {downloaded, loaded}, actions: actions(downloaded, loaded)});
  assert.equal(primaryAction(item(false, false)).id, 'download');
  assert.equal(primaryAction(item(true, false)).id, 'load');
  assert.equal(primaryAction(item(true, true)).id, 'unload');
  assert.equal(stageLabel(item(false, false), steps).text, '未取得');
  assert.equal(stageLabel(item(true, false), steps).text, '取得済み');
  assert.equal(stageLabel(item(true, true), steps).text, 'ロード中');
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
