import {test} from 'node:test';
import assert from 'node:assert/strict';
import {modelState, validateDownload, testPayload, cosineSimilarity, formatBytes} from '../src/webui/state.js';

test('download completion never implies runtime or inference readiness', () => {
  const model = {id: 'synthetic', loaded: false, type: 'unknown'};
  assert.match(modelState(model, {}, [{local_name: model.id, status: 'completed'}])[0], /互換性未確認/);
  model.loaded = true; model.type = 'embedding';
  assert.match(modelState(model, {loaded: [{id: model.id, type: model.type, last_inference_at: null}]}, [])[0], /推論未確認/);
  assert.match(modelState(model, {loaded: [{id: model.id, type: model.type, last_inference_at: 42}]}, [])[0], /推論確認済み/);
  assert.match(modelState(model, {loaded: []}, [])[0], /推論未確認/);
});

test('latest job distinguishes in-progress, partial failure and successful retry', () => {
  const m = {id: 'synthetic', loaded: false};
  for (const status of ['queued', 'downloading']) assert.match(modelState(m, {}, [{local_name: m.id, status}])[0], /取得中/);
  assert.match(modelState(m, {}, [{local_name: m.id, status: 'failed'}])[0], /部分ファイル/);
  assert.match(modelState(m, {}, [{local_name: m.id, status: 'completed'}, {local_name: m.id, status: 'failed'}])[0], /互換性未確認/);
});

test('download input rejects malformed repositories and unsafe destinations', () => {
  for (const repo of ['', 'synthetic', 'a/b/c', '../model', 'a/..', 'a/b?token=x']) assert.ok(validateDownload(repo, ''));
  for (const dest of ['../x', './x', '/models/x', 'a//b', 'a/../b', 'a\\b', 'C:/x']) assert.ok(validateDownload('synthetic/model', dest));
  assert.equal(validateDownload('synthetic/model', ''), '');
  assert.equal(validateDownload('synthetic/model', 'embedding/日本語-model'), '');
});

test('playground validates boundary values and preserves OpenAI payload shape', () => {
  assert.deepEqual(testPayload('synthetic', ' A ', ' B ', ''), {model: 'synthetic', input: ['A', 'B'], encoding_format: 'float'});
  assert.equal(testPayload('synthetic', 'a'.repeat(2000), 'B', '1').dimensions, 1);
  for (const dims of ['0', '-1', '1.5', 'NaN', 'Infinity', '9007199254740992']) assert.throws(() => testPayload('synthetic', 'A', 'B', dims));
  assert.throws(() => testPayload('', 'A', 'B', ''));
  assert.throws(() => testPayload('synthetic', ' ', 'B', ''));
  assert.throws(() => testPayload('synthetic', 'a'.repeat(2001), 'B', ''));
});

test('cosine shows meaningful finite results and rejects malformed vectors', () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  assert.equal(cosineSimilarity([1, 0], [-1, 0]), -1);
  assert.equal(cosineSimilarity([0, 0], [1, 0]), null);
  for (const bad of [[], [1], [NaN, 0], [Infinity, 0], 'base64']) assert.throws(() => cosineSimilarity([1, 0], bad));
  assert.equal(formatBytes(null), '不明');
  assert.equal(formatBytes(1024), '1.0 KiB');
});
