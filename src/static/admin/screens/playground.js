/** embedding-api Playground: embed two texts with a loaded embedding model and
 * compare them (cosine similarity, output dimensions, latency). Calls the real
 * OpenAI-compatible POST /v1/embeddings; the model list comes from the 'health' feed.
 */
import { defineScreen } from '../core/registry.js';
import { button, clear, el, field, panelHeader } from '../core/dom.js';
import { json } from '../core/api.js';
import { v1Client } from '../service/v1.js';

const MAX_TEXT = 2000;

/** Request body for POST /v1/embeddings; throws a user-facing message on invalid input. */
export function testPayload(model, a, b, dimensions) {
  if (!model) throw new Error('ロード済みの embedding モデルを選択してください。');
  const inputs = [String(a).trim(), String(b).trim()];
  if (inputs.some(text => !text || text.length > MAX_TEXT)) throw new Error('テキストはそれぞれ 1〜2,000 文字で入力してください。');
  const body = { model, input: inputs, encoding_format: 'float' };
  if (String(dimensions).trim() !== '') {
    const n = Number(dimensions);
    if (!Number.isSafeInteger(n) || n < 1) throw new Error('出力次元は 1 以上の整数にしてください。');
    body.dimensions = n;
  }
  return body;
}

export function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || a.length !== b.length
      || [...a, ...b].some(n => typeof n !== 'number' || !Number.isFinite(n))) {
    throw new Error('サーバーから有効なベクトルを取得できませんでした。');
  }
  const normA = Math.hypot(...a);
  const normB = Math.hypot(...b);
  if (!normA || !normB) return null;
  return Math.max(-1, Math.min(1, a.reduce((sum, v, i) => sum + v * b[i], 0) / (normA * normB)));
}

/** Loaded embedding model ids from the /admin/health payload (`loaded`: id -> type). */
export function loadedEmbeddingModels(health) {
  return Object.entries(health?.loaded || {})
    .filter(([, type]) => type === 'embedding')
    .map(([id]) => id)
    .sort();
}

function metric(parent, label, value) {
  const node = el('div', 'field-block');
  node.append(el('div', 'result-label', label), el('div', 'result-value', String(value)));
  parent.append(node);
}

function renderResult(container, data, elapsedMs) {
  clear(container);
  const ordered = [...(data?.data || [])].sort((x, y) => x.index - y.index);
  const a = ordered[0]?.embedding;
  const b = ordered[1]?.embedding;
  const similarity = cosineSimilarity(a, b);
  const card = el('div', 'card');
  card.append(el('div', 'card-title', 'ベクトル生成が完了しました'), el('div', 'field-help', String(data.model || '')));
  const metrics = el('div', 'result-metrics');
  metric(metrics, 'Cosine similarity', similarity == null ? '算出不可 (ゼロベクトル)' : similarity.toFixed(4));
  metric(metrics, '出力次元', a.length);
  metric(metrics, '応答時間', Math.round(elapsedMs) + ' ms');
  card.append(metrics, el('div', 'field-help', 'トークン数 ' + (data.usage?.total_tokens ?? '—') + ' (サーバーの概算)。類似度は品質の保証ではありません。'));
  const vector = el('details', 'result-vector');
  vector.append(el('summary', '', 'ベクトルの先頭 8 要素'), el('pre', 'json-result', JSON.stringify({ A: a.slice(0, 8), B: b.slice(0, 8) }, null, 2)));
  card.append(vector);
  container.append(card);
}

let unsubscribe = null;

export default defineScreen({
  id: 'playground',
  section: 'Playground',
  label: '▷ Playground',

  mount(panel, ctx) {
    const v1 = v1Client(ctx);
    panelHeader(panel, 'Embeddings を試す');
    panel.append(el('div', 'callout', 'ロード済みの embedding モデルで 2 つのテキストのベクトルと類似度を確認します。入力はこのサーバーへ送信されます。'));
    const grid = el('div', 'fields-grid');
    const model = field(grid, { label: 'ロード済みモデル', id: 'pg-model', tag: 'select' });
    const dimensions = field(grid, {
      label: '出力次元 (任意)', id: 'pg-dimensions', type: 'number', placeholder: 'モデルの標準次元',
      help: '上限はモデルごとに異なります。切り詰めの品質はモデルの MRL 対応に依存します。',
    });
    dimensions.min = '1';
    dimensions.step = '1';
    const textA = field(grid, { label: 'テキスト A', id: 'pg-text-a', tag: 'textarea', value: '猫は窓辺で眠っています。' });
    const textB = field(grid, { label: 'テキスト B', id: 'pg-text-b', tag: 'textarea', value: '窓のそばで猫が寝ています。' });
    textA.maxLength = MAX_TEXT;
    textB.maxLength = MAX_TEXT;
    const actions = el('div', 'field-block full');
    const validation = el('div', 'field-warn');
    validation.id = 'pg-validation';
    validation.setAttribute('role', 'alert');
    const result = el('div');
    result.id = 'pg-result';
    result.setAttribute('aria-live', 'polite');

    let models = [];
    const run = button('ベクトルを生成', 'btn btn-primary', async () => {
      validation.textContent = '';
      clear(result);
      let body;
      try {
        body = testPayload(model.value, textA.value, textB.value, dimensions.value);
      } catch (error) {
        validation.textContent = error.message;
        return;
      }
      run.disabled = true;
      run.textContent = '生成中…';
      ctx.setBusy(true);
      const started = performance.now();
      try {
        const data = await v1('v1/embeddings', json('POST', body));
        renderResult(result, data, performance.now() - started);
      } catch (error) {
        validation.textContent = error.message;
      } finally {
        ctx.setBusy(false);
        run.textContent = 'ベクトルを生成';
        run.disabled = !models.length;
        ctx.feed.refresh('health').catch(() => {});
      }
    });
    run.id = 'pg-run';
    actions.append(run, validation);
    grid.append(actions);
    panel.append(grid, result);

    model.addEventListener('change', () => clear(result));
    unsubscribe = ctx.feed.subscribe('health', health => {
      const next = loadedEmbeddingModels(health);
      if (JSON.stringify(next) !== JSON.stringify(models)) {
        const previous = model.value;
        models = next;
        model.replaceChildren(...models.map(id => {
          const option = el('option', '', id);
          option.value = id;
          return option;
        }));
        if (models.includes(previous)) model.value = previous;
      }
      model.disabled = !models.length;
      if (!ctx.isBusy()) run.disabled = !models.length;
      if (!models.length) validation.textContent = 'ロード済みの embedding モデルがありません。Model Downloads でロードしてください。';
      else if (validation.textContent.startsWith('ロード済みの embedding')) validation.textContent = '';
    });
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
