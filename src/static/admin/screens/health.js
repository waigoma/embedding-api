/** embedding-api Server Health: device / backend / VRAM / loaded models with
 * inference evidence, plus the read-only startup configuration (from /admin/health).
 */
import { defineScreen } from '../core/registry.js';
import { button, clear, el, keyValueRow, panelHeader } from '../core/dom.js';
import { formatTime } from '../core/format.js';

let unsubscribe = null;

const list = values => (Array.isArray(values) && values.length ? values.join(', ') : 'なし');

function render(container, data) {
  clear(container);
  const grid = el('div', 'card-grid');
  const server = el('div', 'card');
  server.append(el('div', 'card-title', 'Server'));
  keyValueRow(server, 'status', data.status || 'unknown', data.status === 'ok' ? 'text-ok' : 'text-fail');
  keyValueRow(server, 'device', data.device || '-');
  keyValueRow(server, 'backend', data.accelerator_backend || '-');
  keyValueRow(server, 'mode', data.device_mode || '-');
  keyValueRow(server, 'auto_load', String(data.auto_load));
  keyValueRow(server, 'idle_ttl', data.idle_ttl ? data.idle_ttl + 's' : 'disabled');
  const gpu = el('div', 'card');
  gpu.append(el('div', 'card-title', 'GPU'));
  if (data.gpu) {
    keyValueRow(gpu, 'name', data.gpu.name);
    keyValueRow(gpu, 'vram used', data.gpu.vram_used_mb + ' / ' + data.gpu.vram_total_mb + ' MB');
    keyValueRow(gpu, 'vram free', data.gpu.vram_free_mb + ' MB');
  } else {
    keyValueRow(gpu, 'gpu', 'none', 'text-unknown');
  }
  grid.append(server, gpu);

  const loaded = el('div', 'card');
  loaded.append(el('div', 'card-title', 'Loaded models'));
  const models = Array.isArray(data.loaded_models) ? data.loaded_models : [];
  if (!models.length) loaded.append(el('div', 'field-help', 'ロード済みモデルはありません。'));
  for (const model of models) {
    const evidence = model.last_inference_at ? '推論確認 ' + formatTime(model.last_inference_at) : '推論未確認 (現ロード)';
    keyValueRow(loaded, model.id, model.type + ' · ' + evidence, model.last_inference_at ? 'text-ok' : 'text-warn');
  }
  loaded.append(el('div', 'field-help', '推論確認は現在のロードで成功した推論の時刻です。アンロード・再起動で消えます。'));

  const config = el('div', 'card');
  config.append(el('div', 'card-title', '起動設定 (読み取り専用)'));
  const c = data.config || {};
  keyValueRow(config, '保存ディレクトリ', c.model_dir ?? '-');
  keyValueRow(config, 'デバイス設定', c.device_mode ?? '-');
  keyValueRow(config, 'リクエスト時の自動ロード', c.auto_load ? '有効' : '無効');
  keyValueRow(config, 'アイドル時の自動解放', c.idle_ttl ? c.idle_ttl + ' 秒' : '無効');
  keyValueRow(config, '事前ロード · embedding', list(c.preload_embedding));
  keyValueRow(config, '事前ロード · reranker', list(c.preload_reranker));
  keyValueRow(config, 'Chat proxy', c.chat_proxy_configured ? '設定済み' : '未設定');
  config.append(el('div', 'field-help', '変更はサービスの起動設定 (環境変数) で行い、再起動後に反映します。'));
  container.append(grid, loaded, config);
}

export default defineScreen({
  id: 'health',
  section: 'Server',
  label: '♥ Server Health',

  mount(panel, ctx) {
    const recheck = button('↻ Refresh', 'btn btn-outline', () => ctx.feed.refresh('health').catch(error => ctx.showError(error.message)));
    panelHeader(panel, 'Server Health', recheck);
    const container = el('div');
    container.id = 'health-body';
    panel.append(container);
    unsubscribe = ctx.feed.subscribe('health', data => render(container, data || {}));
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
