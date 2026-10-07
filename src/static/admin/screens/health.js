/** Server Health: device / backend / VRAM / loaded models (from /admin/health). */
import { defineScreen } from '../core/registry.js';
import { button, clear, el, keyValueRow, panelHeader } from '../core/dom.js';

let unsubscribe = null;

function render(container, data) {
  clear(container);
  const grid = el('div', 'health-grid');
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
  const entries = Object.entries(data.loaded || {});
  if (!entries.length) loaded.append(el('div', 'field-help', 'ロード済みモデルはありません。'));
  for (const [id, type] of entries) keyValueRow(loaded, id, type);
  container.append(grid, loaded);
}

export default defineScreen({
  id: 'health',
  section: 'Server',
  label: '♥ Server Health',

  mount(panel, ctx) {
    const recheck = button('↻ Refresh', 'btn btn-outline', () => ctx.feed.refresh('health').catch(error => ctx.showError(error.message)));
    panelHeader(panel, 'Server Health', recheck);
    const container = el('div');
    panel.append(container);
    unsubscribe = ctx.feed.subscribe('health', data => render(container, data || {}));
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
