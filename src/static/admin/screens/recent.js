/** Shared Recent Requests screen: columns are declared by the server. */
import { defineScreen } from '../core/registry.js';
import { el, panelHeader } from '../core/dom.js';
import { renderTable } from '../core/table.js';
import { formatClock } from '../core/format.js';

let unsubscribe = null;

export default defineScreen({
  id: 'recent',
  section: 'Observability',
  label: '📊 Recent Requests',

  mount(panel, ctx) {
    panelHeader(panel, 'Recent Requests');
    const meta = el('div', 'obs-poll-meta', '読み込み中…');
    meta.id = 'obs-poll-meta';
    const scroll = el('div', 'table-scroll');
    const table = el('table', 'obs-table');
    table.id = 'obs-table';
    scroll.append(table);
    panel.append(meta, scroll);
    unsubscribe = ctx.feed.subscribe('interactions', data => {
      if (!data || !Array.isArray(data.columns) || !Array.isArray(data.items)) {
        ctx.showError('interactions: columns/items が不正です');
        return;
      }
      renderTable(table, data.columns, data.items, { emptyText: 'まだ interaction がありません', errorKey: 'error' });
      const state = ctx.feed.status();
      meta.textContent = '最終更新: ' + formatClock() + (state.mode === 'sse' ? ' (SSE)' : ' (polling, 5 秒間隔)');
    });
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
