/** Shared Models screen: local models + download jobs + load state in one table.
 * Capabilities come from the server; this file never checks the service name.
 */
import { defineScreen } from '../core/registry.js';
import { button, checkbox, el, field, panelHeader } from '../core/dom.js';
import { renderTable } from '../core/table.js';
import { formatBytes, formatClock, formatPercent } from '../core/format.js';
import { json } from '../core/api.js';

const BASE_COLUMNS = [
  { key: 'status', label: 'status', kind: 'status' },
  { key: 'local_name', label: 'local_name', kind: 'mono' },
];
const JOB_COLUMNS = [
  { key: 'repo_id', label: 'repo_id', kind: 'mono' },
  { key: 'progress_text', label: 'progress', kind: 'mono' },
  { key: 'speed_text', label: 'speed', kind: 'mono' },
  { key: 'size_text', label: 'size', kind: 'mono' },
  { key: 'error', label: 'error', kind: 'text' },
];
const LOADED_COLUMN = { key: 'loaded_text', label: 'loaded', kind: 'text' };
const ACTIONS_COLUMN = { key: '_actions', label: 'actions', kind: 'actions' };

const ACTIVE_STATUSES = new Set(['queued', 'downloading']);

export function columnsFor(listing) {
  const caps = listing.capabilities || {};
  const manageable = caps.load || caps.unload;
  const hasJobs = caps.download || caps.cancel || (listing.items || []).some(item => item.status !== 'installed');
  return [
    ...BASE_COLUMNS,
    ...(manageable ? [LOADED_COLUMN] : []),
    ...(hasJobs ? JOB_COLUMNS : []),
    ...(listing.columns || []),
    ...(manageable || caps.cancel ? [ACTIONS_COLUMN] : []),
  ];
}

export function toRow(item) {
  const installed = item.status === 'installed';
  return {
    ...item,
    loaded_text: item.loaded === null || item.loaded === undefined ? '' : (item.loaded ? 'yes' : 'no'),
    repo_id: installed ? '-' : item.repo_id,
    progress_text: installed ? '-' : formatPercent(item.progress_percent),
    speed_text: installed ? '-' : (item.speed_mbps == null ? '-' : item.speed_mbps + ' MB/s'),
    size_text: installed ? formatBytes(item.size_bytes) : formatBytes(item.downloaded_bytes) + ' / ' + formatBytes(item.total_bytes),
  };
}

export function parsePrefixes(text) {
  return String(text || '').split(',').map(part => part.trim()).filter(Boolean);
}

export function applyFilter(items, prefixes) {
  if (!prefixes.length) return items;
  return items.filter(item => prefixes.some(prefix => item.local_name.startsWith(prefix)));
}

function encodeName(localName) {
  return localName.split('/').map(encodeURIComponent).join('/');
}

let unsubscribe = null;

export default defineScreen({
  id: 'models',
  section: 'Models',
  label: '↓ Model Downloads',

  async mount(panel, ctx) {
    const refs = { table: null, meta: null, filterInput: null, listing: null };
    const filterKey = ctx.service + '-dl-path-filter';
    let filterText = '';
    try { filterText = localStorage.getItem(filterKey) || ''; } catch { /* storage disabled */ }

    const listing = ctx.feed.cached('models') || await ctx.api('models');
    const caps = listing.capabilities || {};
    panelHeader(panel, caps.download ? 'Model Downloads' : 'Models');

    if (caps.download) {
      const grid = el('div', 'fields-grid');
      const repo = field(grid, { label: 'repo_id', id: 'dl-repo-id', placeholder: 'owner/repo' });
      const name = field(grid, { label: 'local_name', id: 'dl-local-name', placeholder: '省略時は repo 名', help: 'models_dir からの相対パス。階層も可 (例: embedding/Qwen3-0.6B)' });
      const force = checkbox(grid, { label: 'force (既存ディレクトリに上書き)', id: 'dl-force' });
      // Service-declared extra inputs (e.g. a pinned revision); keys travel in body.extra.
      const extraInputs = (listing.download_fields || []).map(spec => [spec.key, field(grid, {
        label: spec.label, id: 'dl-extra-' + spec.key, placeholder: spec.placeholder || '', help: spec.help || undefined,
        className: spec.full ? 'field-block full' : 'field-block',
      })]);
      const actions = el('div', 'field-block actions');
      const start = button('ダウンロード開始', 'btn btn-primary', async () => {
        const repoId = repo.value.trim();
        if (!repoId) { ctx.showError('repo_id is required'); return; }
        const extra = Object.fromEntries(extraInputs.map(([key, input]) => [key, input.value.trim()]).filter(([, value]) => value !== ''));
        start.disabled = true;
        ctx.setBusy(true);
        try {
          await ctx.api('models/download', json('POST', { repo_id: repoId, local_name: name.value.trim() || null, force: force.checked, extra }));
          await ctx.feed.refresh('models');
        } catch (error) {
          ctx.showError(error.message);
        } finally {
          start.disabled = false;
          ctx.setBusy(false);
        }
      });
      start.id = 'dl-start';
      actions.append(start);
      grid.append(actions);
      panel.append(grid);
    }

    const filterRow = el('div', 'inline-row');
    const filterLabel = el('label', '', 'パスフィルター:');
    filterLabel.htmlFor = 'dl-path-filter';
    refs.filterInput = el('input', 'field-input');
    refs.filterInput.id = 'dl-path-filter';
    refs.filterInput.placeholder = 'tts/  または  tts/, stt/  (カンマ区切り)';
    refs.filterInput.value = filterText;
    refs.filterInput.addEventListener('input', () => {
      try { localStorage.setItem(filterKey, refs.filterInput.value); } catch { /* storage disabled */ }
      render(refs.listing);
    });
    filterRow.append(filterLabel, refs.filterInput);
    panel.append(filterRow);

    refs.meta = el('div', 'obs-poll-meta', '読み込み中…');
    refs.meta.id = 'dl-poll-meta';
    const scroll = el('div', 'table-scroll');
    refs.table = el('table', 'obs-table');
    refs.table.id = 'dl-table';
    scroll.append(refs.table);
    panel.append(refs.meta, scroll);

    const manage = async (item, action) => {
      ctx.setBusy(true);
      try {
        await ctx.api('models/' + encodeName(item.local_name) + '/' + action, { method: 'POST' });
        await ctx.feed.refresh('models');
      } catch (error) {
        ctx.showError(error.message);
      } finally {
        ctx.setBusy(false);
      }
    };
    const cancel = async item => {
      ctx.setBusy(true);
      try {
        await ctx.api('models/downloads/' + encodeURIComponent(item.id) + '/cancel', { method: 'POST' });
        await ctx.feed.refresh('models');
      } catch (error) {
        ctx.showError(error.message);
      } finally {
        ctx.setBusy(false);
      }
    };
    const actionsFor = item => {
      if (item.status !== 'installed') {
        return caps.cancel && item.id && ACTIVE_STATUSES.has(item.status)
          ? [{ label: 'Cancel', className: 'btn-danger', onClick: () => cancel(item) }] : [];
      }
      if (item.loaded && caps.unload) return [{ label: 'Unload', className: 'btn-danger', onClick: () => manage(item, 'unload') }];
      if (!item.loaded && caps.load) return [{ label: 'Load', className: 'btn-primary', onClick: () => manage(item, 'load') }];
      return [];
    };

    const render = (data) => {
      if (!data || !refs.table) return;
      refs.listing = data;
      const rows = applyFilter((data.items || []).map(toRow), parsePrefixes(refs.filterInput.value));
      renderTable(refs.table, columnsFor(data), rows, {
        actions: actionsFor, pathPrefix: data.models_dir, errorKey: 'error', emptyText: 'モデルがありません',
      });
      const state = ctx.feed.status();
      refs.meta.textContent = '最終更新: ' + formatClock() + (state.mode === 'sse' ? ' (SSE)' : ' (polling)');
    };
    render(listing);
    unsubscribe = ctx.feed.subscribe('models', render);
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
