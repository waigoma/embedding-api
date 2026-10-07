/** embedding-api: preset model catalog (GET /v1/models/catalog) with one-click download.
 * Downloads go through the shared admin endpoint (POST models/download); state comes
 * from the 'models' feed plus the download ledger, so the newest job for a model wins.
 */
import { defineScreen } from '../core/registry.js';
import { el, panelHeader } from '../core/dom.js';
import { renderTable } from '../core/table.js';
import { formatClock } from '../core/format.js';
import { json } from '../core/api.js';
import { v1Client } from '../service/v1.js';

const COLUMNS = [
  { key: 'repo_id', label: 'repo_id', kind: 'mono' },
  { key: 'type', label: 'type', kind: 'text' },
  { key: 'local_name', label: '保存先', kind: 'mono' },
  { key: 'state', label: '状態', kind: 'status' },
  { key: 'note', label: '備考', kind: 'text' },
  { key: '_actions', label: '操作', kind: 'actions' },
];
const ACTIVE = new Set(['queued', 'downloading']);
const NOT_DOWNLOADED = '未取得';

/** Same policy as the previous UI: an explicit local_name, else the repo name. */
export function catalogLocalName(entry) {
  return entry.local_name || String(entry.repo_id).split('/').at(-1);
}

/** State of one catalog entry from the listing rows and the job ledger (newest first wins). */
export function catalogState(entry, listingItems, jobs) {
  const name = catalogLocalName(entry);
  const latest = [...jobs]
    .filter(job => job.local_name === name)
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0];
  const installed = listingItems.some(item => item.status === 'installed' && item.local_name === name);
  if (latest && ACTIVE.has(latest.status)) return { state: latest.status, note: '取得中', action: null };
  if (latest && latest.status === 'failed') {
    return { state: 'failed', note: '部分ファイルの可能性があります。再試行すると既存ファイルに上書き取得します。', action: { label: '再試行', force: true } };
  }
  if (installed) return { state: 'installed', note: 'ロード・推論は Model Downloads / Playground で確認してください。', action: null };
  if (latest && latest.status === 'completed') return { state: 'completed', note: 'モデル設定ファイルが見つかりません。', action: null };
  return { state: NOT_DOWNLOADED, note: '', action: { label: '取得', force: false } };
}

export function jobSignature(listing) {
  return JSON.stringify((listing?.items || []).filter(item => item.status !== 'installed').map(item => [item.local_name, item.status]));
}

let unsubscribe = null;
let current = null; // view state of the mounted screen; cleared on unmount

export default defineScreen({
  id: 'catalog',
  section: 'Models',
  label: '☆ カタログ',

  async mount(panel, ctx) {
    const v1 = v1Client(ctx);
    const view = { listing: null, jobs: [], entries: [], signature: null, alive: true };
    current = view;
    panelHeader(panel, 'モデルカタログ');
    panel.append(el('div', 'callout', 'ファイルの取得だけでは実行を保証しません。依存ライブラリ・モデル形式・メモリの確認と、ロード後の推論テストが必要です。任意のリポジトリは Model Downloads のフォームから取得できます。'));
    const meta = el('div', 'obs-poll-meta', '読み込み中…');
    const scroll = el('div', 'table-scroll');
    const table = el('table', 'obs-table');
    table.id = 'catalog-table';
    scroll.append(table);
    panel.append(meta, scroll);

    const render = () => {
      if (!view.alive) return;
      const items = view.listing?.items || [];
      const rows = view.entries.map(entry => ({
        repo_id: entry.repo_id,
        type: entry.type || '種別未判定',
        local_name: catalogLocalName(entry),
        entry,
        ...catalogState(entry, items, view.jobs),
      }));
      renderTable(table, COLUMNS, rows, {
        emptyText: 'カタログが空です (MODEL_CATALOG_JSON)。Model Downloads のフォームから取得先を指定できます。',
        actions: row => row.action ? [{ label: row.action.label, className: 'btn-primary', onClick: () => start(row.entry, row.action.force) }] : [],
      });
      meta.textContent = '最終更新: ' + formatClock();
    };

    const loadJobs = async () => {
      const data = await ctx.api('models/downloads');
      view.jobs = Array.isArray(data?.items) ? data.items : [];
    };

    const start = async (entry, force) => {
      ctx.setBusy(true);
      try {
        await ctx.api('models/download', json('POST', { repo_id: entry.repo_id, local_name: catalogLocalName(entry), force }));
        await loadJobs();
        await ctx.feed.refresh('models');
      } catch (error) {
        ctx.showError(error.message);
      } finally {
        ctx.setBusy(false);
        render();
      }
    };

    const [catalog] = await Promise.all([v1('v1/models/catalog'), loadJobs()]);
    view.entries = Array.isArray(catalog?.data) ? catalog.data : [];
    view.listing = ctx.feed.cached('models') || await ctx.api('models');
    view.signature = jobSignature(view.listing);
    if (!view.alive) return; // navigated away while loading
    render();
    // Job rows change only on state transitions or progress; refetch the ledger on transitions.
    unsubscribe = ctx.feed.subscribe('models', async listing => {
      view.listing = listing;
      const signature = jobSignature(listing);
      if (signature !== view.signature) {
        view.signature = signature;
        try { await loadJobs(); } catch (error) { ctx.showError(error.message); }
      }
      render();
    });
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
    if (current) current.alive = false;
    current = null;
  },
});
