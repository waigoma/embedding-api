/** Admin shell: sidebar from the screen registry, navigation with dirty/busy
 * guards, one snapshot feed, error banner, hash routing. It knows no screen.
 */
import screens, { feedLoaders } from './screens/index.js';
import { createHttpClient } from './core/api.js';
import { createSnapshotFeed } from './core/events.js';
import { groupBySection, validateRegistry } from './core/registry.js';
import { button, clear, el } from './core/dom.js';
import { formatClock } from './core/format.js';

const base = new URL('../', import.meta.url); // /<prefix>/admin/assets/shell.js -> /<prefix>/admin/
const api = createHttpClient(base);
const service = document.documentElement.dataset.service || 'admin';
const registry = validateRegistry(screens);
const byId = id => document.getElementById(id);
// Service registries add keys as `key: api => api('path')`; the shell binds the client.
const loaders = {
  models: () => api('models'),
  interactions: () => api('interactions?limit=50'),
  engines: () => api('engines'),
  ...Object.fromEntries(Object.entries(feedLoaders).map(([key, load]) => [key, () => load(api)])),
};
const feed = createSnapshotFeed({ url: new URL('events', base).href, loaders });

let current = null; // { screen, itemId }
let dirty = false;
let busy = false;
let errorTimer = null;
let settingHash = false;

const ctx = Object.freeze({
  api, feed, service,
  url: path => new URL(path, base).href,
  showError(message) {
    if (errorTimer !== null) clearTimeout(errorTimer);
    const banner = byId('err-banner');
    banner.textContent = '⚠ ' + message;
    banner.hidden = false;
    errorTimer = setTimeout(() => { banner.hidden = true; errorTimer = null; }, 6000);
  },
  markDirty() { dirty = true; renderSidebar(); },
  clearDirty() { dirty = false; renderSidebar(); },
  isDirty: () => dirty,
  confirmDiscard() { return !dirty || window.confirm('未保存の変更があります。破棄して続けますか?'); },
  setBusy(flag) {
    busy = Boolean(flag);
    byId('panel').setAttribute('aria-busy', String(busy));
  },
  isBusy: () => busy,
  navigate,
  refreshSidebar: () => renderSidebar(),
  setFooter(text) { byId('sidebar-footer').textContent = text; },
  current: () => current,
});

function sidebarButton(label, active, onClick) {
  const node = button(label, 'sidebar-item', onClick);
  node.classList.toggle('active', active);
  if (active) node.setAttribute('aria-current', 'page');
  if (active && dirty) node.classList.add('dirty');
  return node;
}

function renderSidebar() {
  const nav = byId('sidebar-nav');
  clear(nav);
  for (const group of groupBySection(registry)) {
    nav.append(el('div', 'section-label', group.section));
    for (const screen of group.screens) {
      const extras = screen.sidebar ? screen.sidebar(ctx) : {};
      nav.append(...(extras.before || []));
      if (screen.items) {
        for (const item of screen.items(ctx)) {
          const active = current?.screen === screen && current.itemId === item.id;
          const node = sidebarButton(item.label, active, () => navigate({ id: screen.id, itemId: item.id }));
          node.dataset.itemId = item.id;
          nav.append(node);
        }
      } else {
        const active = current?.screen === screen && current.itemId == null;
        nav.append(sidebarButton(screen.label, active, () => navigate({ id: screen.id })));
      }
      nav.append(...(extras.after || []));
    }
  }
}

function sameTarget(target) {
  return current && current.screen.id === target.id && (current.itemId ?? null) === (target.itemId ?? null);
}

async function navigate(target) {
  const screen = registry.find(entry => entry.id === target.id);
  if (!screen) { ctx.showError('Unknown screen: ' + target.id); return false; }
  if (target.itemId != null && !screen.mountItem) { ctx.showError('Screen has no items: ' + target.id); return false; }
  if (sameTarget(target) && !target.force) return true;
  if (busy) { ctx.showError('処理中です。完了してから移動してください'); return false; }
  if (!ctx.confirmDiscard()) return false;
  if (current) {
    try { current.screen.unmount(); } catch (error) { console.error(error); }
  }
  dirty = false;
  const panel = byId('panel');
  clear(panel);
  current = { screen, itemId: target.itemId ?? null };
  settingHash = true;
  location.hash = target.itemId != null ? screen.id + '/' + encodeURIComponent(target.itemId) : screen.id;
  renderSidebar();
  try {
    if (target.itemId != null) await screen.mountItem(panel, ctx, target.itemId);
    else await screen.mount(panel, ctx);
  } catch (error) {
    console.error(error);
    ctx.showError(error.message);
  }
  return true;
}

function targetFromHash() {
  const raw = location.hash.replace(/^#/, '');
  if (!raw) return null;
  const [id, item] = raw.split('/');
  return { id, itemId: item ? decodeURIComponent(item) : undefined };
}

window.addEventListener('hashchange', () => {
  if (settingHash) { settingHash = false; return; }
  const target = targetFromHash();
  if (target && !sameTarget(target)) void navigate(target);
});
window.addEventListener('beforeunload', event => {
  if (dirty || busy) { event.preventDefault(); event.returnValue = ''; }
});
window.addEventListener('pagehide', () => {
  feed.stop();
  if (current) current.screen.unmount();
});
feed.onStatus(state => {
  const node = byId('feed-status');
  if (!node) return;
  const mode = { sse: 'SSE', poll: 'polling', idle: '—' }[state.mode] || state.mode;
  node.textContent = mode + (state.lastUpdate ? ' · ' + formatClock(new Date(state.lastUpdate)) : '');
  if (state.error) ctx.showError(state.error.message);
});

renderSidebar();
feed.start();
// Screens that list items (voices) load them in init(); wait so a deep link resolves.
await Promise.all(registry.filter(screen => screen.init).map(screen =>
  Promise.resolve(screen.init(ctx)).catch(error => ctx.showError(error.message))));
const initial = targetFromHash();
if (initial) void navigate(initial);
