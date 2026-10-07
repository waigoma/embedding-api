/** Shared engine status cards. tts composes renderEngineCards() into its Runtime screen. */
import { defineScreen } from '../core/registry.js';
import { button, clear, el, panelHeader, placeholder } from '../core/dom.js';
import { loadedBadge } from '../core/status.js';

export function renderEngineCards(container, items, { onUnload } = {}) {
  clear(container);
  if (!items.length) {
    container.append(placeholder('有効なエンジンがありません'));
    return;
  }
  for (const engine of items) {
    const card = el('div', 'engine-card');
    card.dataset.engine = engine.name;
    const header = el('div', 'engine-card-header');
    const nameWrap = el('div');
    nameWrap.append(el('div', 'engine-name', engine.display_name ? engine.display_name + ' (' + engine.name + ')' : engine.name));
    if (engine.description) nameWrap.append(el('div', 'engine-desc', engine.description));
    const side = el('div', 'engine-card-side');
    const badge = loadedBadge(engine.loaded);
    side.append(el('span', badge.className, badge.label));
    if (engine.loaded && engine.in_flight > 0) side.append(el('span', 'badge badge-busy', 'in-flight=' + engine.in_flight));
    if (engine.loaded && onUnload) {
      const unload = button('Unload', 'btn btn-sm btn-outline', () => onUnload(engine));
      unload.disabled = engine.in_flight > 0;
      side.append(unload);
    }
    header.append(nameWrap, side);
    card.append(header);
    for (const [key, value] of Object.entries(engine.config || {})) {
      const row = el('div', 'config-row');
      row.append(el('span', 'config-key', key), el('span', 'config-val', value === null || value === undefined || value === '' ? '—' : String(value)));
      card.append(row);
    }
    container.append(card);
  }
}

let unsubscribe = null;

export default defineScreen({
  id: 'engines',
  section: 'Runtime',
  label: '⚙ Runtime Status',

  mount(panel, ctx) {
    const recheck = button('↻ Re-check', 'btn btn-outline', () => ctx.feed.refresh('engines').catch(error => ctx.showError(error.message)));
    panelHeader(panel, 'Runtime Status', recheck);
    const container = el('div');
    container.id = 'engine-cards';
    panel.append(container);
    const onUnload = async engine => {
      ctx.setBusy(true);
      try {
        await ctx.api('engines/' + encodeURIComponent(engine.name) + '/unload', { method: 'POST' });
        await ctx.feed.refresh('engines');
      } catch (error) {
        ctx.showError(error.message);
      } finally {
        ctx.setBusy(false);
      }
    };
    unsubscribe = ctx.feed.subscribe('engines', data => renderEngineCards(container, data?.items || [], { onUnload }));
  },

  unmount() {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
  },
});
