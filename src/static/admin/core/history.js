/** URL <-> screen sync with browser history. Pure logic; the shell injects
 * window/history so this can run under node tests.
 *
 * Contract (kept from the services' earlier admin_navigation.js):
 * - `?tab=<screen>` (+ `&item=<id>` for sidebar-listed items) is the canonical deep link;
 *   `#<screen>` and `#<screen>/<id>` are accepted as a fallback.
 * - initial load and back/forward never push history; user navigation pushes once.
 * - unknown tabs normalise to `fallback` with replaceState; unrelated query/hash survive.
 * - a refused dirty-form guard on popstate moves history back to the rendered entry.
 */
export function createHistoryRouter({
  window: win, aliases = {}, fallback, itemParam = 'item', isKnown, onNavigate, confirmLeave = () => true,
}) {
  const history = win.history;
  let index = Number.isInteger(history.state?.adminTabIndex) ? history.state.adminTabIndex : 0;
  let rendered = null;      // { id, itemId } currently shown
  let renderedUrl = null;
  let restoring = false;
  let hashConsumed = false; // a `#screen` deep link is rewritten to `?tab=`; other hashes survive

  function read() {
    const url = new URL(win.location.href);
    let id = url.searchParams.get('tab');
    let itemId = url.searchParams.get(itemParam);
    if (!id && url.hash.length > 1) {
      const [hashId, hashItem] = url.hash.slice(1).split('/');
      id = hashId;
      itemId = hashItem ? decodeURIComponent(hashItem) : null;
      if (Object.prototype.hasOwnProperty.call(aliases, id)) id = aliases[id];
      hashConsumed = isKnown({ id, itemId: itemId || null });
    }
    if (Object.prototype.hasOwnProperty.call(aliases, id)) id = aliases[id];
    const target = { id, itemId: itemId || null };
    return isKnown(target) ? target : { id: fallback, itemId: null };
  }

  function destination(target) {
    const url = new URL(win.location.href);
    url.searchParams.set('tab', target.id);
    if (target.itemId != null) url.searchParams.set(itemParam, target.itemId);
    else url.searchParams.delete(itemParam);
    if (hashConsumed) { url.hash = ''; hashConsumed = false; }
    return url;
  }

  const same = (a, b) => a && b && a.id === b.id && (a.itemId ?? null) === (b.itemId ?? null);

  /** Called by the shell after it rendered `target`. mode: 'push' | 'replace' | 'none'. */
  function rendered_(target, mode) {
    rendered = { id: target.id, itemId: target.itemId ?? null };
    renderedUrl = destination(rendered).href;
    if (mode === 'push') { index += 1; history.pushState({ adminTabIndex: index }, '', renderedUrl); }
    else if (mode === 'replace') history.replaceState({ adminTabIndex: index }, '', renderedUrl);
  }

  function onPopState(event) {
    if (restoring) { restoring = false; return; }
    const next = event.state && event.state.adminTabIndex;
    if (!confirmLeave()) {
      if (Number.isInteger(next) && next !== index) { restoring = true; history.go(index - next); }
      else if (renderedUrl) history.replaceState({ adminTabIndex: index }, '', renderedUrl);
      return;
    }
    if (Number.isInteger(next)) index = next;
    const target = read();
    if (same(target, rendered)) { rendered_(target, 'replace'); return; }
    onNavigate(target, 'replace');
  }

  win.addEventListener('popstate', onPopState);

  return {
    read,
    rendered: rendered_,
    current: () => rendered,
    start() {
      const target = read();
      onNavigate(target, 'replace');
    },
  };
}
