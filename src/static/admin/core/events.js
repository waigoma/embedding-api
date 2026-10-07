/** Snapshot feed: one SSE connection with per-key polling fallback.
 * Screens subscribe to keys ("models", "interactions", "engines"). While the
 * EventSource is open, snapshots arrive pushed; on error (or without
 * EventSource) the feed polls the subscribed keys using the given loaders,
 * one request at a time, next delay starting after the previous load settles.
 */
export function createSnapshotFeed({
  url, loaders, intervalMs = 5000,
  EventSourceImpl = globalThis.EventSource,
  schedule = setTimeout, cancel = clearTimeout, now = Date.now,
}) {
  const subscribers = new Map(); // key -> Set<fn>
  const cache = new Map();       // key -> { data, at, source }
  let source = null;
  let mode = 'idle';
  let timer = null;
  let generation = 0;
  let polling = false;
  let lastUpdate = null;
  const listeners = new Set();

  function emitStatus() {
    for (const fn of listeners) fn(status());
  }
  function status() {
    return { mode, lastUpdate };
  }
  function deliver(key, data, sourceName) {
    const entry = { data, at: now(), source: sourceName };
    cache.set(key, entry);
    lastUpdate = entry.at;
    for (const fn of subscribers.get(key) || []) fn(data, entry);
    emitStatus();
  }

  async function pollOnce(version) {
    const current = () => polling && version === generation;
    if (!current()) return;
    for (const key of [...subscribers.keys()]) {
      if (!current()) return;
      const load = loaders[key];
      if (!load || !(subscribers.get(key) || new Set()).size) continue;
      try {
        const data = await load();
        if (current()) deliver(key, data, 'poll');
      } catch (error) {
        if (current()) for (const fn of listeners) fn({ ...status(), error });
      }
    }
    if (current()) timer = schedule(() => pollOnce(version), intervalMs);
  }
  function startPolling() {
    if (polling) return;
    polling = true;
    mode = 'poll';
    generation++;
    emitStatus();
    void pollOnce(generation);
  }
  function stopPolling() {
    polling = false;
    generation++;
    if (timer !== null) cancel(timer);
    timer = null;
  }

  function start() {
    if (!EventSourceImpl) { startPolling(); return; }
    if (source) return;
    source = new EventSourceImpl(url);
    source.addEventListener('open', () => { stopPolling(); mode = 'sse'; emitStatus(); });
    source.addEventListener('admin_update', event => {
      let snapshot;
      try { snapshot = JSON.parse(event.data); } catch { return; }
      if (!snapshot || typeof snapshot !== 'object') return;
      mode = 'sse';
      for (const [key, data] of Object.entries(snapshot)) deliver(key, data, 'sse');
    });
    source.addEventListener('error', () => { startPolling(); });
    // The browser reconnects by itself; polling covers the gap.
  }
  function stop() {
    stopPolling();
    if (source) { source.close(); source = null; }
    mode = 'idle';
    emitStatus();
  }
  function subscribe(key, fn) {
    if (!subscribers.has(key)) subscribers.set(key, new Set());
    subscribers.get(key).add(fn);
    const cached = cache.get(key);
    if (cached) fn(cached.data, cached);
    else if (mode === 'poll' && loaders[key]) void refresh(key).catch(() => {});
    return () => { subscribers.get(key)?.delete(fn); };
  }
  async function refresh(key) {
    const load = loaders[key];
    if (!load) throw new Error('No loader for feed key: ' + key);
    const data = await load();
    deliver(key, data, 'refresh');
    return data;
  }
  function onStatus(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }
  return { start, stop, subscribe, refresh, status, onStatus, cached: key => cache.get(key)?.data };
}
