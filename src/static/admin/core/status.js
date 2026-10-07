/** Status vocabulary -> presentation. A table, not a branch chain.
 * Model statuses (spec): installed | queued | downloading | completed | failed.
 * Interaction statuses seen across services are mapped as well; unknown values
 * fall back to a neutral marker instead of being hidden.
 */
export const STATUS_PRESENTATION = Object.freeze({
  installed: { className: 'status-installed', label: 'installed' },
  queued: { className: 'status-active', label: 'queued' },
  downloading: { className: 'status-active', label: 'downloading' },
  completed: { className: 'status-ok', label: '✓ completed' },
  failed: { className: 'status-fail', label: '✗ failed' },
  ok: { className: 'status-ok', label: '✓ ok' },
  success: { className: 'status-ok', label: '✓ success' },
  error: { className: 'status-fail', label: '✗ error' },
});

export function presentStatus(status) {
  const known = STATUS_PRESENTATION[status];
  if (known) return known;
  return { className: 'status-unknown', label: '● ' + (status == null || status === '' ? 'unknown' : String(status)) };
}

export function loadedBadge(loaded) {
  return loaded
    ? { className: 'badge badge-loaded', label: '● loaded' }
    : { className: 'badge badge-unloaded', label: '○ unloaded' };
}
