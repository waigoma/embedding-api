/** Pure formatting helpers shared by every screen. */
export function formatBytes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = Number(value);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return (i === 0 ? String(Math.round(n)) : n.toFixed(1)) + ' ' + units[i];
}

export function formatDuration(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return '';
  const n = Number(ms);
  if (n < 1000) return Math.round(n) + 'ms';
  if (n < 60000) return (n / 1000).toFixed(1) + 's';
  return Math.floor(n / 60000) + 'm' + Math.round((n % 60000) / 1000) + 's';
}

export function toDate(value) {
  if (value === null || value === undefined || value === '') return null;
  const date = typeof value === 'number' ? new Date(value < 1e12 ? value * 1000 : value) : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatTime(value) {
  const date = toDate(value);
  return date ? date.toLocaleString('ja-JP') : '';
}

export function formatRelativeTime(value, now = Date.now()) {
  const date = toDate(value);
  if (!date) return '';
  const seconds = Math.max(0, (now - date.getTime()) / 1000);
  if (seconds < 60) return Math.floor(seconds) + 's ago';
  if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
  if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
  return date.toLocaleString('ja-JP');
}

export function formatPercent(value) {
  if (value === null || value === undefined) return '-';
  return Number(value).toFixed(1) + '%';
}

export function formatClock(date = new Date()) {
  return date.toLocaleTimeString('ja-JP');
}
