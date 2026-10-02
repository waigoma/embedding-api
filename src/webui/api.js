// Resolve relative to the UI module to preserve a reverse proxy path prefix.
const base = new URL('../../', import.meta.url);

export async function request(path, {body, signal} = {}) {
  const timeout = new AbortController();
  const abort = () => timeout.abort();
  signal?.addEventListener('abort', abort, {once: true});
  if (signal?.aborted) timeout.abort();
  const timer = setTimeout(abort, body ? 120000 : 15000);
  try {
    const response = await fetch(new URL(path, base), {
      method: body ? 'POST' : 'GET', credentials: 'same-origin', cache: 'no-store',
      headers: body ? {'Content-Type': 'application/json'} : {},
      body: body ? JSON.stringify(body) : undefined, signal: timeout.signal,
    });
    const raw = await response.text();
    let data;
    try { data = raw ? JSON.parse(raw) : {}; }
    catch { throw new Error(response.ok ? 'サーバーの応答が JSON ではありません。' : `HTTP ${response.status} · サーバーの応答を確認してください。`); }
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(d => d.msg).join(' / ')
        : data.detail || data.error?.message || response.statusText;
      throw new Error(`HTTP ${response.status} · ${detail}`);
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError' && !signal?.aborted)
      throw new Error(body ? '応答を確認できません。処理が継続している可能性があります。状態を更新してから再操作してください。' : '状態の取得がタイムアウトしました。接続を確認して再取得してください。');
    throw error;
  } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
}

export async function snapshot(signal) {
  const paths = ['health', 'v1/models', 'v1/models/catalog', 'v1/models/downloads', 'v1/logs/inference?limit=50', 'ui/status'];
  const results = await Promise.allSettled(paths.map(path => request(path, {signal})));
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const names = ['health', 'models', 'catalog', 'jobs', 'logs', 'status'];
  const values = {}, failures = [];
  results.forEach((result, i) => {
    if (result.status === 'fulfilled') values[names[i]] = result.value;
    else failures.push(`${names[i]}: ${result.reason.message}`);
  });
  return {values, failures};
}
