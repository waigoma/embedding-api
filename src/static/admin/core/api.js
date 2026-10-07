/** HTTP adapter: owns URLs, JSON encoding, credentials, and response handling.
 * Paths are relative to the admin base (e.g. "models", "engines/x/unload").
 */
export function createHttpClient(baseUrl, fetchImpl = fetch) {
  const base = new URL(baseUrl);
  if (!base.pathname.endsWith('/')) throw new Error('Admin base URL must end in /');
  return async function request(path, options = {}) {
    const url = new URL(path, base);
    if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname)) {
      throw new Error('Request must stay within the configured admin origin and path');
    }
    const headers = new Headers(options.headers);
    if (!headers.has('Accept')) headers.set('Accept', 'application/json');
    const isForm = typeof FormData !== 'undefined' && options.body instanceof FormData;
    if (options.body != null && !isForm && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    const response = await fetchImpl(url, {
      ...options, headers,
      credentials: 'same-origin', redirect: 'error', cache: 'no-store',
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body);
      } catch { /* Not JSON. */ }
      throw new Error(response.status + ' ' + detail);
    }
    if (response.status === 204) return null;
    const type = (response.headers.get('content-type') || '').split(';')[0].trim();
    return type === 'application/json' || type.endsWith('+json') ? response.json() : response.blob();
  };
}

export const json = (method, value) => ({ method, body: JSON.stringify(value) });
