/** embedding-api: client for the OpenAI-compatible /v1 API used by service screens.
 * Resolved from the admin base's parent so a reverse-proxy prefix survives
 * (/prefix/admin/ -> /prefix/v1/...). Reuses the shared HTTP adapter.
 */
import { createHttpClient } from '../core/api.js';

export function v1Client(ctx) {
  return createHttpClient(new URL('../', ctx.url('')));
}
