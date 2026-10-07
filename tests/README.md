# Admin UI verification

The management UI is the shared admin shell (`src/admin_shell/`, `src/static/admin/`)
copied from the fastapi-admin-ui-template, served at `/admin/ui` with assets under
`/admin/assets/`. `/ui` and `/webui` redirect (307, relative) to it. It uses local
ES modules and needs no build or external fonts/scripts. `/admin/*` is
unauthenticated on purpose: the same trust boundary as `/v1/*`.

## Scope and expectations

| Requirement | Verification |
| --- | --- |
| Existing clients retain their contracts | `test_management_ui.py`: embeddings/responses/rerank aliases, float/base64 and dimensions, model/load/unload/download payloads (`DownloadStatusResponse` fields). |
| Reading UI state performs no inference/download/load | HTTP read tests (`/health`, `/admin/health`, `/admin/interactions`, catalog, downloads) assert loaders are not called; the browser asserts no POST on open. |
| Admin wiring | `test_admin.py`: `/admin/ui` without auth, redirects, assets + traversal 404, model roots and guessed type, `/v1` download validation 400/409/429 through the shared catalog, empty job list shape, load/unload 409s, partial-download load gate, `/admin/health`, `/admin/interactions` columns, SSE hub keys. |
| Downloaded files are distinct from loaded and inference-confirmed | Server Health shows per-load inference evidence; a failed/active download blocks the admin load. Config discovery does not verify weight completeness or provenance. |
| Service screens | `test_admin_screens.js`: registry order, catalog naming/state rules, playground payload/cosine validation, `/v1` client prefix handling. |
| Real browser | `test_admin_browser.cjs` (Chromium, desktop 1440x1000 and mobile 390x844): SSE feed, models table/actions, catalog one-click download, playground, logs, health, no overflow, `/prefix` reverse-proxy path. |

## CPU tests

Use Python 3.13 with the pinned CPU test dependencies. Inference runtimes and
torch are replaced in tests; HTTP/framework behavior is real. No model is downloaded.

```sh
python -m pip install -r tests/requirements.txt
npm ci --ignore-scripts --no-audit --no-fund
python -m pytest tests -q
npm test
```

## Browser fixture

Use a locally installed Playwright and Chromium. The checks run against the real
`server.app` (admin routers, SSE hub, `/v1` API) with synthetic vectors, stubbed
loaders and a fake download runner. They do **not** validate model compatibility,
real GPU inference, weight integrity or production proxy/TLS.

```sh
npx --no-install playwright install --with-deps chromium
npm run test:browser
```

The harness starts/stops its own loopback fixture on port 19975 and refuses an
occupied port. Use PLAYWRIGHT_MODULE / CHROMIUM_PATH for an existing local
installation and UI_ARTIFACT_DIR to select the screenshot directory. The browser
writes screenshots and `browser-qa.json`. Never run the fixture as a production
entrypoint.

## Operational limits

Download jobs live in memory (shared `DownloadJobRegistry`): at most 3 concurrent,
history capped, lost on restart, no cancellation, pinned revision or checksum
verification. Job logs are no longer kept (`logs_count` is 0, `last_log` null on
the legacy `/v1/models/downloads*` routes). Load type in the admin UI is guessed
from `config.json` architectures (`*ForSequenceClassification` = reranker) or a
`rerank` directory name; `/v1/models/load` still accepts an explicit type.
The playground only offers already-loaded embedding models; a concurrent
external unload may still cause the existing AUTO_LOAD behavior on its next request.

The CPU and UI tests workflow runs on pull requests, main updates and manual
dispatch. It checks out the exact PR head, uses pinned CPU/Playwright dependencies,
and has a 12-minute job limit with cancellation of superseded runs. No inference
runtime, model weights or production credentials are used. The existing Docker
publishing workflow is preserved.
