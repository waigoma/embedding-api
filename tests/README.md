# Management UI verification

The UI shares the speech gateways' dark panels, 230px sidebar, controls and
responsive navigation. It is served at both `/ui` and `/webui`, uses local ES
modules, and needs no build or external fonts/scripts.

## Scope and expectations

| Requirement | Verification |
| --- | --- |
| Existing clients retain their contracts | Real FastAPI HTTP tests cover embeddings/responses/rerank aliases, float/base64 and dimensions, model/load/unload/download payloads. |
| Reading UI state performs no inference/download/load | HTTP read tests and browser request assertions. |
| Downloaded files are distinct from loaded and inference-confirmed | State transitions, failed partial-download gate, unload/reload evidence reset. Config discovery does not verify weight completeness or provenance. |
| Forms give actionable errors | Repository/path validation, text length/dimension boundaries, backend 400s, partial outage and recovery. |
| Mobile and keyboard access | 1440×1000 and 390×844 Chromium checks, labelled forms, native navigation buttons/dialog, visible focus, no page overflow. |
| Server strings stay literal | Hostile synthetic model name and no generated image element; rendering uses textContent. |
| Deployment paths work | `/ui`, `/webui`, and `/prefix/webui` assets and API resolution. |
| Refresh preserves work | Inputs/selection, expanded download logs, one refresh cycle, aborted read ignored. |

## CPU tests

Use Python 3.13 with the pinned CPU test dependencies: Inference
runtimes and torch are replaced in tests; HTTP/framework behavior is real.
No model is downloaded.

```sh
python -m pip install -r tests/requirements.txt
npm ci --ignore-scripts --no-audit --no-fund
python -m pytest tests -q
npm test
```

## Browser fixture

Use a locally installed Playwright and Chromium. These checks run against
synthetic vectors and simulated resources. They do **not** validate model
compatibility, real GPU inference, weight integrity or production proxy/TLS.

```sh
npx --no-install playwright install --with-deps chromium
npm run test:browser
```

The browser harness starts/stops its own loopback fixture, refuses an occupied
port, runs the desktop/mobile scenario and six isolated interaction regressions.
Use PLAYWRIGHT_MODULE / CHROMIUM_PATH for an existing local installation and
UI_ARTIFACT_DIR to select a screenshot directory. The regressions cover repeated
cancel/Escape, double confirmation, read failure after a successful write, load
failure/retry, duplicate Playground submissions and coalesced refresh requests.

The fixture stubs both loaders and download workers, and must never be used as
a production entrypoint. The browser writes screenshots and `browser-qa.json`.
Do not run it against a production service: its fixture-only reset endpoints
and explicit synthetic download/load/unload actions are intentional.

## Operational limits

`GET /ui/status` exposes only non-secret startup configuration and the last
successful inference time for each currently loaded model. An unload/reload
or restart clears that evidence. Existing model listing/health/inference API
payloads remain unchanged. The settings view is read-only; no config write,
authentication policy, dependency/runtime upgrade or deployment is included.
The playground only offers already-loaded embedding models; a concurrent
external unload may still cause the existing AUTO_LOAD behavior on its next
request. This UI cannot reserve a model or GPU memory.

The backend's existing downloader has no cancellation or pinned-revision /
checksum-verification contract. A completed download is labelled as inference
unverified, and local config detection is labelled as compatibility unverified.
The UI never claims those files are verified weights. A failed job blocks its
local model's UI load until retried successfully. Same-name catalog candidates
without provenance are labelled explicitly.

The CPU and UI tests workflow runs on pull requests, main updates and manual
dispatch. It checks out the exact PR head, uses pinned CPU/Playwright dependencies,
and has a 12-minute job limit with cancellation of superseded runs. No inference
runtime, model weights or production credentials are used.

This personal repository is public. Runner enumeration returned a permission
error, so no usable self-hosted runner was verified. Its existing Docker workflow
already uses ubuntu-latest; standard hosted runner minutes are free for public
repositories ([GitHub billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)).
The test workflow creates no caches or uploaded artifacts, and records evidence
in logs/the job summary. The existing Docker publishing workflow is preserved.
