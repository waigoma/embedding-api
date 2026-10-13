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
| Service screens | `test_admin_screens.js`: registry order, model-overview registry and action rules, playground payload/cosine validation, `/v1` client prefix handling. |
| Real browser | `test_admin_browser.cjs` (Chromium, desktop 1440x1000 and mobile 390x844): SSE feed, model overview (families, roots scope, drawer, load/unload, catalog download, old `?tab=` aliases), playground, logs, health, no overflow, `/prefix` reverse-proxy path. |

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

## Model profiles and real inference

`test_model_profiles.py` covers official query/document prefix selection, catalog and
local-name resolution, profile-specific remote-code settings, Gemma FP16 rejection,
PLaMo adapter dispatch, HTTP aliases/default `input_type`, MRL fallback normalization
and rejection of non-finite output. They also cover Voyage mask argument compatibility,
GTE non-persistent buffer restoration and PLaMo's current SentencePiece constructor.
These are CPU contracts with inference doubles.

Run the real model smoke test from the repository root in an isolated environment
with production requirements, a compatible PyTorch install, and sufficient model
storage. Each model gets a separate subprocess, sequentially, so VRAM is released
before the next model. The script writes one JSON result per model with dependency
versions, load/inference results, dimensions and the download's pinned HF revision.
Existing local models are reused; their revision is unknown unless a
`.smoke-model.json` receipt records it. Config presence does not establish provenance.

```sh
DEVICE_MODE=cuda python scripts/smoke_models.py --models-dir /tmp/embedding-smoke/models
# Select a smaller group:
DEVICE_MODE=cuda python scripts/smoke_models.py --models-dir /tmp/embedding-smoke/models \
  --models google/embeddinggemma-2 cl-nagoya/ruri-v3-310m microsoft/harrier-oss-v1-0.6b
```

The script checks query/document/native/128-dimensional embeddings for finite values
and L2 normalization, and rerank output for finite scores and expected ordering.
It downloads missing models from official repositories pinned to `model_info().sha`.
It does not accept gated model terms. Download/load/inference exceptions are recorded
and the remaining models continue. All checkpoints, CPU/AMD, performance under load,
and hardware capacity are not established by a representative CUDA smoke test.

## 2026-10-13 CUDA verification

The final representative smoke run passed for 17 unique checkpoints on an RTX 5060 Ti
16 GB, using PyTorch `2.11.0+cu130`, Sentence Transformers `6.1.0`, Transformers
`5.19.0`, and Hugging Face Hub `1.33.0`. The 15 embedding checkpoints passed
query/document/native/128-dimension shape, finite-value and L2 checks (tolerance
`2e-3`). The two rerankers returned finite scores with the relevant document first.
This is inference compatibility evidence, not a retrieval-quality benchmark.

The [recorded results](results/gpu_smoke_2026-10-13.json) retain dimensions, scores,
norms when recorded, versions, local names and the downloaded revision. Ruri and
Harrier reused existing local weights: inference passed, but their HF revisions and
provenance were not established. No gated terms were accepted. The listed revisions
identify weight snapshots; GTE additionally loads the official `Alibaba-NLP/new-impl`
remote code, whose separate revision is not captured in this JSON record.

| Official model | Check | Weight revision |
| --- | --- | --- |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | PASS: 768d / 128d, query + document | [`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/tree/9bbca17d9273fd0d03d5725c7a4b0f6b45142062) |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | PASS: 1024d / 128d, query + document | [`5617a9f61b028005a4858fdac845db406aefb181`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181) |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | PASS: rerank ordering | [`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | PASS: 1024d / 128d, query + document | [`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | PASS: rerank ordering | [`e61197ed45024b0ed8a2d74b80b4d909f1255473`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473) |
| [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) | PASS: 768d / 128d, query + document | Unknown (existing local weights) |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | PASS: 768d / 128d, query + document | [`914f7f89142e33e77833254d9c9b90c3cef7303b`](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) |
| [hotchpotch/bekko-embedding-v1-a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m) | PASS: 384d / 128d, query + document | [`c721113d59a1d91b447450324f51c4b3332c924a`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/tree/c721113d59a1d91b447450324f51c4b3332c924a) |
| [ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | PASS: 384d / 128d, query + document | [`835ad14087e140460703cf0fae09f97d469d65c2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/tree/835ad14087e140460703cf0fae09f97d469d65c2) |
| [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | PASS: 1024d / 128d, query + document | [`274baa43b0e13e37fafa6428dbc7938e62e5c439`](https://huggingface.co/intfloat/multilingual-e5-large-instruct/tree/274baa43b0e13e37fafa6428dbc7938e62e5c439) |
| [jinaai/jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | PASS: 768d / 128d, query + document | [`8a7f00aac812071b69403df470f1038ec85f8925`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/tree/8a7f00aac812071b69403df470f1038ec85f8925) |
| [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | PASS: 1024d / 128d, query + document | Unknown (existing local weights) |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | PASS: 2048d / 128d, query + document | [`c0c9fea93ea424587517f2c59e20db9f1d6bf615`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/tree/c0c9fea93ea424587517f2c59e20db9f1d6bf615) |
| [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | PASS: 1024d / 128d, query + document | [`2c4d510dd4a732063c31a0f70193e35067b51fd8`](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b/tree/2c4d510dd4a732063c31a0f70193e35067b51fd8) |
| [pfnet/plamo-embedding-1b](https://huggingface.co/pfnet/plamo-embedding-1b) | PASS: 2048d / 128d, query + document | [`f47268047fb3df74113675700c1e9dacdc68cb47`](https://huggingface.co/pfnet/plamo-embedding-1b/tree/f47268047fb3df74113675700c1e9dacdc68cb47) |
| [sbintuitions/sarashina-embedding-v2-1b](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b) | PASS: 1792d / 128d, query + document | [`1f3408afaa7b617e3445d891310a9c26dd0c68a5`](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b/tree/1f3408afaa7b617e3445d891310a9c26dd0c68a5) |
| [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | PASS: 2048d / 128d, query + document | [`67fabc9bef010dabc5f6024aa1b1b6b93410426f`](https://huggingface.co/voyageai/voyage-4-nano/tree/67fabc9bef010dabc5f6024aa1b1b6b93410426f) |

The following 12 catalog checkpoints were not downloaded or inferred in this run.
Their profiles follow the official engine/module/prefix contracts; this run does not
establish their individual compatibility or memory capacity on 16 GB:

- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Embedding-8B`
- `Qwen/Qwen3-Reranker-4B`
- `cl-nagoya/ruri-v3-130m`
- `cl-nagoya/ruri-v3-70m`
- `cl-nagoya/ruri-v3-30m`
- `cl-nagoya/ruri-v3-reranker-310m`
- `microsoft/harrier-oss-v1-270m`
- `ibm-granite/granite-embedding-311m-multilingual-r2`
- `perplexity-ai/pplx-embed-v1-4b`
- `google/embeddinggemma-300m`
- `hotchpotch/bekko-embedding-v1-a25m`

`google/embeddinggemma-300m` also requires the user's own gated license acceptance
and token. The larger 4B/8B checkpoints were not claimed to fit this GPU. CPU/AMD,
concurrent requests and performance testing remain unmeasured.
