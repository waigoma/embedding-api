# embedding-api

OpenAI-compatible な Embedding / Rerank API サーバーです。

## GPU モード切替

`server.py` は `DEVICE_MODE` で実行デバイスを切り替えます。

- `auto`: GPU (CUDA/HIP) があれば使用、なければ CPU
- `gpu` / `cuda`: GPU を強制利用 (GPU がなければ起動失敗)
- `cpu`: CPU のみ

`/health` には `accelerator_backend` が含まれます。
- NVIDIA 環境: `cuda`
- AMD ROCm 環境: `rocm`

## Docker Compose の使い分け

### NVIDIA

```bash
docker compose -f compose.yml -f compose.nvidia.yml up -d
```

### AMD ROCm

```bash
docker compose -f compose.yml -f compose.amd.yml up -d
```

## ローカルで Docker image をビルド (NVIDIA / AMD)

`Dockerfile.nvidia` と `Dockerfile.amd` の 2 つを用意しています。  
環境に応じて `FLAVOR` を切り替えてビルドします。

```bash
FLAVOR=nvidia ./build.sh
FLAVOR=amd ./build.sh
```

手動で実行する場合:

```bash
docker compose -f compose.yml -f compose.build.nvidia.yml build
docker compose -f compose.yml -f compose.build.amd.yml build
```

## ghcr.io イメージを使う

`compose.yml` は `EMBEDDING_IMAGE` を参照するため、`ghcr.io` のイメージをそのまま指定できます。

### 1) (Private の場合) 認証

```bash
docker login ghcr.io
```

### 2) イメージ指定して起動

`AMD ROCm` の例 (`workflow` で publish される `-amd` image):

```bash
export EMBEDDING_IMAGE=ghcr.io/<owner>/<repo>-amd:<tag>
docker compose -f compose.yml -f compose.amd.yml up -d
```

`NVIDIA` の例 (`workflow` で publish される `-nvidia` image):

```bash
export EMBEDDING_IMAGE=ghcr.io/<owner>/<repo>-nvidia:<tag>
docker compose -f compose.yml -f compose.nvidia.yml up -d
```

### 3) pull のみ先に実行したい場合

```bash
export EMBEDDING_IMAGE=ghcr.io/<owner>/<repo>-amd:<tag>   # or -nvidia
docker compose pull
```

### GitHub Actions で自動 publish

`.github/workflows/publish-ghcr.yml` を追加しています。

- `main` への push で publish
- `v*` タグ push で publish
- `workflow_dispatch` で手動実行
- `nvidia` / `amd` の 2 種類を matrix build

デフォルトの publish 先:

```text
ghcr.io/<owner>/<repo>-nvidia
ghcr.io/<owner>/<repo>-amd
```

## Qwen3-Embedding-4B の注意

`Qwen/Qwen3-Embedding-4B` は 4B モデルのため、FP16 では VRAM 消費が大きくなります。  
8GB クラス GPU では OOM になる可能性が高く、16GB クラスでも運用条件次第で厳しくなる場合があります。

- バッチサイズを小さくする
- 同時リクエストを制限する
- より軽量な `Qwen3-Embedding-0.6B` で先に運用検証する

SentenceTransformer に渡すオプションは JSON で指定できます。

```bash
export SENTENCE_TRANSFORMER_KWARGS='{"model_kwargs":{"attn_implementation":"flash_attention_2","torch_dtype":"float16"}}'
```

## 管理 UI (モデル DL 管理)

`/admin/ui` で管理 UI を開くと、次を操作できます。旧 URL の `/ui` と `/webui` は
`/admin/ui` へリダイレクト (307) します。

- **モデル管理**: カタログ (`MODEL_CATALOG_JSON`、既定は下記のテキストモデル) と
  保存領域のカタログ外モデルをファミリー別に一覧し、行クリックの詳細ドロワーで取得・ロード・解放を操作します
  (種別は `config.json` の architectures とディレクトリ名から推定)。
  `repo_id` 手入力の「リポジトリから取得」フォームと、取得進捗 (`%`, `MB/s`) もここにあります。
  カタログのモデルは `<最初の root>/<リポジトリ名>` (例: `embedding/ruri-v3-310m`) に保存し、
  その名前がそのまま API の `model` になります。旧 `?tab=catalog` / `?tab=downloads` はここを開きます。
- **Playground**: ロード済み embedding モデルで 2 テキストのベクトル・次元・cosine similarity を確認
- **Inference Logs**: 推論ログ (`embedding` / `responses_embedding` / `rerank` / `chat_completions`)
- **Server Health**: デバイス・GPU メモリ・ロード済みモデルと推論確認・起動設定 (読み取り専用)

ローカル一覧は `config.json` または `adapter_config.json` があるフォルダをモデルルートとして
表示し、`.cache` 配下などは出しません。ダウンロードは `server.py` 内で
`huggingface_hub.snapshot_download()` をバックグラウンドジョブとして実行します
(同時 3 件まで、超過は 429。ジョブ履歴はメモリ内のみで再起動で消えます)。

そのため、モデルを保存する `/models` は書き込み可能である必要があります (`:rw`)。
private / gated model を落とす場合は `HF_TOKEN` を設定してください。

モデルの一覧は環境変数 `ADMIN_MODEL_ROOTS` (`MODEL_DIR` からの相対パス、カンマ区切り、絶対パス可。
既定 `embedding`) の配下に限ります。対象は管理画面、`GET /admin/models`、公開 API の
`GET /v1/models` と `GET /models` のすべてです。別用途のモデル (`stt/` など) や HF cache は返しません。
名前を指定したロード (`/v1/models/load` など) の挙動は変わりません。

### 起動後に開く URL

```text
http://localhost:7997/admin/ui
```

### モデル DL API

- `POST /v1/models/download`
- `GET /v1/models/downloads`
- `GET /v1/models/downloads/{job_id}`
- `GET /v1/logs/inference?limit=30`
- `GET /v1/models/catalog`
- `POST /v1/responses` (OpenAI Responses API 互換サブセット)
- `POST /v1/chat/completions` (OpenAI 互換, upstream proxy)

`POST /v1/models/download` の例:

```bash
curl -X POST http://localhost:7997/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"Qwen/Qwen3-Embedding-4B","local_name":"embedding/Qwen3-Embedding-4B"}'
```

`repo_id` は必ず `owner/repo` 形式です。`Qwen` のような namespace のみ指定は失敗します。

`local_name` は `MODEL_DIR` からの相対パスです (区切りは `/`)。省略時は `repo_id` の末尾名のみの 1 段です。  
Embedding / Rerank / 将来の LLM 用など、用途ごとに `embedding/...` や `reranker/ruri-v3-reranker-310m` のように階層を分けられます。  
`..` や `MODEL_DIR` の外へ抜けるパスは拒否されます。

ロードや推論で参照する `model` / `model_id` は、上記と同じ相対パス文字列に揃えてください。

### `/v1/responses` 互換エンドポイント

`9router` などで `responses` エンドポイントしか選べない場合向けに、  
`POST /v1/responses` を追加しています。内部では embedding を実行し、  
互換フィールド (`output_text`) と拡張フィールド (`data`) を返します。

```bash
curl -X POST http://localhost:7997/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":"test from responses api"}'
```

返却 JSON の `data` に embedding ベクトルが入ります。

### `/v1/chat/completions` 互換エンドポイント

将来 `GGUF` モデルを `llama.cpp` / `Ollama` など OpenAI 互換 API 経由で使うために、  
`/v1/chat/completions` を upstream proxy として追加しています。

設定 (`.env`):

```text
LLM_PROXY_BASE_URL=http://localhost:11434/v1
LLM_PROXY_API_KEY=
LLM_PROXY_TIMEOUT_SEC=120
```

呼び出し例:

```bash
curl -X POST http://localhost:7997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5:7b",
    "messages":[{"role":"user","content":"hello"}]
  }'
```

### Embedding 動作確認のコツ

`curl` で `{"detail":"There was an error parsing the body"}` が出る場合、  
Windows / Git Bash の文字コード差分で JSON 本文が壊れている可能性があります。

まずは ASCII 文字列で確認してください。

```bash
curl -X POST http://localhost:7997/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":"test"}'
```

日本語を送る場合は UTF-8 の JSON ファイル経由が安全です。

```bash
cat > payload.json <<'EOF'
{"model":"Qwen3-Embedding-0.6B","input":"これはテストです"}
EOF
curl -X POST http://localhost:7997/v1/embeddings \
  -H "Content-Type: application/json; charset=utf-8" \
  --data-binary @payload.json
```

### `dimensions` (出力次元の切り詰め)

`dimensions` パラメータで出力ベクトルの次元数を指定できます。

```bash
curl -X POST http://localhost:7997/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":"test","dimensions":512}'
```

- 未指定時はモデルの標準次元数をそのまま返します
- 指定時は先頭 `dimensions` 要素へ切り詰めます
- `SentenceTransformer.encode()` が `truncate_dim` をサポートしている場合はそちらを使用し、未対応の場合はベクトルを Python でスライスします
- 指定可能範囲: `1 <= dimensions <= model.get_sentence_embedding_dimension()`
- 範囲外の場合は `400` エラー

> **品質保証の注意**: 切り詰め後も高品質なベクトルが得られるのは **Matryoshka Representation Learning (MRL)** 対応モデルに限ります (例: `Qwen3-Embedding-0.6B`, `Qwen3-Embedding-4B`)。
> 非 MRL モデルでは単純な先頭切り詰めになるため品質劣化が大きくなる場合があります。

`encoding_format: "base64"` との併用も可能です。

```bash
curl -X POST http://localhost:7997/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":"test","dimensions":256,"encoding_format":"base64"}'
```

### 管理 UI の構成

管理 UI は共通テンプレート (fastapi-admin-ui-template) 由来の admin shell (`src/admin_shell/`,
`src/static/admin/`) です。外部 CDN やフロントエンドのビルドは不要です。
`src/admin_shell/`、`src/embedding_admin.py`、`src/embedding_overview.py`、`src/model_profiles.py`、
`src/plamo_embedding.py`、`src/plamo_tokenizer_compat.py`、`src/voyage_compat.py`、`src/gte_compat.py`、`src/static/` を `server.py` と一緒に
配布してください (Dockerfile は対応済み)。

- `/admin/*` は `/v1/*` と同じく**認証なし**です。信頼できるネットワーク内でのみ公開してください。
- 管理 API: `GET /admin/models` (ローカル + ジョブ + ロード状態)、`GET /admin/models/overview` (モデル管理画面の read model)、`POST /admin/models/download`、
  `GET /admin/models/downloads[/{job_id}]`、`POST /admin/models/{local_name}/load|unload`、
  `GET /admin/interactions`、`GET /admin/health`、`GET /admin/events` (SSE)。
- 取得失敗・取得中のジョブがあるモデルは、部分ファイルの可能性があるため管理 UI からはロードできません (409)。
  ローカル検出や取得完了だけでは、重みの完全性・互換性・実行可能性を保証しません。
- `GET /admin/health` は `/health` に非機密の起動設定と、現ロードでの推論時刻を加えたものです
  (旧 `GET /ui/status` の内容)。アンロード・再起動で推論時刻は消えます。
- `/v1/*` と既存の別名エンドポイント・レスポンス形は維持しています。ただし取得ジョブのログは
  保持しなくなったため、`logs_count` は常に `0`、`last_log` は `null` です。

検証手順と実モデル未検証の範囲は [tests/README.md](tests/README.md) に記載しています。

## テキストモデルのプロファイル

既定カタログは 29 checkpoint (Embedding 25 / Reranker 4) です。
カタログの登録は、その環境での重み取得・GPU 推論成功を意味しません。
各行のリンクは公式モデルカードです。pooling と projection は checkpoint の公式 modules を
Sentence Transformers でロードし、PLaMo のみ公式 `encode_query` / `encode_document` を adapter から呼びます。

| モデル ID | 標準次元 | 推論時の設定 |
| --- | ---: | --- |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1024 | last token、query / document |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 2560 | last token、query / document |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 4096 | last token、query / document、16 GB ではメモリ不足に注意 |
| [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) | 768 | mean、prefix なし |
| [cl-nagoya/ruri-v3-130m](https://huggingface.co/cl-nagoya/ruri-v3-130m) | 512 | mean、prefix なし |
| [cl-nagoya/ruri-v3-70m](https://huggingface.co/cl-nagoya/ruri-v3-70m) | 384 | mean、prefix なし |
| [cl-nagoya/ruri-v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-30m) | 256 | mean、prefix なし |
| [pfnet/plamo-embedding-1b](https://huggingface.co/pfnet/plamo-embedding-1b) | 2048 | 公式 instruction masking + mean、remote code |
| [sbintuitions/sarashina-embedding-v2-1b](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b) | 1792 | last token、公式日本語 instruction / `text: `、非商用条件 |
| [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 1024 | last token、`web_search_query` / prefix なし |
| [microsoft/harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m) | 640 | last token、`web_search_query` / prefix なし |
| [ibm-granite/granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) | 768 | CLS、prefix なし |
| [ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | 384 | CLS、prefix なし |
| [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | 2048 | mean、公式 query / document prefix、remote code |
| [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | 1024 | mean、prefix なし、公式 int8 dense、remote code |
| [perplexity-ai/pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b) | 2560 | mean、prefix なし、公式 int8 dense、remote code |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | 2048 | mean、`query: ` / `passage: ` |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | 768 | mean + projection、テキストのみ、`SearchQuery` / `Document` |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 768 | 公式 modules、query / document、Gemma 規約への承諾と token が必要 |
| [jinaai/jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | 768 | `task=retrieval`、`Query: ` / `Document: `、remote code、CC BY-NC 4.0 |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 1024 | CLS、prefix なし、dense 出力のみ |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | 768 | CLS、prefix なし、dense 出力のみ、remote code |
| [hotchpotch/bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) | 384 | mean、prefix なし |
| [hotchpotch/bekko-embedding-v1-a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m) | 384 | mean、prefix なし |
| [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 1024 | mean、公式 retrieval instruction / prefix なし |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | — | CrossEncoder の公式 chat template + yes/no LogitScore |
| [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | — | CrossEncoder の公式 chat template + yes/no LogitScore |
| [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m) | — | CrossEncoder、sequence classification |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | — | CrossEncoder、sequence classification、Apache 2.0 |

### 検索クエリと文書の区別

`/v1/embeddings`、`/embeddings`、既存の Responses 互換経路 (`/v1/responses`、`/responses`) は
任意の `input_type: "query" | "document"` を受け取ります。省略時は従来どおり `query` です。
モデルに応じた prefix はサーバーが付けるため、`input` には素のテキストを渡してください。
検索対象を登録するときは `document`、検索するときは `query` を使い、同じモデル・次元を揃えます。

```bash
curl -X POST http://localhost:7997/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embedding/embeddinggemma-2","input":"Tokyo is the capital of Japan.","input_type":"document","dimensions":256}'
```

`model` はロード・取得時と同じローカル名です。標準の `<root>/<repo 末尾名>` または HF の正式 ID から
プロファイルを識別します。別名保存する場合は `MODEL_CATALOG_JSON` の該当行に `local_name` を明示してください。
既知モデルのローカルファイルと無関係な HF owner を、末尾名の一致だけで信頼することはありません。
カタログ外モデルは従来の汎用 SentenceTransformer / CrossEncoder 経路へ渡します。

### 依存関係と精度設定

`src/requirements.base.txt` は安定版 `sentence-transformers>=6.1.0`、`transformers>=5.19.0` を要求します。
EG2 の [公式 checkpoint 設定](https://huggingface.co/google/embeddinggemma-2/blob/main/config_sentence_transformers.json)
は ST 6.1.0 を要求し、Transformers の [v5.19.0 の native 実装](https://github.com/huggingface/transformers/tree/v5.19.0/src/transformers/models/embedding_gemma2)
を使います。Qwen3-Reranker は最新公式 checkpoint の `LogitScore` modules により、既存 CrossEncoder 経路で推論します。
以前取得した Qwen3-Reranker に `modules.json` と `1_LogitScore/` がない場合は、公式 checkpoint を別名で再取得してください。
PLaMo の tokenizer 用に `sentencepiece>=0.2.2`、Jina Text の公式 task adapter 用に `peft>=0.15.2` も導入します。
PLaMo tokenizer の旧 `Init` 呼び出しは、モデル内に限定した adapter で現行 SentencePiece の
公開 constructor に変換し、語彙・thread 設定を維持します。
TF 5 が除く `max_length` は公式の `max_position_embeddings` から復元し、checkpoint に保存されない
RoPE buffer を [公式推論コード](https://huggingface.co/pfnet/plamo-embedding-1b/blob/main/modeling_plamo.py) の式で再生成します。
Jina Text の公式要件に合わせ、PyTorch は `torch>=2.8.0` を要求します。
Voyage の公式 remote code では mask 関数の旧引数名を使うため、Voyage module 内に限定した
互換 adapter で Transformers 5.19 の引数名へ変換します。公式 bidirectional mask と projection は維持します。
GTE は Transformers 5 の meta load 後に、公式実装が checkpoint に保存しない位置・RoPE buffer を
公式の `arange` / `_init_rope` で再生成し、旧 encoder の padding mask を instance 内に補完します。
学習済みパラメータは変更しません。
Jina Text の公式 ST module が標準次元を報告しない場合は、該当プロファイルの 768 次元を
`dimensions` 検証に使います。実出力が要求次元と違う場合は 500 で拒否します。

EG2 は `config_kwargs={"vision_config": null, "audio_config": null}` で未使用 encoder を省きます。
EmbeddingGemma 2 と初代は、GPU が対応すれば BF16、その他では FP32 を既定にし、FP16 設定を 400 で拒否します。
全ベクトルは短縮・量子化・BF16 の計算後に API 境界で L2 正規化します。
推論結果が NaN / Inf の場合は成功扱いせず 500 を返します。
それ以外のモデルで GPU メモリを節約する場合は、対応するモデルに合わせて次を指定できます。

```bash
export SENTENCE_TRANSFORMER_KWARGS='{"model_kwargs":{"dtype":"bfloat16"}}'
export CROSS_ENCODER_KWARGS='{"model_kwargs":{"dtype":"bfloat16"}}'
```

PLaMo、Voyage、pplx、Jina Text、GTE の公式実装に必要な `trust_remote_code=True` はプロファイル単位です。
環境変数の明示設定はモデルごとの既定値を上書きします。
pplx は公式 `FlexibleQuantizer` の int8 密ベクトルを L2 正規化し、既存の float / base64 float32 応答として返します。
未量子化 float ベクトルや binary packed 出力を公開する API ではありません。

### 今回追加しない出力形式

- [pplx Context](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b) は文書ごとのチャンク列を入力し、チャンクごとのベクトルを返します。現在の `str | list[str]` の入力と 1 テキスト 1 ベクトルの契約とは異なるため対象外です。
- [Jina v5 Omni Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano) / [Small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) の media 入力、EG2 の画像・音声・動画入力、BGE-M3 / GTE の sparse / multi-vector 出力は公開しません。
- [EG2 Decision / MediaPipe Decision Maker](https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker/index) は Boolean / Choice / Score と別の LiteRT アセットを使う機能です。Embedding / Rerank API の独立 checkpoint として登録しません。

ライセンス制約は各公式モデルカードに従ってください。gated repo の承諾は管理 UI から自動で行いません。
GPU の実測範囲と再実行コマンドは [tests/README.md](tests/README.md) を参照してください。
