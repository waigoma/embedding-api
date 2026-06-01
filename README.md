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

## WebUI (モデル DL 管理)

`/ui` で WebUI を開くと、次を操作できます。

- `Hugging Face` からモデルをダウンロード開始 (`repo_id` は手入力)
- ダウンロード進捗 (`0-100%`, `MB/s`, `downloaded/total size`) の確認
- ローカルモデル一覧の確認と `Load` / (ロード済みのみ) `Unload`（一覧は `config.json` または `adapter_config.json` があるフォルダをモデルルートとして表示し、`.cache` 配下などは出しません）
- ロード済みモデルのトップ表示
- 推論ログ表示 (`embedding` / `rerank` 実行時)

`/v1/models/download` は `download_from_huggingface.py` を直接実行する方式ではなく、  
`server.py` 内の `huggingface_hub.snapshot_download()` をバックグラウンドジョブで実行します。

そのため、モデルを保存する `/models` は書き込み可能である必要があります (`:rw`)。
private / gated model を落とす場合は `HF_TOKEN` を設定してください。

### 起動後に開く URL

```text
http://localhost:7997/ui
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
