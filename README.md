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

- カタログから `repo_id` を選択
- `Hugging Face` からモデルをダウンロード開始
- ダウンロードジョブ状態の確認
- ローカルモデル一覧の確認

### 起動後に開く URL

```text
http://localhost:7997/ui
```

### モデル DL API

- `POST /v1/models/download`
- `GET /v1/models/downloads`
- `GET /v1/models/downloads/{job_id}`
- `GET /v1/models/catalog`

`POST /v1/models/download` の例:

```bash
curl -X POST http://localhost:7997/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"Qwen/Qwen3-Embedding-4B","local_name":"Qwen3-Embedding-4B"}'
```
