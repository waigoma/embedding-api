# systemd による native デプロイ

`deploy/systemd/embedding-api.service` を標準の起動定義として使用する。
配置例は共通のシステム領域だけを使い、ホストの実行ユーザーやモデル保存先は環境ファイルと drop-in で指定する。
ソース、venv、モデル、キャッシュを別々に管理する。

## 前提と配置

Linux の systemd、Git、Python 3.12、uv が必要になる。
以下の依存導入例は NVIDIA CUDA 13.0 用で、対応する NVIDIA ドライバをホスト側に用意する。
CPU / AMD 向けの固定依存ファイルは提供していないため、この CUDA 用 lock を流用せず、対象 backend の PyTorch と `src/requirements.base.txt` から別環境を作成する。

| 用途 | 標準の配置例 |
| --- | --- |
| ソース | `/opt/embedding-api` |
| venv | `/opt/embedding-api/venv` |
| 実行ユーザーと HOME | `embedding-api` / `/var/lib/embedding-api` |
| モデル | `/var/lib/embedding-api/models` |
| HF cache | `/var/cache/embedding-api/huggingface` |
| 環境ファイル | `/etc/default/embedding-api` |

## ソースと依存の準備

以下は uv を実行できる root shell での初回導入例である。
`<repository-url>` と `<commit-sha>` を取得先と 40 桁の commit SHA に置き換える。
稼働中のソースや venv を上書きする更新には使わない。

```sh
git clone '<repository-url>' /opt/embedding-api
git -C /opt/embedding-api checkout --detach '<commit-sha>'
uv venv --python /usr/bin/python3.12 /opt/embedding-api/venv
uv pip install --python /opt/embedding-api/venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python /opt/embedding-api/venv/bin/python \
  -r /opt/embedding-api/deploy/native/requirements-cu130.lock
uv pip check --python /opt/embedding-api/venv/bin/python
```

固定依存は `deploy/native/requirements-cu130.lock` が管理する。
PyTorch、torchvision、torchaudio は上記の index から先に導入する。
起動時に依存をインストールする処理はない。

## 実行ユーザーと環境設定

```sh
useradd --system --user-group --home-dir /var/lib/embedding-api \
  --create-home --shell /usr/sbin/nologin embedding-api
install -d -o embedding-api -g embedding-api \
  /var/lib/embedding-api/models /var/cache/embedding-api/huggingface
install -m 0600 -o root -g root \
  /opt/embedding-api/deploy/systemd/embedding-api.env.example \
  /etc/default/embedding-api
```

既存ユーザーを利用する場合は `useradd` を省き、次節の drop-in で `User` と `Group` を変更する。
モデルとキャッシュは実行ユーザーが読み書きできる領域を指定する。
既存の共有領域の所有権を再帰的に変更する必要はない。

`/etc/default/embedding-api` をローカルで編集し、`MODEL_DIR`、`HF_HOME`、`DEVICE_MODE` を対象サーバーに合わせる。
標準ポートは `7997`、自動ロードは `AUTO_LOAD=1`、アイドル解放は `IDLE_TTL=6000` である。
private / gated model に必要な `HF_TOKEN` などの秘密は、このファイルだけに記載する。

## Unit の配置と起動

```sh
install -m 0644 -o root -g root \
  /opt/embedding-api/deploy/systemd/embedding-api.service \
  /etc/systemd/system/embedding-api.service
systemd-analyze verify /etc/systemd/system/embedding-api.service
systemctl daemon-reload
systemctl enable --now embedding-api
curl --fail http://127.0.0.1:7997/health
```

unit は `/usr/bin/env` を経由して `${VIRTUAL_ENV}/bin/uvicorn server:app` を実行する。
systemd が `VIRTUAL_ENV`、`HOST`、`PORT` を引数へ展開し、`env` が同じプロセスで uvicorn を実行する。
shell や venv の activate は不要である。
異常終了時は 5 秒後に再起動し、ログは `journalctl -u embedding-api` で確認できる。

## 配置先と unit の上書き

環境変数は `/etc/default/embedding-api`、systemd の設定は `/etc/systemd/system/embedding-api.service.d/*.conf` で上書きする。
`User`、`WorkingDirectory`、`RequiresMountsFor` は環境ファイルだけでは変更できない。
例えば、版別のソースと venv を使う場合は次の drop-in を追加する。
パス中の `VERSION` は選択した版へ置き換える。

```ini
[Unit]
RequiresMountsFor=
RequiresMountsFor=/srv/models /srv/model-cache

[Service]
User=embedding-api
Group=embedding-api
WorkingDirectory=/opt/embedding-api/releases/VERSION/src
EnvironmentFile=
EnvironmentFile=/etc/default/embedding-api
```

環境ファイル側には `VIRTUAL_ENV=/opt/embedding-api/venvs/VERSION`、`MODEL_DIR=/srv/models`、`HF_HOME=/srv/model-cache` を指定する。
`EnvironmentFile` の値は unit の `Environment` より優先されるため、同じ変数を drop-in の `Environment` だけで変更しても環境ファイルの値は置き換わらない。
変更後は `systemctl daemon-reload` と `systemctl restart embedding-api` を実行する。

## 更新と切り戻し

新しい SHA のソースと venv を別ディレクトリに準備し、依存の整合を確認してから設定を切り替える。
unit も選択したソース版の定義で更新する。
以前のソースと venv、環境設定を保持しておけば、設定を戻して再起動できる。
モデルとキャッシュは版別ソースの外に維持する。

Ansible などの構成管理では、アプリの unit をそのまま配置し、ホスト設定を drop-in から渡す。
起動モジュール名と再起動ポリシーの標準定義はこのリポジトリが管理する。
