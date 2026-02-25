from huggingface_hub import snapshot_download

# “local_dir” で直下に展開
snapshot_download(
    repo_id="cl-nagoya/ruri-v3-reranker-310m",
    local_dir="/mnt/ai-models/embedding/ruri-v3-reranker-310m",
    local_dir_use_symlinks=False,
    resume_download=True
)
