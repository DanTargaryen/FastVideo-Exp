import os

from huggingface_hub import login, snapshot_download

repo_id = "YuryyyLee/world-renderer-controlnet-union"
save_dir = "/vePFS-buaa/linming/workspace/worldrender/world-renderer-controlnet-union"

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

path = snapshot_download(
    repo_id=repo_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,  # 不用软链接，方便搬运
    resume_download=True,          # 断点续传
)

print("✅ Downloaded to:", path)
