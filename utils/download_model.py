from huggingface_hub import snapshot_download
from pathlib import Path
import os

def ensure_model_exists(repo_id: str, local_dir: str = "./models"):
    target_path = Path(local_dir) / repo_id.replace("/", "_")
    if target_path.exists():
        print(f"[✓] 모델 존재함: {target_path}")
        return str(target_path)

    print(f"[↓] 모델 다운로드 시작: {repo_id} → {target_path}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_path,
        local_dir_use_symlinks=False
    )
    return str(target_path)
