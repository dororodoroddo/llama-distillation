# utils/download_model.py
import os
from huggingface_hub import snapshot_download

def ensure_model_exists(model_id: str, local_dir: str = "models"):
    local_path = os.path.join(local_dir, model_id.replace("/", "_"))
    if os.path.exists(local_path):
        print(f"[✓] 모델 이미 존재: {local_path}")
        return local_path

    print(f"[↓] 모델 다운로드 시작: {model_id} → {local_path}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"[✓] 모델 다운로드 완료: {local_path}")
    return local_path
