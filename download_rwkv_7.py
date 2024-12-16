from huggingface_hub import hf_hub_download
import os


def download_file(repo_id, filename, local_dir="./downloads"):
    os.makedirs(local_dir, exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    return path


for model in [
    "RWKV-x070-Pile-168M-20241120-ctx4096.pth", 
    "RWKV-x070-Pile-421M-20241127-ctx4096.pth", 
    "RWKV-x070-Pile-1.47B-20241210-ctx4096.pth",
    ]:
    download_file(
        repo_id="BlinkDL/rwkv-7-pile",
        filename=model,
        local_dir="rwkv_model"
    )