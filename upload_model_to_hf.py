# One-off utility to push the trained model to a plain (free) Hugging
# Face Hub model repo - not a Space, so no PRO subscription needed. Run
# manually after `huggingface-cli login` (never hardcode a token here).
#
# Usage: python upload_model_to_hf.py <your-hf-username>/garda-setfit
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from configs import paths

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / paths.SETFIT_MODEL_PATH

def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_model_to_hf.py <username>/<model-name>")
        sys.exit(1)

    repo_id = sys.argv[1]
    api = HfApi()

    create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(MODEL_PATH), repo_id=repo_id, repo_type="model")

    print(f"Done: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
