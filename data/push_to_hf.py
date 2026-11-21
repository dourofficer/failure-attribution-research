#!/usr/bin/env python3
"""Upload data/ to the private HF dataset repo. If it gets interrupted, run it again to resume.

    python3 data/push_to_hf.py
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "nguyenthanhdo/failure-attribution-data"
HERE = Path(__file__).resolve().parent

api = HfApi()  # uses the token from `hf auth login`

# LFS rules first, so the Hub stores the same files in LFS as the local git repo does.
api.upload_file(path_or_fileobj=HERE / ".gitattributes", path_in_repo=".gitattributes",
                repo_id=REPO, repo_type="dataset")

# Everything else. .git/ and .cache/huggingface/ are skipped automatically.
api.upload_folder(folder_path=HERE, repo_id=REPO, repo_type="dataset")

print("Done. To line data/.git up with the Hub: git -C data fetch origin && git -C data reset origin/main")
