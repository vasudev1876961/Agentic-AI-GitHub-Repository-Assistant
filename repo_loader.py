import os
import re
import shutil
from typing import Tuple
from git import Repo
from coderag.config import REPOS_DIR

def _sanitize_repo_name(url: str) -> str:
    base = url.rstrip("/").split("/")[-2:]
    name = "_".join(base).replace(".git", "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def clone_repo(github_url: str, clean_if_exists: bool = True) -> Tuple[str, str]:
    """
    Clone a GitHub repo into REPOS_DIR. Returns (repo_dir, repo_name).
    If folder exists and clean_if_exists=True, it deletes and reclones.
    """
    os.makedirs(REPOS_DIR, exist_ok=True)
    repo_name = _sanitize_repo_name(github_url)
    repo_dir = os.path.join(REPOS_DIR, repo_name)

    if os.path.isdir(repo_dir):
        if clean_if_exists:
            shutil.rmtree(repo_dir)
        else:
            return repo_dir, repo_name

    Repo.clone_from(github_url, repo_dir)
    return repo_dir, repo_name
