# ingestor.py
import logging
import os
from typing import Iterable, Tuple

from coderag.config import ALLOWED_EXTENSIONS, IGNORE_PATHS
from coderag.embeddings import generate_embeddings
from coderag.index import add_to_index, clear_index, save_index

logger = logging.getLogger(__name__)


def _should_ignore(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return any(skip in p for skip in (s.lower() for s in IGNORE_PATHS))


def _iter_files(root: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (abs_path, rel_path) for files we want to index.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if _should_ignore(dirpath):
            dirnames[:] = [
                d for d in dirnames if not _should_ignore(os.path.join(dirpath, d))
            ]
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                abs_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(abs_path, root)
                yield abs_path, rel_path


def index_directory(root: str) -> int:
    """
    Clear FAISS + metadata, then index all allowed files in root.
    Returns number of files successfully indexed.
    """
    clear_index()
    count = 0
    for abs_path, rel_path in _iter_files(root):
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            vec = generate_embeddings(content)
            if vec is None:
                logger.warning(f"Skipping (no embedding): {rel_path}")
                continue
            add_to_index(vec, content, os.path.basename(rel_path), abs_path)
            count += 1
        except Exception as e:
            logger.error(f"Failed to index {rel_path}: {e}")
    save_index()
    return count
