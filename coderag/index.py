import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from coderag.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_FILE,
    WATCHED_DIR,
)

logger = logging.getLogger(__name__)

# Global FAISS index and metadata
index: Optional[faiss.Index] = faiss.IndexFlatIP(EMBEDDING_DIM)

metadata: List[Dict[str, Any]] = []


def initialize_index(
    dim: int = EMBEDDING_DIM,
) -> None:
    """
    Initialize a new FAISS index.
    """

    try:
        globals()["index"] = faiss.IndexFlatIP(dim)

        logger.info(f"FAISS index initialized with " f"dimension: {dim}")

    except Exception as e:
        logger.error(f"Failed to initialize FAISS index: {e}")
        raise


# Initialize once on import
initialize_index()


def _l2_normalize(
    mat: np.ndarray,
) -> np.ndarray:
    """
    Normalize vectors to unit length.
    """

    if mat is None or mat.size == 0:
        return mat

    faiss.normalize_L2(mat)

    return mat


def clear_index() -> None:
    """
    Delete FAISS index and metadata files,
    then reinitialize.
    """

    try:
        if os.path.exists(FAISS_INDEX_FILE):

            os.remove(FAISS_INDEX_FILE)

            logger.info(f"Deleted FAISS index file: " f"{FAISS_INDEX_FILE}")

        meta_file = "metadata.npy"

        if os.path.exists(meta_file):

            os.remove(meta_file)

            logger.info(f"Deleted metadata file: " f"{meta_file}")

        initialize_index(EMBEDDING_DIM)

        globals()["metadata"] = []

        logger.info(f"Reinitialized FAISS index " f"(dim={EMBEDDING_DIM})")

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise


def add_to_index(
    embeddings: np.ndarray,
    full_content: str,
    filename: str,
    filepath: str,
) -> None:
    """
    Add embeddings and metadata to FAISS index.
    """

    try:
        if embeddings is None or embeddings.size == 0:

            logger.warning(f"Empty embeddings for file: " f"{filename}")

            return

        if embeddings.shape[1] != index.d:

            logger.warning(
                "Dimension mismatch: "
                f"embedding={embeddings.shape[1]}, "
                f"index={index.d}. "
                "Reinitializing FAISS index..."
            )

            initialize_index(embeddings.shape[1])

        # Normalize embeddings
        vecs = embeddings.astype(
            "float32",
            copy=True,
        )

        vecs = _l2_normalize(vecs)

        # Add to FAISS
        index.add(vecs)

        # Relative path handling
        try:
            relative_path = os.path.relpath(
                filepath,
                WATCHED_DIR,
            )

        except ValueError:
            relative_path = filepath

        # Store metadata
        metadata.append(
            {
                "filename": filename,
                "filepath": relative_path,
                "content": (full_content[:3000] if full_content else ""),
            }
        )

        logger.debug(f"Added {filename} " f"(total: {index.ntotal} entries)")

    except Exception as e:
        logger.error(f"Failed to add {filename} " f"to FAISS index: {e}")

        raise


def save_index() -> None:
    """
    Persist FAISS index and metadata.
    """

    try:
        faiss.write_index(
            index,
            FAISS_INDEX_FILE,
        )

        np.save(
            "metadata.npy",
            np.array(
                metadata,
                dtype=object,
            ),
        )

        logger.info(
            f"Saved FAISS index " f"({index.ntotal} entries, " f"dim={index.d})"
        )

    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")

        raise


def load_index() -> Optional[faiss.Index]:
    """
    Load FAISS index and metadata.
    """

    try:
        if not os.path.exists(FAISS_INDEX_FILE):

            logger.warning(f"FAISS index not found: " f"{FAISS_INDEX_FILE}")

            return None

        if not os.path.exists("metadata.npy"):

            logger.warning("Metadata file missing: " "metadata.npy")

            return None

        globals()["index"] = faiss.read_index(FAISS_INDEX_FILE)

        globals()["metadata"] = np.load(
            "metadata.npy",
            allow_pickle=True,
        ).tolist()

        logger.info("Loaded FAISS index successfully " f"from {FAISS_INDEX_FILE}")

        return globals()["index"]

    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")

        return None


def get_metadata() -> List[Dict[str, Any]]:
    """
    Return metadata list.
    """

    return metadata


def retrieve_vectors(
    n: int = 5,
) -> np.ndarray:
    """
    Retrieve vectors for inspection.
    """

    n = min(n, index.ntotal)

    vecs = np.zeros(
        (n, index.d),
        dtype=np.float32,
    )

    for i in range(n):
        vecs[i] = index.reconstruct(i)

    logger.info(f"Retrieved {n} vectors " f"for inspection.")

    return vecs


def inspect_metadata(
    n: int = 5,
) -> None:
    """
    Print metadata entries.
    """

    try:
        for i, data in enumerate(metadata[:n]):

            logger.info(f"\nEntry {i + 1}:")

            logger.info(f"Filename: " f"{data['filename']}")

            logger.info(f"Path: " f"{data['filepath']}")

            logger.info(f"Snippet: " f"{data['content'][:120]}...")

    except Exception as e:
        logger.error(f"Error inspecting metadata: {e}")
