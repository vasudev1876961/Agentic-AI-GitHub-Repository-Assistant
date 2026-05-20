import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from coderag.config import EMBEDDING_DIM, FAISS_INDEX_FILE, WATCHED_DIR

logger = logging.getLogger(__name__)

#  Initialize FAISS index and metadata
index: Optional[faiss.Index] = None
metadata: List[Dict[str, Any]] = []


def initialize_index(dim: int = EMBEDDING_DIM) -> None:
    """Initialize a new FAISS index with the given embedding dimension."""
    global index, metadata
    try:
        index = faiss.IndexFlatIP(dim)
        metadata = []
        logger.info(f" FAISS index initialized with dimension: {dim}")
    except Exception as e:
        logger.error(f" Failed to initialize FAISS index: {e}")
        raise


# Initialize once on import
initialize_index()


#  Utility Functions
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length in-place."""
    if mat is None or mat.size == 0:
        return mat
    faiss.normalize_L2(mat)
    return mat


#  Clear / Rebuild Index
def clear_index() -> None:
    """Delete FAISS and metadata files, then reinitialize."""
    global index, metadata

    try:
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
            logger.info(f" Deleted FAISS index file: {FAISS_INDEX_FILE}")

        meta_file = "metadata.npy"
        if os.path.exists(meta_file):
            os.remove(meta_file)
            logger.info(f" Deleted metadata file: {meta_file}")

        initialize_index(EMBEDDING_DIM)
        logger.info(f" Reinitialized FAISS index (dim={EMBEDDING_DIM})")

    except Exception as e:
        logger.error(f" Error clearing index: {e}")
        raise


#  Add Embeddings to Index
def add_to_index(
    embeddings: np.ndarray, full_content: str, filename: str, filepath: str
) -> None:
    """Add a file’s embedding and metadata to the FAISS index."""
    global index, metadata

    try:
        if embeddings is None or embeddings.size == 0:
            logger.warning(f" Empty embeddings for file: {filename}")
            return

        if embeddings.shape[1] != index.d:
            logger.warning(
                f" Dimension mismatch: embedding={embeddings.shape[1]}, index={index.d}. "
                f"Reinitializing FAISS index..."
            )
            initialize_index(embeddings.shape[1])

        # Normalize for cosine similarity
        vecs = embeddings.astype("float32", copy=True)
        vecs = _l2_normalize(vecs)

        # Add to FAISS
        index.add(vecs)

        # Handle relative paths
        try:
            relative_path = os.path.relpath(filepath, WATCHED_DIR)
        except ValueError:
            relative_path = filepath

        # Save metadata
        metadata.append(
            {
                "filename": filename,
                "filepath": relative_path,
                "content": full_content[:3000] if full_content else "",
            }
        )

        logger.debug(f"📦 Added {filename} (total: {index.ntotal} entries)")

    except Exception as e:
        logger.error(f" Failed to add {filename} to FAISS index: {e}")
        raise


#  Save / Load Index
def save_index() -> None:
    """Persist FAISS index and metadata to disk."""
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
        np.save("metadata.npy", np.array(metadata, dtype=object))
        logger.info(f" Saved FAISS index ({index.ntotal} entries, dim={index.d})")
    except Exception as e:
        logger.error(f" Error saving FAISS index: {e}")
        raise


def load_index() -> Optional[faiss.Index]:
    """Load FAISS index and metadata from disk."""
    global index, metadata
    try:
        if not os.path.exists(FAISS_INDEX_FILE):
            logger.warning(f" FAISS index not found: {FAISS_INDEX_FILE}")
            return None

        if not os.path.exists("metadata.npy"):
            logger.warning(" Metadata file missing: metadata.npy")
            return None

        index = faiss.read_index(FAISS_INDEX_FILE)
        metadata = np.load("metadata.npy", allow_pickle=True).tolist()
        logger.info(f" Loaded FAISS index ({index.ntotal} entries, dim={index.d})")
        return index

    except Exception as e:
        logger.error(f" Failed to load FAISS index: {e}")
        return None


#  Inspection / Debugging Tools
def get_metadata() -> List[Dict[str, Any]]:
    """Return the full metadata list."""
    return metadata


def retrieve_vectors(n: int = 5) -> np.ndarray:
    """Retrieve the first n vectors for inspection."""
    n = min(n, index.ntotal)
    vecs = np.zeros((n, index.d), dtype=np.float32)
    for i in range(n):
        vecs[i] = index.reconstruct(i)
    logger.info(f" Retrieved {n} vectors for inspection.")
    return vecs


def inspect_metadata(n: int = 5) -> None:
    """Print first few metadata entries (for debugging)."""
    try:
        for i, data in enumerate(metadata[:n]):
            logger.info(f"\n📁 Entry {i + 1}:")
            logger.info(f"  Filename: {data['filename']}")
            logger.info(f"  Path: {data['filepath']}")
            logger.info(f"  Snippet: {data['content'][:120]}...")
    except Exception as e:
        logger.error(f" Error inspecting metadata: {e}")
