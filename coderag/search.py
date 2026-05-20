import logging
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from coderag.embeddings import generate_embeddings
from coderag.index import get_metadata, load_index
from coderag.config import TOP_K

logger = logging.getLogger(__name__)

#  Search Function
def search_code(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search the FAISS index using a natural-language query.

    Args:
        query (str): The search query text.
        k (int, optional): Number of top results to return. Defaults to TOP_K from config.

    Returns:
        List[Dict[str, Any]]: Results containing filename, path, content, and similarity score.
    """
    try:
        if not query or not query.strip():
            logger.warning(" Empty query received for search.")
            return []

        # Load FAISS index
        index = load_index()
        if index is None or index.ntotal == 0:
            logger.warning(" FAISS index is empty or not loaded. Please re-index the project.")
            return []

        metadata = get_metadata()
        if not metadata:
            logger.warning(" Metadata is empty. Ensure files were indexed successfully.")
            return []

        # === Step 1: Generate query embedding ===
        query_embedding: Optional[np.ndarray] = generate_embeddings(query)
        if query_embedding is None or query_embedding.size == 0:
            logger.error(" Failed to generate embedding for query.")
            return []

        # === Step 2: Normalize for cosine similarity ===
        faiss.normalize_L2(query_embedding)

        # === Step 3: Limit k ===
        k = k or TOP_K
        k = min(k, index.ntotal)

        # === Step 4: Perform FAISS search ===
        distances, indices = index.search(query_embedding, k)

        # === Step 5: Format results ===
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(metadata):
                entry = metadata[idx]
                score = float(distances[0][rank])
                results.append(
                    {
                        "rank": rank + 1,
                        "filename": entry.get("filename", "Unknown"),
                        "filepath": entry.get("filepath", ""),
                        "content": entry.get("content", "")[:600],
                        "distance": round(score, 4),
                    }
                )
            else:
                logger.warning(f" Invalid metadata index {idx} (length={len(metadata)}).")

        # Sort descending by similarity score
        results.sort(key=lambda r: r["distance"], reverse=True)

        logger.info(
            f" Search completed for query '{query[:60]}...' → {len(results)} results found."
        )
        return results

    except Exception as e:
        logger.error(f" Error during FAISS search: {e}")
        return []
