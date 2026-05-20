import logging
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from coderag.config import TOP_K
from coderag.embeddings import generate_embeddings
from coderag.index import get_metadata, load_index

logger = logging.getLogger(__name__)


def search_code(
    query: str,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index using a natural-language query.
    """

    try:
        if not query or not query.strip():
            logger.warning("Empty query received for search.")
            return []

        # Load FAISS index
        index = load_index()

        if index is None or index.ntotal == 0:
            logger.warning(
                "FAISS index is empty or not loaded. " "Please re-index the project."
            )

            return []

        metadata = get_metadata()

        if not metadata:
            logger.warning(
                "Metadata is empty. " "Ensure files were indexed successfully."
            )

            return []

        # Generate embedding
        query_embedding: Optional[np.ndarray] = generate_embeddings(query)

        if query_embedding is None or query_embedding.size == 0:
            logger.error("Failed to generate embedding for query.")

            return []

        # Normalize embedding
        faiss.normalize_L2(query_embedding)

        # Limit k
        k = k or TOP_K
        k = min(k, index.ntotal)

        # Search
        distances, indices = index.search(
            query_embedding,
            k,
        )

        # Format results
        results: List[Dict[str, Any]] = []

        for rank, idx in enumerate(indices[0]):

            if 0 <= idx < len(metadata):

                entry = metadata[idx]
                score = float(distances[0][rank])

                results.append(
                    {
                        "rank": rank + 1,
                        "filename": entry.get(
                            "filename",
                            "Unknown",
                        ),
                        "filepath": entry.get(
                            "filepath",
                            "",
                        ),
                        "content": entry.get(
                            "content",
                            "",
                        )[:600],
                        "distance": round(score, 4),
                    }
                )

            else:
                logger.warning(
                    f"Invalid metadata index {idx} " f"(length={len(metadata)})."
                )

        # Sort by similarity
        results.sort(
            key=lambda r: r["distance"],
            reverse=True,
        )

        logger.info(
            "Search completed for query "
            f"'{query[:60]}...' → "
            f"{len(results)} results found."
        )

        return results

    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")

        return []
