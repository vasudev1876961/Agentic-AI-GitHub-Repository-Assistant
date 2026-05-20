import logging
from typing import List, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer 

logger = logging.getLogger(__name__)

#  Initialize Local Embedding Model
try:
    MODEL_NAME = "all-MiniLM-L6-v2" 
    logger.info(f"🔍 Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(" Local SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error(f" Failed to load SentenceTransformer model: {e}")
    model = None


#  Utility — Text Chunking
def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    """Split long text into manageable chunks for embedding."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


#  Embedding Generation (Batch)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def _embed_batch(inputs: List[str]) -> np.ndarray:
    """Generate embeddings locally using SentenceTransformers."""
    if not model:
        raise RuntimeError("Embedding model not initialized.")
    try:
        embeddings = model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")
    except Exception as e:
        logger.error(f" Error during local embedding generation: {e}")
        raise


#  Public Function — Generate Embeddings
def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """Generate averaged embeddings for text input."""
    if not text or not text.strip():
        logger.warning(" Empty text provided for embedding generation.")
        return None

    if not model:
        logger.error(" Embedding model unavailable.")
        return None

    try:
        chunks = _chunk_text(text)
        chunk_embeddings = [_embed_batch([chunk]) for chunk in chunks]
        avg_vec = np.mean(np.vstack(chunk_embeddings), axis=0, dtype="float32").reshape(1, -1)
        logger.debug(f" Generated embedding with shape: {avg_vec.shape}")
        return avg_vec

    except Exception as e:
        logger.error(f" Failed to generate embeddings: {e}")
        return None


#  Test Code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_text = "This is a simple test sentence to generate embeddings locally."

    try:
        embedding = generate_embeddings(test_text)
        if embedding is not None:
            print(" Embedding generated successfully!")
            print(f"🔹 Shape: {embedding.shape}")
            print(f"🔹 First 10 values: {embedding[0][:10]}")
        else:
            print(" No embedding returned.")
    except Exception as e:
        print(f" Error during embedding test: {e}")
