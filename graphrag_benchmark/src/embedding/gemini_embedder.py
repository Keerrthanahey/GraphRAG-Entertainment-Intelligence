from typing import Optional
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiEmbedder:
    """
    Local embedding model using sentence-transformers
    """

    def __init__(self):

        logger.info("Loading local embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.total_tokens = 0
        self.total_api_calls = 0
        self.total_cost = 0.0

        logger.info("Embedding model loaded successfully")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text
        """

        if not text:
            return []

        embedding = self.model.encode(text)

        self.total_tokens += len(text.split())

        return embedding.tolist()

    def embed_chunks(self, chunks, show_progress=True):
        """
        Embed list of chunks
        """

        total = len(chunks)

        logger.info(f"Embedding {total} chunks...")

        for idx, chunk in enumerate(chunks):

            embedding = self.embed_text(chunk.content)

            chunk.embedding = embedding

            if show_progress and idx % 100 == 0:
                logger.info(f"Progress: {idx}/{total}")

        logger.info("Chunk embedding completed")

        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Embed search query
        """

        return self.embed_text(query)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return embedding statistics
        """

        return {
            "total_tokens": self.total_tokens,
            "total_api_calls": 0,
            "estimated_cost_usd": 0,
            "model": "all-MiniLM-L6-v2",
        }