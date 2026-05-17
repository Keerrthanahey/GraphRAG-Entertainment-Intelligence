"""
GraphRAG Pipeline
"""

import time
from typing import Dict, Any, Optional

from src.utils.langchain_helper import run_chain
from src.pipelines.base_pipeline import BasePipeline, PipelineResult
from src.embedding.gemini_embedder import GeminiEmbedder
from src.storage.chroma_store import ChromaVectorStore
from src.utils.metrics import PerformanceMetrics, RetrievalMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphRAGPipeline(BasePipeline):

    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        store: Optional[ChromaVectorStore] = None,
        top_k: int = 5,
        traversal_depth: int = 2,
        max_nodes: int = 20,
    ):
        super().__init__("GraphRAG")

        self.embedder = embedder
        self.store = store

        self.top_k = top_k
        self.traversal_depth = traversal_depth
        self.max_nodes = max_nodes

        # Lightweight graph — only populated on demand, not at init
        self._local_graph = {}
        self._relationships = []

        logger.info("GraphRAG initialized")

    # ------------------------------------------------------------------
    # INITIALIZE — no graph build at startup (too expensive for 3k docs)
    # ------------------------------------------------------------------

    def initialize(self):

        if not self.embedder:
            self.embedder = GeminiEmbedder()

        if not self.store:
            self.store = (
                ChromaVectorStore()
                .connect("persistent")
                .create_collection()
            )

        self._is_initialized = True
        logger.info("GraphRAG initialized successfully")
        return self

    # ------------------------------------------------------------------
    # SEARCH — direct ChromaDB semantic search, no graph traversal
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs):

        if not self._is_initialized:
            self.initialize()

        perf = PerformanceMetrics()
        retrieval = RetrievalMetrics()
        start = time.perf_counter()

        try:
            query_embedding = self.embedder.embed_query(query)

            results = self.store.search(
                query_embedding,
                top_k=self.top_k,
            )

            perf.latency_ms = (time.perf_counter() - start) * 1000

            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=results,
                performance=perf,
                retrieval=retrieval,
            )

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=[],
                performance=perf,
                retrieval=retrieval,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # ANSWER — LLM generation from retrieved context
    # ------------------------------------------------------------------

    def answer(self, query: str, context: str, **kwargs):

        prompt = f"""
You are a knowledgeable entertainment assistant.

Context:
{context}

Question:
{query}

Answer clearly based on the context. Include movie titles, ratings,
directors, actors, and any relevant details from the context.
"""

        try:
            return run_chain(query, prompt)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"

    # ------------------------------------------------------------------
    # SEARCH AND ANSWER — full pipeline
    # ------------------------------------------------------------------

    def search_and_answer(self, query: str, **kwargs):

        result = self.search(query, **kwargs)

        if result.error or not result.results:
            return result

        context = "\n\n".join(
            [r["content"] for r in result.results]
        )

        result.answer = self.answer(query, context)
        result.context_used = context[:500]

        return result

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    def get_name(self):
        return "GraphRAG"

    def get_config(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "traversal_depth": self.traversal_depth,
            "max_nodes": self.max_nodes,
        }