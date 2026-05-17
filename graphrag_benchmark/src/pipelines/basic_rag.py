"""
Basic RAG Pipeline
------------------
Standard retrieval-augmented generation using vector similarity search.
Simple and fast, serves as the baseline for comparison.
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


class BasicRAGPipeline(BasePipeline):
    """
    Basic RAG Pipeline.

    Flow:
    1. Embed query
    2. Vector similarity search in ChromaDB
    3. Retrieve top-k results
    4. Generate answer using LangChain + Gemini
    """

    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        store: Optional[ChromaVectorStore] = None,
        top_k: int = 5,
        score_threshold: float = 0.5
    ):
        super().__init__("BasicRAG")

        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

        logger.info(
            f"BasicRAG initialized: top_k={top_k}, "
            f"threshold={score_threshold}"
        )

    def initialize(self) -> "BasicRAGPipeline":
        """Initialize pipeline components."""

        logger.info("Initializing BasicRAG pipeline...")

        if not self.embedder:
            self.embedder = GeminiEmbedder()

        if not self.store:
            self.store = (
                ChromaVectorStore()
                .connect("persistent")
                .create_collection()
            )

        self._is_initialized = True

        logger.info("BasicRAG pipeline initialized")

        return self

    def search(self, query: str, **kwargs) -> PipelineResult:
        """
        Execute vector similarity search.

        Args:
            query: User query

        Returns:
            PipelineResult
        """

        if not self._is_initialized:
            self.initialize()

        top_k = kwargs.get("top_k", self.top_k)
        threshold = kwargs.get(
            "score_threshold",
            self.score_threshold
        )

        perf = PerformanceMetrics()
        retrieval = RetrievalMetrics()

        start = time.perf_counter()

        try:
            # STEP 1: Embed query
            embed_start = time.perf_counter()

            query_embedding = self.embedder.embed_query(query)

            embed_time = (
                time.perf_counter() - embed_start
            ) * 1000

            # STEP 2: Vector search
            search_start = time.perf_counter()

            raw_results = self.store.search(
                query_embedding,
                top_k=top_k * 2
            )

            search_time = (
                time.perf_counter() - search_start
            ) * 1000

            # STEP 3: Filter by similarity threshold
            results = [
                r for r in raw_results
                if r["score"] >= threshold
            ][:top_k]

            # Performance metrics
            perf.latency_ms = (
                time.perf_counter() - start
            ) * 1000

            perf.token_usage = len(query) // 4

            logger.info(
                f"BasicRAG: {len(results)} results "
                f"in {perf.latency_ms:.1f}ms "
                f"(embed: {embed_time:.1f}ms, "
                f"search: {search_time:.1f}ms)"
            )

            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=results,
                performance=perf,
                retrieval=retrieval
            )

        except Exception as e:

            logger.error(f"BasicRAG search failed: {e}")

            perf.latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=[],
                performance=perf,
                retrieval=retrieval,
                error=str(e)
            )

    def answer(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Generate answer using LangChain + Gemini.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Generated answer
        """

        try:
            response = run_chain(
                query=query,
                context=context
            )

            return response

        except Exception as e:

            logger.error(
                f"Answer generation failed: {e}"
            )

            return f"Error generating answer: {e}"

    def search_and_answer(
        self,
        query: str,
        **kwargs
    ) -> PipelineResult:
        """
        Full RAG pipeline:
        Search + Answer Generation
        """

        # STEP 1: Retrieve documents
        result = self.search(query, **kwargs)

        if result.error or not result.results:
            return result

        # STEP 2: Build context
        context_parts = []

        for i, res in enumerate(result.results, start=1):

            content = res.get("content", "")

            context_parts.append(
                f"[{i}] {content}"
            )

        context = "\n\n".join(context_parts)

        # STEP 3: Generate answer
        result.answer = self.answer(
            query=query,
            context=context,
            **kwargs
        )

        # Store shortened context
        if len(context) > 500:
            result.context_used = context[:500] + "..."
        else:
            result.context_used = context

        return result

    def get_name(self) -> str:
        """Return pipeline name."""
        return "Basic RAG (Vector Search)"

    def get_config(self) -> Dict[str, Any]:
        """Return pipeline configuration."""

        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "embedding_model": (
                self.embedder.model
                if self.embedder
                else "unknown"
            ),
            "store_type": "ChromaDB"
        }