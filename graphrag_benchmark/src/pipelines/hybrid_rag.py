"""
Hybrid RAG Pipeline
"""

import time
import re

from typing import Dict, Any, Optional

from src.utils.langchain_helper import run_chain
from src.pipelines.base_pipeline import BasePipeline, PipelineResult
from src.embedding.gemini_embedder import GeminiEmbedder
from src.storage.chroma_store import ChromaVectorStore
from src.utils.metrics import PerformanceMetrics, RetrievalMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRAGPipeline(BasePipeline):

    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        store: Optional[ChromaVectorStore] = None,
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):

        super().__init__("HybridRAG")

        self.embedder = embedder
        self.store = store

        self.top_k = top_k

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.keyword_index = {}
        self.doc_content = {}

        logger.info("HybridRAG initialized")

    def initialize(self):

        if not self.embedder:
            self.embedder = GeminiEmbedder()

        if not self.store:
            self.store = (
                ChromaVectorStore()
                .connect("persistent")
                .create_collection()
            )

        self._build_keyword_index()

        self._is_initialized = True

        logger.info("HybridRAG initialized successfully")

        return self

    def _tokenize(self, text):

        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", " ", text)

        words = text.split()

        return [w for w in words if len(w) > 2]

    def _build_keyword_index(self):

        try:

            all_docs = self.store.collection.get(
                include=["documents"]
            )

            for doc_id, content in zip(
                all_docs["ids"],
                all_docs["documents"],
            ):

                self.doc_content[doc_id] = content

                words = self._tokenize(content)

                for word in words:

                    if word not in self.keyword_index:
                        self.keyword_index[word] = []

                    self.keyword_index[word].append(doc_id)

        except Exception as e:

            logger.error(f"Keyword indexing failed: {e}")

    def _keyword_search(self, query):

        scores = {}

        terms = self._tokenize(query)

        for term in terms:

            docs = self.keyword_index.get(term, [])

            for doc in docs:

                scores[doc] = scores.get(doc, 0) + 1

        results = []

        for doc_id, score in scores.items():

            results.append(
                {
                    "id": doc_id,
                    "content": self.doc_content.get(doc_id, ""),
                    "score": score,
                    "search_type": "keyword",
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results

    def search(self, query: str, **kwargs):

        if not self._is_initialized:
            self.initialize()

        perf = PerformanceMetrics()
        retrieval = RetrievalMetrics()

        start = time.perf_counter()

        try:

            query_embedding = self.embedder.embed_query(query)

            semantic_results = self.store.search(
                query_embedding,
                top_k=self.top_k,
            )

            keyword_results = self._keyword_search(query)

            combined = {}

            for r in semantic_results:

                combined[r["id"]] = {
                    "id": r["id"],
                    "content": r["content"],
                    "score": (
                        r["score"] * self.semantic_weight
                    ),
                    "search_type": "semantic",
                }

            for r in keyword_results:

                if r["id"] in combined:

                    combined[r["id"]]["score"] += (
                        r["score"] * self.keyword_weight
                    )

                    combined[r["id"]][
                        "search_type"
                    ] = "hybrid"

                else:

                    combined[r["id"]] = {
                        "id": r["id"],
                        "content": r["content"],
                        "score": (
                            r["score"] * self.keyword_weight
                        ),
                        "search_type": "keyword",
                    }

            results = list(combined.values())

            results.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            perf.latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=results[: self.top_k],
                performance=perf,
                retrieval=retrieval,
            )

        except Exception as e:

            logger.error(f"HybridRAG search failed: {e}")

            return PipelineResult(
                query=query,
                pipeline_name=self.name,
                results=[],
                performance=perf,
                retrieval=retrieval,
                error=str(e),
            )

    def answer(
        self,
        query: str,
        context: str,
        **kwargs,
    ):

        prompt = f"""
You are a knowledgeable entertainment assistant.

Context:
{context}

Question:
{query}

Provide a detailed and accurate answer.
"""

        try:

            response = run_chain(query, prompt)

            return response

        except Exception as e:

            logger.error(f"Answer generation failed: {e}")

            return f"Error generating answer: {e}"

    def search_and_answer(self, query: str, **kwargs):

        result = self.search(query, **kwargs)

        if result.error or not result.results:
            return result

        context = "\n\n".join(
            [r["content"] for r in result.results]
        )

        result.answer = self.answer(
            query,
            context,
        )

        result.context_used = context[:500]

        return result

    def get_name(self):

        return "HybridRAG"

    def get_config(self) -> Dict[str, Any]:

        return {
            "top_k": self.top_k,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
        }
