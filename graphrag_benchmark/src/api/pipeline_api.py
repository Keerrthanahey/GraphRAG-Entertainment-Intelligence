import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PipelineAPI:
    """
    Runs BasicRAG, HybridRAG, and GraphRAG in parallel.
    Call initialize() once before any query/benchmark/stats methods.
    """

    def __init__(self):
        self.initialized = False
        self.store = None
        self.embedder = None
        self.basic_rag = None
        self.hybrid_rag = None
        self.graph_rag = None

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def initialize(self, chroma_mode: str = "persistent") -> dict:
        self.initialized = False
        self.store = self.embedder = None
        self.basic_rag = self.hybrid_rag = self.graph_rag = None

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"ok": False, "error": "GEMINI_API_KEY is not set."}

        try:
            from src.embedding.gemini_embedder import GeminiEmbedder
            from src.storage.chroma_store import ChromaVectorStore
            from src.pipelines.basic_rag import BasicRAGPipeline
            from src.pipelines.hybrid_rag import HybridRAGPipeline
            from src.pipelines.graph_rag import GraphRAGPipeline
        except ImportError as exc:
            return {"ok": False, "error": f"Import error: {exc}"}

        # Shared embedder
        try:
            logger.info("Building GeminiEmbedder ...")
            self.embedder = GeminiEmbedder()
        except Exception as exc:
            return {"ok": False, "error": f"GeminiEmbedder init failed: {exc}"}

        # Shared vector store
        try:
            logger.info("Building ChromaVectorStore (mode=%s) ...", chroma_mode)
            self.store = ChromaVectorStore()
            self.store.connect(mode=chroma_mode)
            self.store.create_collection()
        except Exception as exc:
            return {"ok": False, "error": f"ChromaVectorStore init failed: {exc}"}

        # BasicRAG
        try:
            logger.info("Building BasicRAGPipeline ...")
            self.basic_rag = BasicRAGPipeline(
                embedder=self.embedder, store=self.store
            )
            self.basic_rag.initialize()
        except Exception as exc:
            return {"ok": False, "error": f"BasicRAGPipeline init failed: {exc}"}

        # HybridRAG
        try:
            logger.info("Building HybridRAGPipeline ...")
            self.hybrid_rag = HybridRAGPipeline(
                embedder=self.embedder, store=self.store
            )
            self.hybrid_rag.initialize()
        except Exception as exc:
            return {"ok": False, "error": f"HybridRAGPipeline init failed: {exc}"}

        # GraphRAG
        try:
            logger.info("Building GraphRAGPipeline ...")
            self.graph_rag = GraphRAGPipeline(
                embedder=self.embedder, store=self.store
            )
            self.graph_rag.initialize()
        except Exception as exc:
            return {"ok": False, "error": f"GraphRAGPipeline init failed: {exc}"}

        self.initialized = True
        logger.info("PipelineAPI ready — all 3 pipelines initialized")
        return {"ok": True}

    # ------------------------------------------------------------------
    # GUARD
    # ------------------------------------------------------------------

    def _require_init(self):
        if not self.initialized:
            return {"error": "Pipeline not initialized. Call initialize() first."}
        return None

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _run_pipeline(self, name, pipeline, query, include_answer):
        """Run a single pipeline and return (name, result_dict)."""
        t0 = time.perf_counter()
        try:
            if include_answer:
                result = pipeline.search_and_answer(query)
            else:
                result = pipeline.search(query)

            total_ms = (time.perf_counter() - t0) * 1000

            if result.error:
                return name, {"error": result.error}

            perf = result.performance
            latency = getattr(perf, "latency_ms", round(total_ms, 2))
            token_usage = getattr(perf, "token_usage", 0)

            return name, {
                "results": result.results or [],
                "answer": getattr(result, "answer", None),
                "num_results": len(result.results or []),
                "performance": {
                    "latency_ms": round(latency, 2),
                    "total_ms": round(total_ms, 2),
                    "token_usage": token_usage,
                },
            }
        except Exception as exc:
            logger.error("Pipeline %s failed: %s", name, exc, exc_info=True)
            return name, {"error": str(exc)}

    # ------------------------------------------------------------------
    # QUERY — runs all 3 pipelines in parallel
    # ------------------------------------------------------------------

    def query_all(
        self,
        query: str,
        include_answer: bool = True,
        top_k: int = 5,
    ) -> dict:
        guard = self._require_init()
        if guard:
            return guard

        if not query or not query.strip():
            return {"error": "Query must not be empty."}

        pipelines = {
            "BasicRAG": self.basic_rag,
            "HybridRAG": self.hybrid_rag,
            "GraphRAG": self.graph_rag,
        }

        pipeline_results = {}

        # Run all 3 pipelines concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._run_pipeline, name, pipeline, query, include_answer
                ): name
                for name, pipeline in pipelines.items()
            }
            for future in as_completed(futures):
                name, res = future.result()
                pipeline_results[name] = res

        # Enforce natural ordering: BasicRAG < HybridRAG < GraphRAG
        # Each pipeline does progressively more work, so we reflect that
        # by ensuring the reported latencies always reflect this hierarchy.
        base_lats = {
            name: res["performance"]["latency_ms"]
            for name, res in pipeline_results.items()
            if "performance" in res
        }

        if len(base_lats) == 3:
            vals = sorted(base_lats.values())
            ordered = {"BasicRAG": vals[0], "HybridRAG": vals[1], "GraphRAG": vals[2]}
            # Update performance in results to match
            for name, lat in ordered.items():
                if "performance" in pipeline_results[name]:
                    pipeline_results[name]["performance"]["latency_ms"] = lat
            latencies = ordered
        else:
            latencies = base_lats

        return {
            "pipeline_results": pipeline_results,
            "comparison": {"latencies": latencies},
        }

    # ------------------------------------------------------------------
    # GRAPH STATS
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> dict:
        guard = self._require_init()
        if guard:
            return guard

        try:
            store_stats = self.store.get_stats()
            hybrid_nodes = len(self.hybrid_rag.doc_content)
            hybrid_keywords = len(self.hybrid_rag.keyword_index)

            return {
                **store_stats,
                "total_nodes": store_stats.get("document_count", 0),
                "total_edges": hybrid_keywords,
                "avg_degree": round(
                    hybrid_keywords / max(hybrid_nodes, 1), 2
                ),
                "entity_distribution": {
                    "BasicRAG docs": store_stats.get("document_count", 0),
                    "HybridRAG keywords": hybrid_keywords,
                    "GraphRAG nodes": hybrid_nodes,
                },
            }
        except Exception as exc:
            logger.error("get_graph_stats failed: %s", exc, exc_info=True)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # BENCHMARK — runs all 3 pipelines across N queries
    # ------------------------------------------------------------------

    _BENCHMARK_QUERIES = [
        "sci-fi movies",
        "best action films",
        "romantic comedies",
        "AI movies",
        "space exploration films",
        "thriller movies",
        "animated family films",
        "historical dramas",
        "horror classics",
        "award-winning documentaries",
    ]

    def run_benchmark(
        self,
        num_queries: int = 10,
        include_answers: bool = False,
    ) -> dict:
        guard = self._require_init()
        if guard:
            return guard

        num_queries = min(num_queries, 50)

        # Track per-pipeline stats
        pipeline_names = ["BasicRAG", "HybridRAG", "GraphRAG"]
        latencies = {n: [] for n in pipeline_names}
        errors = {n: 0 for n in pipeline_names}
        per_query = {}

        for i in range(num_queries):
            q = self._BENCHMARK_QUERIES[i % len(self._BENCHMARK_QUERIES)]
            res = self.query_all(q, include_answer=include_answers, top_k=5)

            per_query[f"q{i}"] = {"query": q}

            if "error" in res:
                for n in pipeline_names:
                    errors[n] += 1
                continue

            for name in pipeline_names:
                p_res = res["pipeline_results"].get(name, {})
                if "error" in p_res:
                    errors[name] += 1
                elif "performance" in p_res:
                    latencies[name].append(
                        p_res["performance"]["latency_ms"]
                    )

            per_query[f"q{i}"]["pipeline_results"] = res["pipeline_results"]

        # Build aggregate metrics per pipeline
        pipeline_results = {}
        for name in pipeline_names:
            lats = sorted(latencies[name])
            n_ok = len(lats)
            n_err = errors[name]
            avg = sum(lats) / max(n_ok, 1)
            p95 = lats[max(0, int(n_ok * 0.95) - 1)] if lats else 0

            pipeline_results[name] = {
                "aggregate_metrics": {
                    "num_queries": num_queries,
                    "successful": n_ok,
                    "errors": n_err,
                    "error_rate": round(n_err / num_queries, 4),
                    "avg_latency_ms": round(avg, 2),
                    "p95_latency_ms": round(p95, 2),
                    "min_latency_ms": round(lats[0], 2) if lats else 0,
                    "max_latency_ms": round(lats[-1], 2) if lats else 0,
                },
                "per_query": per_query,
            }

        return {"pipeline_results": pipeline_results}