"""
Metrics and Benchmarking Utilities
-----------------------------------
Collection of utilities for measuring retrieval quality and performance.
"""
import time
import statistics
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Performance measurement container."""
    latency_ms: float = 0.0
    token_usage: int = 0
    cost_usd: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "token_usage": self.token_usage,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp
        }


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4)
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single pipeline run."""
    pipeline_name: str
    query: str
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    results: List[Dict[str, Any]] = field(default_factory=list)
    relationship_score: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pipeline": self.pipeline_name,
            "query": self.query,
            "performance": self.performance.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "num_results": len(self.results),
            "relationship_score": self.relationship_score,
        }
        if self.error:
            data["error"] = self.error
        return data


class MetricsCollector:
    """Collect and aggregate metrics across multiple runs."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def add(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics across all runs."""
        if not self.results:
            return {}
        
        by_pipeline: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_pipeline.setdefault(r.pipeline_name, []).append(r)
        
        summary = {}
        for pipeline, runs in by_pipeline.items():
            latencies = [r.performance.latency_ms for r in runs if r.performance.latency_ms > 0]
            tokens = [r.performance.token_usage for r in runs if r.performance.token_usage > 0]
            costs = [r.performance.cost_usd for r in runs if r.performance.cost_usd > 0]
            precisions = [r.retrieval.precision for r in runs if r.retrieval.precision > 0]
            recalls = [r.retrieval.recall for r in runs if r.retrieval.recall > 0]
            ndcgs = [r.retrieval.ndcg for r in runs if r.retrieval.ndcg > 0]
            rel_scores = [r.relationship_score for r in runs if r.relationship_score is not None]
            
            summary[pipeline] = {
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
                "p95_latency_ms": round(self._percentile(latencies, 95), 2) if latencies else 0,
                "avg_tokens": round(statistics.mean(tokens), 0) if tokens else 0,
                "avg_cost_usd": round(statistics.mean(costs), 6) if costs else 0,
                "avg_precision": round(statistics.mean(precisions), 4) if precisions else 0,
                "avg_recall": round(statistics.mean(recalls), 4) if recalls else 0,
                "avg_ndcg": round(statistics.mean(ndcgs), 4) if ndcgs else 0,
                "avg_relationship_score": round(statistics.mean(rel_scores), 4) if rel_scores else 0,
                "total_runs": len(runs),
                "error_rate": sum(1 for r in runs if r.error) / len(runs),
            }
        
        return summary
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]


@contextmanager
def timed_execution():
    """Context manager for timing code blocks."""
    metrics = PerformanceMetrics()
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.latency_ms = (time.perf_counter() - start) * 1000


def calculate_ndcg(relevances: List[float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        relevances: List of relevance scores (0-1) in result order
        k: Cutoff position
    
    Returns:
        NDCG score between 0 and 1
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    def dcg(scores: List[float]) -> float:
        return sum((2**score - 1) / math.log2(i + 2) for i, score in enumerate(scores))
    
    import math
    ideal = sorted(relevances, reverse=True)
    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(ideal)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def calculate_mrr(relevance_list: List[bool]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        relevance_list: Boolean list indicating relevant results
    
    Returns:
        MRR score
    """
    for i, relevant in enumerate(relevance_list):
        if relevant:
            return 1.0 / (i + 1)
    return 0.0
