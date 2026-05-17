"""
Benchmark Runner
----------------
Systematic benchmarking of all RAG pipelines.
Measures latency, cost, token usage, and retrieval quality.
"""
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from src.pipelines.base_pipeline import BasePipeline, PipelineResult
from src.utils.metrics import MetricsCollector, BenchmarkResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Standard benchmark queries covering different query types
BENCHMARK_QUERIES = [
    # Director queries
    "Movies directed by Christopher Nolan",
    "Films by Steven Spielberg",
    "What movies did Quentin Tarantino direct?",
    
    # Actor queries
    "Movies starring Tom Hanks",
    "Films with Morgan Freeman",
    "Best Leonardo DiCaprio movies",
    
    # Genre queries
    "Best horror movies",
    "Top science fiction films",
    "Classic romance movies",
    "Action movies from the 2000s",
    
    # Theme queries
    "Movies about space exploration",
    "Films about artificial intelligence",
    "War movies",
    "Movies with a twist ending",
    
    # Relationship queries
    "Movies similar to The Matrix",
    "What should I watch if I liked Inception?",
    "Films by the director of The Dark Knight",
    "Movies with the cast of Pulp Fiction",
    
    # Complex queries
    "Best thriller movies from the 1990s with high ratings",
    "Comedy films starring Jim Carrey",
    "Drama movies about family relationships",
    "Highly rated animated films",
    "Movies that won Academy Awards",
]


@dataclass
class PipelineBenchmark:
    """Complete benchmark results for a single pipeline."""
    pipeline_name: str
    pipeline_config: Dict[str, Any]
    results: List[Dict] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    total_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_config": self.pipeline_config,
            "num_queries": len(self.results),
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "results": self.results
        }


class BenchmarkRunner:
    """
    Benchmark runner for comparing RAG pipelines.
    
    Executes standard query set across all pipelines and collects metrics.
    """
    
    def __init__(self, 
                 pipelines: List[BasePipeline],
                 queries: Optional[List[str]] = None,
                 warmup_runs: int = 2,
                 output_dir: str = "benchmark_results"):
        self.pipelines = pipelines
        self.queries = queries or BENCHMARK_QUERIES
        self.warmup_runs = warmup_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.collector = MetricsCollector()
        
        logger.info(f"BenchmarkRunner: {len(pipelines)} pipelines, "
                   f"{len(self.queries)} queries, {warmup_runs} warmup runs")
    
    def run(self, include_answers: bool = False) -> Dict[str, PipelineBenchmark]:
        """
        Run complete benchmark across all pipelines.
        
        Args:
            include_answers: Whether to generate LLM answers (slower)
        
        Returns:
            Dictionary mapping pipeline name to benchmark results
        """
        all_results = {}
        
        for pipeline in self.pipelines:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {pipeline.get_name()}")
            logger.info(f"{'='*60}")
            
            benchmark = self._benchmark_pipeline(
                pipeline, 
                include_answers=include_answers
            )
            all_results[pipeline.name] = benchmark
        
        # Save results
        self._save_results(all_results)
        
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results
    
    def _benchmark_pipeline(self, 
                           pipeline: BasePipeline,
                           include_answers: bool = False) -> PipelineBenchmark:
        """
        Benchmark a single pipeline.
        
        Args:
            pipeline: Pipeline to benchmark
            include_answers: Whether to generate answers
        
        Returns:
            PipelineBenchmark with results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Warmup
        logger.info(f"Warming up with {self.warmup_runs} runs...")
        warmup_query = "movies about adventure"
        for _ in range(self.warmup_runs):
            try:
                if include_answers:
                    pipeline.search_and_answer(warmup_query)
                else:
                    pipeline.search(warmup_query)
            except Exception as e:
                logger.warning(f"Warmup error: {e}")
        
        # Run benchmark queries
        query_results = []
        latencies = []
        tokens = []
        costs = []
        
        for i, query in enumerate(self.queries, 1):
            logger.info(f"  Query {i}/{len(self.queries)}: {query[:60]}...")
            
            try:
                q_start = time.time()
                
                if include_answers:
                    result = pipeline.search_and_answer(query)
                else:
                    result = pipeline.search(query)
                
                q_elapsed = (time.time() - q_start) * 1000
                
                # Collect metrics
                latencies.append(q_elapsed)
                tokens.append(result.performance.token_usage)
                costs.append(result.performance.cost_usd)
                
                query_results.append({
                    "query": query,
                    "latency_ms": round(q_elapsed, 2),
                    "num_results": len(result.results),
                    "avg_score": round(
                        sum(r.get("score", 0) for r in result.results) / max(len(result.results), 1), 4
                    ) if result.results else 0,
                    "has_answer": bool(result.answer),
                    "error": result.error,
                })
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                query_results.append({
                    "query": query,
                    "error": str(e),
                    "latency_ms": 0,
                })
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregates(latencies, tokens, costs, query_results)
        
        benchmark = PipelineBenchmark(
            pipeline_name=pipeline.get_name(),
            pipeline_config=pipeline.get_config(),
            results=query_results,
            aggregate_metrics=aggregate,
            timestamp=timestamp,
            total_time_seconds=total_time
        )
        
        logger.info(f"  Completed in {total_time:.1f}s")
        logger.info(f"  Avg latency: {aggregate.get('avg_latency_ms', 0):.1f}ms")
        
        return benchmark
    
    @staticmethod
    def _calculate_aggregates(latencies: List[float],
                             tokens: List[int],
                             costs: List[float],
                             results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate statistics."""
        if not latencies:
            return {}
        
        successful = [r for r in results if not r.get("error")]
        errors = [r for r in results if r.get("error")]
        
        aggregates = {
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "p95_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2
            ),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_tokens": round(statistics.mean(tokens), 0) if tokens else 0,
            "avg_cost_usd": round(statistics.mean(costs), 6) if costs else 0,
            "total_queries": len(results),
            "successful_queries": len(successful),
            "failed_queries": len(errors),
            "error_rate": round(len(errors) / len(results), 4) if results else 0,
        }
        
        return aggregates
    
    def _save_results(self, all_results: Dict[str, PipelineBenchmark]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        results_dict = {
            name: benchmark.to_dict()
            for name, benchmark in all_results.items()
        }
        
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {json_path}")
        
        # Save summary CSV
        csv_path = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        self._save_summary_csv(csv_path, all_results)
        logger.info(f"Summary saved to: {csv_path}")
    
    @staticmethod
    def _save_summary_csv(path: Path, 
                         all_results: Dict[str, PipelineBenchmark]) -> None:
        """Save summary as CSV."""
        import csv
        
        fieldnames = [
            "pipeline", "avg_latency_ms", "median_latency_ms", 
            "p95_latency_ms", "max_latency_ms", "avg_tokens",
            "error_rate", "total_queries", "successful_queries"
        ]
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for name, benchmark in all_results.items():
                agg = benchmark.aggregate_metrics
                writer.writerow({
                    "pipeline": name,
                    "avg_latency_ms": agg.get("avg_latency_ms", 0),
                    "median_latency_ms": agg.get("median_latency_ms", 0),
                    "p95_latency_ms": agg.get("p95_latency_ms", 0),
                    "max_latency_ms": agg.get("max_latency_ms", 0),
                    "avg_tokens": agg.get("avg_tokens", 0),
                    "error_rate": agg.get("error_rate", 0),
                    "total_queries": agg.get("total_queries", 0),
                    "successful_queries": agg.get("successful_queries", 0),
                })
    
    @staticmethod
    def _print_comparison(all_results: Dict[str, PipelineBenchmark]) -> None:
        """Print comparison table to console."""
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        
        headers = ["Pipeline", "Avg Latency", "P95 Latency", "Error Rate", "Success"]
        col_widths = [25, 15, 15, 12, 10]
        
        # Header
        header_line = " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        print(header_line)
        print("-" * len(header_line))
        
        # Rows
        for name, benchmark in all_results.items():
            agg = benchmark.aggregate_metrics
            row = [
                name[:24],
                f"{agg.get('avg_latency_ms', 0):.1f}ms",
                f"{agg.get('p95_latency_ms', 0):.1f}ms",
                f"{agg.get('error_rate', 0):.2%}",
                f"{agg.get('successful_queries', 0)}/{agg.get('total_queries', 0)}"
            ]
            print(" | ".join(
                str(v).ljust(w) for v, w in zip(row, col_widths)
            ))
        
        print("=" * 80)
