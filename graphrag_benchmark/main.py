"""
GraphRAG Entertainment Intelligence - Main Entry Point
=======================================================

Preprocess IMDb dataset, generate embeddings, store in ChromaDB, and test retrieval.

Usage:
    # Full pipeline
    python main.py --csv data/raw/imdb_top_1000.csv --full-pipeline
    
    # Just preprocessing
    python main.py --csv data/raw/imdb_top_1000.csv --preprocess-only
    
    # Just embedding and storage
    python main.py --embed-only
    
    # Test retrieval
    python main.py --test-query "Movies directed by Christopher Nolan"
    
    # Run benchmark
    python main.py --benchmark
    
    # Launch dashboard
    streamlit run dashboard.py
"""
import os
import sys
import json
import argparse
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.preprocessing.orchestrator import PreprocessingOrchestrator
from src.embedding.gemini_embedder import GeminiEmbedder
from src.storage.chroma_store import ChromaVectorStore
from src.pipelines.basic_rag import BasicRAGPipeline
from src.pipelines.hybrid_rag import HybridRAGPipeline
from src.pipelines.graph_rag import GraphRAGPipeline
from src.benchmark.runner import BenchmarkRunner

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Entertainment Intelligence System"
    )
    
    parser.add_argument("--csv", type=str, 
                       default="data/raw/imdb_top_1000.csv",
                       help="Path to IMDb CSV file")
    parser.add_argument("--sample", type=int, default=None,
                       help="Use only N random rows (for testing)")
    parser.add_argument("--chunk-strategy", type=str, 
                       default="movie_aware",
                       choices=["fixed", "semantic", "movie_aware", "hierarchical"],
                       help="Text chunking strategy")
    
    # Pipeline stages
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run complete pipeline: preprocess + embed + store")
    parser.add_argument("--preprocess-only", action="store_true",
                       help="Run only preprocessing")
    parser.add_argument("--embed-only", action="store_true",
                       help="Run only embedding and storage")
    
    # Testing
    parser.add_argument("--test-query", type=str, default=None,
                       help="Test query for retrieval")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark suite")
    parser.add_argument("--benchmark-queries", type=int, default=10,
                       help="Number of benchmark queries")
    
    # Storage
    parser.add_argument("--chroma-mode", type=str, default="persistent",
                       choices=["persistent", "memory"],
                       help="ChromaDB storage mode")
    parser.add_argument("--clear-collection", action="store_true",
                       help="Clear collection before adding")
    
    # General
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def check_api_key():
    """Check if GEMINI_API_KEY is set."""
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("=" * 60)
        logger.error("GEMINI_API_KEY environment variable not set!")
        logger.error("=" * 60)
        logger.error("\nTo set it:")
        logger.error("  export GEMINI_API_KEY='your-key-here'")
        logger.error("\nGet your key at: https://makersuite.google.com/app/apikey")
        return False
    return True


def run_preprocessing(args) -> PreprocessingOrchestrator:
    """Run preprocessing pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    orchestrator = PreprocessingOrchestrator()
    
    stats = orchestrator.run(
        csv_path=args.csv,
        sample_size=args.sample,
        chunk_strategy=args.chunk_strategy,
        save_artifacts=True
    )
    
    logger.info("\nPreprocessing Stats:")
    logger.info(json.dumps(stats, indent=2))
    
    return orchestrator


def run_embedding_and_storage(args, orchestrator: PreprocessingOrchestrator):
    """Run embedding and storage pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING & STORAGE PIPELINE")
    logger.info("=" * 60)
    
    if not check_api_key():
        logger.error("Skipping embedding - no API key")
        return None, None
    
    # Initialize embedder
    embedder = GeminiEmbedder()
    logger.info(f"Embedder: {embedder.model}")
    
    # Embed chunks
    logger.info(f"Embedding {len(orchestrator.chunks)} chunks...")
    embedded_chunks = embedder.embed_chunks(orchestrator.chunks, show_progress=True)
    
    embed_stats = embedder.get_stats()
    logger.info(f"\nEmbedding Stats:")
    logger.info(json.dumps(embed_stats, indent=2))
    
    # Initialize ChromaDB
    store = ChromaVectorStore()
    store.connect(mode=args.chroma_mode)
    
    if args.clear_collection:
        try:
            store.delete_collection()
        except Exception:
            pass
    
    store.create_collection()
    
    # Store chunks
    logger.info(f"\nStoring chunks in ChromaDB...")
    insert_stats = store.add_chunks(embedded_chunks, batch_size=100)
    
    logger.info(f"\nStorage Stats:")
    logger.info(json.dumps(insert_stats, indent=2))
    
    # Store stats
    logger.info(f"\nCollection Stats:")
    logger.info(json.dumps(store.get_stats(), indent=2))
    
    return embedder, store


def test_retrieval(embedder, store, query: str):
    """Test retrieval with a query."""
    logger.info("\n" + "=" * 60)
    logger.info("RETRIEVAL TEST")
    logger.info("=" * 60)
    
    logger.info(f"Query: '{query}'")
    
    # Embed query
    query_emb = embedder.embed_query(query)
    logger.info(f"Query embedded: {len(query_emb)} dimensions")
    
    # Search
    results = store.search(query_emb, top_k=5)
    
    logger.info(f"\nFound {len(results)} results:\n")
    
    for i, res in enumerate(results, 1):
        logger.info(f"--- Result {i} ---")
        logger.info(f"Score: {res['score']:.4f}")
        logger.info(f"ID: {res['id']}")
        logger.info(f"Content: {res['content'][:200]}...")
        logger.info("")
    
    return results


def run_benchmark_suite(embedder, store, num_queries: int = 10):
    """Run full benchmark across all pipelines."""
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUITE")
    logger.info("=" * 60)
    
    # Initialize pipelines
    pipelines = [
        BasicRAGPipeline(embedder=embedder, store=store),
        HybridRAGPipeline(embedder=embedder, store=store),
        GraphRAGPipeline(embedder=embedder, store=store),
    ]
    
    # Initialize each
    for p in pipelines:
        try:
            p.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize {p.name}: {e}")
    
    # Run benchmark
    runner = BenchmarkRunner(
        pipelines=pipelines,
        warmup_runs=1
    )
    
    results = runner.run(include_answers=False)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("\n")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  GraphRAG Entertainment Intelligence - Pipeline Runner       ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    embedder = None
    store = None
    orchestrator = None
    
    try:
        # Determine what to run
        run_all = args.full_pipeline
        run_preprocess = run_all or args.preprocess_only or not (args.embed_only or args.benchmark)
        run_embed = run_all or args.embed_only or args.test_query or args.benchmark
        
        # Step 1: Preprocessing
        if run_preprocess:
            if not Path(args.csv).exists():
                logger.error(f"CSV file not found: {args.csv}")
                logger.info("Please place your IMDb dataset CSV in the data/raw/ directory")
                return
            
            orchestrator = run_preprocessing(args)
            
            if args.preprocess_only:
                logger.info("\nPreprocessing complete. Exiting.")
                return
        
        # Step 2: Embedding & Storage
        if run_embed:
            if orchestrator is None:
                # Try to load existing artifacts
                orchestrator = PreprocessingOrchestrator()
                load_stats = orchestrator.load_artifacts()
                logger.info(f"Loaded existing artifacts: {load_stats}")
                
                if not orchestrator.chunks:
                    logger.error("No chunks available. Run preprocessing first.")
                    return
            
            embedder, store = run_embedding_and_storage(args, orchestrator)
            
            if args.embed_only:
                logger.info("\nEmbedding and storage complete.")
                return
        
        # Step 3: Test Query
        if args.test_query and embedder and store:
            test_retrieval(embedder, store, args.test_query)
        
        # Step 4: Benchmark
        if args.benchmark and embedder and store:
            run_benchmark_suite(embedder, store, args.benchmark_queries)
        
        # Default: run everything
        if not any([args.preprocess_only, args.embed_only, args.test_query, args.benchmark]):
            if orchestrator is None:
                orchestrator = run_preprocessing(args)
            if embedder is None:
                embedder, store = run_embedding_and_storage(args, orchestrator)
            
            # Test with sample query
            test_retrieval(embedder, store, "Best science fiction movies")
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("  - Launch dashboard: streamlit run dashboard.py")
            logger.info("  - Run benchmark: python main.py --benchmark")
            logger.info("  - Test query: python main.py --test-query 'your question'")
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user.")
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
