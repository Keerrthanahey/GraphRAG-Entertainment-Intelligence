"""
Preprocessing Orchestrator
--------------------------
Main entry point for data preprocessing pipeline.
Coordinates CSV loading, cleaning, chunking, and entity extraction.
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from .csv_loader import CSVLoader, DataSplitter
from .chunker import ChunkingEngine
from .entity_extractor import EntityExtractor
from .models import Movie, TextChunk
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)


class PreprocessingOrchestrator:
    """
    Main orchestrator for the preprocessing pipeline.
    
    Pipeline stages:
    1. Load and validate CSV
    2. Clean and normalize data
    3. Convert to Movie objects
    4. Generate text chunks
    5. Extract entities and relationships
    6. Save processed artifacts
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.loader: Optional[CSVLoader] = None
        self.chunker: Optional[ChunkingEngine] = None
        self.extractor: Optional[EntityExtractor] = None
        
        # Outputs
        self.movies: List[Movie] = []
        self.train_movies: List[Movie] = []
        self.val_movies: List[Movie] = []
        self.test_movies: List[Movie] = []
        self.chunks: List[TextChunk] = []
        self.entities = []
        self.relationships = []
        
        logger.info("PreprocessingOrchestrator initialized")
    
    def run(self, csv_path: Optional[str] = None,
            sample_size: Optional[int] = None,
            chunk_strategy: str = "movie_aware",
            save_artifacts: bool = True) -> dict:
        """
        Run the full preprocessing pipeline.
        
        Args:
            csv_path: Path to CSV file (overrides config)
            sample_size: Load only N rows for testing
            chunk_strategy: Chunking strategy name
            save_artifacts: Save processed data to disk
        
        Returns:
            Dictionary with pipeline statistics and output paths
        """
        import time
        start_time = time.time()
        stats = {"stages": {}}
        
        # Stage 1: Load CSV
        logger.info("=" * 50)
        logger.info("Stage 1: Loading CSV Dataset")
        logger.info("=" * 50)
        
        csv_path = csv_path or self.config.dataset.raw_path
        self.loader = CSVLoader(csv_path)
        self.movies = self.loader.load(sample_size=sample_size)
        stats["stages"]["load"] = {
            "total_movies": len(self.movies),
            "csv_path": str(csv_path)
        }
        
        # Stage 2: Split dataset
        logger.info("\n" + "=" * 50)
        logger.info("Stage 2: Splitting Dataset")
        logger.info("=" * 50)
        
        self.train_movies, self.val_movies, self.test_movies = DataSplitter.split(
            self.movies
        )
        stats["stages"]["split"] = {
            "train": len(self.train_movies),
            "val": len(self.val_movies),
            "test": len(self.test_movies)
        }
        
        # Stage 3: Chunking
        logger.info("\n" + "=" * 50)
        logger.info("Stage 3: Text Chunking")
        logger.info("=" * 50)
        
        self.chunker = ChunkingEngine(
            chunk_size=self.config.dataset.chunk_size,
            chunk_overlap=self.config.dataset.chunk_overlap,
            min_chunk_length=self.config.dataset.min_chunk_length
        )
        self.chunks = self.chunker.chunk_movies(self.train_movies, strategy=chunk_strategy)
        
        # Chunk type distribution
        type_dist = {}
        for c in self.chunks:
            type_dist[c.chunk_type] = type_dist.get(c.chunk_type, 0) + 1
        
        stats["stages"]["chunking"] = {
            "total_chunks": len(self.chunks),
            "strategy": chunk_strategy,
            "type_distribution": type_dist,
            "avg_chunk_size": sum(len(c.content) for c in self.chunks) / max(len(self.chunks), 1)
        }
        
        # Stage 4: Entity Extraction
        logger.info("\n" + "=" * 50)
        logger.info("Stage 4: Entity & Relationship Extraction")
        logger.info("=" * 50)
        
        self.extractor = EntityExtractor()
        self.entities, self.relationships = self.extractor.extract_from_movies(self.train_movies)
        
        stats["stages"]["extraction"] = {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": self.extractor.get_entity_stats(),
            "relationship_types": self.extractor.get_relationship_stats()
        }
        
        # Stage 5: Save artifacts
        if save_artifacts:
            logger.info("\n" + "=" * 50)
            logger.info("Stage 5: Saving Artifacts")
            logger.info("=" * 50)
            
            artifact_paths = self._save_artifacts()
            stats["stages"]["artifacts"] = artifact_paths
        
        elapsed = time.time() - start_time
        stats["total_time_seconds"] = round(elapsed, 2)
        
        logger.info("\n" + "=" * 50)
        logger.info(f"Pipeline Complete in {elapsed:.1f}s")
        logger.info("=" * 50)
        
        return stats
    
    def _save_artifacts(self) -> dict[str, str]:
        """Save all processed artifacts to disk."""
        output_dir = Path(self.config.dataset.processed_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save movies as JSON
        movies_path = output_dir / "movies.json"
        with open(movies_path, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in self.movies], f, indent=2, ensure_ascii=False)
        paths["movies_json"] = str(movies_path)
        
        # Save train/val/test splits
        for split_name, split_data in [
            ("train", self.train_movies),
            ("val", self.val_movies),
            ("test", self.test_movies)
        ]:
            split_path = output_dir / f"{split_name}_movies.json"
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump([m.to_dict() for m in split_data], f, indent=2, ensure_ascii=False)
            paths[f"{split_name}_movies"] = str(split_path)
        
        # Save chunks
        chunks_path = output_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2, ensure_ascii=False)
        paths["chunks_json"] = str(chunks_path)
        
        # Save chunks as pickle (preserves full objects)
        chunks_pkl = output_dir / "chunks.pkl"
        with open(chunks_pkl, 'wb') as f:
            pickle.dump(self.chunks, f)
        paths["chunks_pkl"] = str(chunks_pkl)
        
        # Save entities
        entities_path = output_dir / "entities.json"
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump([{"name": e.name, "type": e.entity_type, "metadata": e.metadata} 
                      for e in self.entities], f, indent=2, ensure_ascii=False)
        paths["entities_json"] = str(entities_path)
        
        # Save relationships
        rels_path = output_dir / "relationships.json"
        with open(rels_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.relationships], f, indent=2, ensure_ascii=False)
        paths["relationships_json"] = str(rels_path)
        
        logger.info(f"Artifacts saved to: {output_dir}")
        return paths
    
    def load_artifacts(self, processed_dir: Optional[str] = None) -> dict:
        """
        Load previously saved artifacts.
        
        Args:
            processed_dir: Directory containing processed files
        
        Returns:
            Dictionary with loaded data statistics
        """
        import pickle
        
        proc_dir = Path(processed_dir or self.config.dataset.processed_path)
        stats = {}
        
        # Load chunks
        chunks_pkl = proc_dir / "chunks.pkl"
        if chunks_pkl.exists():
            with open(chunks_pkl, 'rb') as f:
                self.chunks = pickle.load(f)
            stats["chunks_loaded"] = len(self.chunks)
            logger.info(f"Loaded {len(self.chunks)} chunks from pickle")
        
        # Load relationships
        rels_path = proc_dir / "relationships.json"
        if rels_path.exists():
            import json
            with open(rels_path, 'r') as f:
                rels_data = json.load(f)
            stats["relationships_loaded"] = len(rels_data)
            logger.info(f"Loaded {len(rels_data)} relationships")
        
        return stats
