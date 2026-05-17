"""
Configuration Manager
--------------------
Centralized configuration management with environment variable support.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ProjectConfig:
    """Project-level configuration."""
    name: str = "GraphRAG Entertainment Intelligence"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    raw_path: str = "data/raw/imdb_top_1000.csv"
    processed_path: str = "data/processed"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 100


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str = "gemini"
    model: str = "text-embedding-004"
    dimension: int = 768
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class ChromaDBConfig:
    """ChromaDB vector store configuration."""
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "entertainment_movies"
    distance_metric: str = "cosine"
    persist_directory: str = "./chroma_db"
    anonymized_telemetry: bool = False


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-1.5-flash-latest"
    temperature: float = 0.3
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))


@dataclass
class TigerGraphConfig:
    """TigerGraph connection configuration."""
    host: str = ""
    graphname: str = "EntertainmentGraph"
    username: str = "tigergraph"
    password: str = field(default_factory=lambda: os.getenv("TIGERGRAPH_PASSWORD", ""))
    secret: str = field(default_factory=lambda: os.getenv("TIGERGRAPH_SECRET", ""))
    token: str = field(default_factory=lambda: os.getenv("TIGERGRAPH_TOKEN", ""))


@dataclass
class PipelineConfig:
    """Pipeline-specific configuration."""
    top_k: int = 5
    score_threshold: float = 0.7
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    rerank: bool = True
    traversal_depth: int = 2
    max_nodes: int = 20


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    num_queries: int = 50
    warmup_runs: int = 5
    metrics: list = field(default_factory=lambda: [
        "latency_ms", "token_usage", "cost_usd",
        "retrieval_precision", "retrieval_recall", "ndcg"
    ])
    output_dir: str = "benchmark_results"


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    graph_layout: str = "spring"
    max_nodes_display: int = 50
    node_size_attribute: str = "imdb_rating"
    color_by: str = "genre"


class Config:
    """Central configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project = ProjectConfig()
        self.dataset = DatasetConfig()
        self.embeddings = EmbeddingConfig()
        self.chromadb = ChromaDBConfig()
        self.gemini = GeminiConfig()
        self.tigergraph = TigerGraphConfig()
        self.benchmark = BenchmarkConfig()
        self.visualization = VisualizationConfig()
        
        # Pipeline configs
        self.basic_rag = PipelineConfig(top_k=5, score_threshold=0.7)
        self.hybrid_rag = PipelineConfig(
            top_k=5, semantic_weight=0.6, keyword_weight=0.4, rerank=True
        )
        self.graphrag = PipelineConfig(traversal_depth=2, max_nodes=20)
        
        if config_path:
            self.load_from_yaml(config_path)
        
        self._validate()
    
    def load_from_yaml(self, path: str) -> None:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        for section, values in data.items():
            if hasattr(self, section) and values:
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _validate(self) -> None:
        """Validate critical configuration."""
        if not self.gemini.api_key:
            print("WARNING: GEMINI_API_KEY not set. Embeddings and LLM will not work.")
        
        if self.embeddings.batch_size > 200:
            print("WARNING: Large batch size may cause API rate limiting.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project": self.project.__dict__,
            "dataset": self.dataset.__dict__,
            "embeddings": self.embeddings.__dict__,
            "chromadb": self.chromadb.__dict__,
            "gemini": {k: v for k, v in self.gemini.__dict__.items() if k != 'password'},
            "basic_rag": self.basic_rag.__dict__,
            "hybrid_rag": self.hybrid_rag.__dict__,
            "graphrag": self.graphrag.__dict__,
            "benchmark": self.benchmark.__dict__,
        }


# Global configuration instance
_settings: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global configuration instance."""
    global _settings
    if _settings is None or config_path:
        yaml_path = config_path or os.getenv("CONFIG_PATH", "config/settings.yaml")
        if Path(yaml_path).exists():
            _settings = Config(yaml_path)
        else:
            _settings = Config()
    return _settings
