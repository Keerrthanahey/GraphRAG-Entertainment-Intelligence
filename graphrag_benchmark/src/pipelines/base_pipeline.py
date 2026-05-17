"""
Base Pipeline Interface
-----------------------
Abstract base class for all RAG pipelines.
Defines the contract that BasicRAG, HybridRAG, and GraphRAG must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.utils.metrics import PerformanceMetrics, RetrievalMetrics


@dataclass
class PipelineResult:
    """Standard result container for all pipeline executions."""
    query: str
    pipeline_name: str
    results: List[Dict[str, Any]]
    performance: PerformanceMetrics
    retrieval: RetrievalMetrics
    context_used: str = ""
    answer: str = ""
    relationship_data: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline_name,
            "query": self.query,
            "num_results": len(self.results),
            "performance": self.performance.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "has_answer": bool(self.answer),
            "has_relationships": self.relationship_data is not None,
            "error": self.error,
        }


class BasePipeline(ABC):
    """
    Abstract base class for all RAG pipelines.
    
    All pipelines must implement:
    - search(): Execute retrieval
    - answer(): Generate response from context
    - get_name(): Return pipeline identifier
    - get_config(): Return pipeline configuration
    """
    
    def __init__(self, name: str):
        self.name = name
        self.config = {}
        self._is_initialized = False
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> PipelineResult:
        """
        Execute search/query against the pipeline.
        
        Args:
            query: User query string
            **kwargs: Pipeline-specific parameters
        
        Returns:
            PipelineResult with results and metrics
        """
        pass
    
    @abstractmethod
    def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Generate an answer from retrieved context.
        
        Args:
            query: User query
            context: Retrieved context string
            **kwargs: Generation parameters
        
        Returns:
            Generated answer string
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return pipeline name for benchmarking."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return pipeline configuration for display."""
        pass
    
    def initialize(self) -> "BasePipeline":
        """Initialize pipeline resources. Override if needed."""
        self._is_initialized = True
        return self
    
    def health_check(self) -> Dict[str, Any]:
        """Check pipeline health status."""
        return {
            "pipeline": self.name,
            "initialized": self._is_initialized,
            "status": "healthy" if self._is_initialized else "not_initialized"
        }
