"""
ChromaDB Vector Store
---------------------
Production-grade ChromaDB integration with collections, metadata filtering,
and multiple search modes for the benchmarking system.
"""
import uuid
import time
from typing import List, Dict, Optional, Any, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.preprocessing.models import TextChunk

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    ChromaDB vector store for semantic search.
    
    Features:
    - Persistent and in-memory modes
    - Metadata filtering
    - Multiple search modes (similarity, MMR)
    - Batch operations
    - Collection management
    """
    
    def __init__(self, 
                 collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[int] = None):
        self.config = get_config()
        
        self.collection_name = collection_name or self.config.chromadb.collection_name
        self.persist_dir = persist_directory or self.config.chromadb.persist_directory
        self.host = host or self.config.chromadb.host
        self.port = port or self.config.chromadb.port
        
        self.client = None
        self.collection = None
        self._is_initialized = False
        
        # Metrics
        self.inserted_count = 0
        self.query_count = 0
        self.total_query_time_ms = 0
        
        logger.info(f"ChromaVectorStore configured: collection={self.collection_name}")
    
    def connect(self, mode: str = "persistent") -> "ChromaVectorStore":
        """
        Connect to ChromaDB.
        
        Args:
            mode: 'persistent', 'memory', or 'server'
        
        Returns:
            Self for chaining
        """
        try:
            if mode == "persistent":
                import os
                os.makedirs(self.persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(
                        anonymized_telemetry=self.config.chromadb.anonymized_telemetry
                    )
                )
                logger.info(f"Connected to persistent ChromaDB at: {self.persist_dir}")
                
            elif mode == "memory":
                self.client = chromadb.EphemeralClient()
                logger.info("Connected to in-memory ChromaDB")
                
            elif mode == "server":
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port
                )
                logger.info(f"Connected to ChromaDB server at {self.host}:{self.port}")
            
            self._is_initialized = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def create_collection(self, 
                          name: Optional[str] = None,
                          distance: str = "cosine") -> "ChromaVectorStore":
        """
        Create or get a collection.
        
        Args:
            name: Collection name (default from config)
            distance: Distance metric ('cosine', 'l2', 'ip')
        
        Returns:
            Self for chaining
        """
        if not self._is_initialized:
            self.connect()
        
        collection_name = name or self.collection_name
        
        try:
            # Try to get existing
            self.collection = self.client.get_collection(name=collection_name)
            count = self.collection.count()
            logger.info(f"Using existing collection '{collection_name}' with {count} documents")
        except Exception:
            # Create new
            distance_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip",
                "euclidean": "l2"
            }
            
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_map.get(distance, "cosine")},
            )
            logger.info(f"Created new collection: {collection_name}")
        
        return self
    
    def add_chunks(self, chunks: List[TextChunk], 
                   batch_size: int = 100) -> Dict[str, Any]:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects (must have embeddings)
            batch_size: Insert batch size
        
        Returns:
            Insertion statistics
        """
        if not self.collection:
            raise ValueError("No collection. Call create_collection() first.")
        
        # Filter to chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if len(valid_chunks) != len(chunks):
            logger.warning(f"Skipping {len(chunks) - len(valid_chunks)} chunks without embeddings")
        
        total_inserted = 0
        start_time = time.time()
        
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            
            ids = [c.chunk_id for c in batch]
            embeddings = [c.embedding for c in batch]
            documents = [c.content for c in batch]
            metadatas = [{
                "doc_id": c.doc_id,
                "source_title": c.source_title,
                "chunk_type": c.chunk_type,
                **{f"meta_{k}": str(v) for k, v in c.metadata.items()}
            } for c in batch]
            
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                total_inserted += len(batch)
                
            except Exception as e:
                logger.error(f"Batch insert failed at offset {i}: {e}")
        
        elapsed = time.time() - start_time
        self.inserted_count += total_inserted
        
        stats = {
            "inserted": total_inserted,
            "skipped": len(chunks) - len(valid_chunks),
            "time_seconds": round(elapsed, 2),
            "total_in_collection": self.collection.count()
        }
        
        logger.info(f"Added {total_inserted} chunks in {elapsed:.1f}s")
        return stats
    
    def search(self, 
               query_embedding: List[float],
               top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filtering.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Metadata filters (e.g., {"chunk_type": "overview"})
        
        Returns:
            List of result dictionaries with score, content, metadata
        """
        if not self.collection:
            raise ValueError("No collection. Call create_collection() first.")
        
        start = time.time()
        
        # Build where clause for metadata filtering
        where_clause = self._build_where_clause(filters) if filters else None
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted = self._format_results(results)
            
            elapsed = (time.time() - start) * 1000
            self.query_count += 1
            self.total_query_time_ms += elapsed
            
            return formatted
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_with_mmr(self,
                        query_embedding: List[float],
                        top_k: int = 5,
                        fetch_k: int = 20,
                        lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance search for diversity.
        
        Args:
            query_embedding: Query vector
            top_k: Final number of results
            fetch_k: Number of candidates to fetch
            lambda_mult: Balance between relevance and diversity (0-1)
        
        Returns:
            Diverse results list
        """
        # Fetch more results
        all_results = self.search(query_embedding, top_k=fetch_k)
        
        if not all_results:
            return []
        
        # MMR reranking
        selected = []
        candidates = all_results.copy()
        
        while len(selected) < top_k and candidates:
            if not selected:
                # Pick most relevant first
                best = max(candidates, key=lambda x: x["score"])
            else:
                # Pick with highest MMR score
                def mmr_score(candidate):
                    relevance = candidate["score"]
                    max_sim = max(self._cosine_similarity(
                        candidate.get("embedding", []),
                        s.get("embedding", [])
                    ) for s in selected) if selected else 0
                    return lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                best = max(candidates, key=mmr_score)
            
            selected.append(best)
            candidates.remove(best)
        
        return selected
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        try:
            result = self.collection.get(ids=[doc_id])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception as e:
            logger.warning(f"Failed to get doc {doc_id}: {e}")
        return None
    
    def delete_collection(self, name: Optional[str] = None) -> bool:
        """Delete a collection."""
        try:
            col_name = name or self.collection_name
            self.client.delete_collection(name=col_name)
            logger.info(f"Deleted collection: {col_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            return {"status": "not_initialized"}
        
        avg_query_time = (self.total_query_time_ms / self.query_count) if self.query_count > 0 else 0
        
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "total_inserted": self.inserted_count,
            "total_queries": self.query_count,
            "avg_query_time_ms": round(avg_query_time, 2),
            "persist_directory": self.persist_dir,
        }
    
    def _build_where_clause(self, filters: Dict) -> Dict:
        """Build ChromaDB where clause from filter dict."""
        conditions = []
        for key, value in filters.items():
            conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
    
    def _format_results(self, raw_results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB query results."""
        formatted = []
        
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        
        for i in range(len(ids)):
            # Convert distance to similarity score (cosine distance -> similarity)
            distance = distances[i] if distances else 0
            similarity = 1.0 - distance if distance is not None else 0
            
            formatted.append({
                "id": ids[i],
                "content": documents[i] if documents else "",
                "metadata": metadatas[i] if metadatas else {},
                "score": round(similarity, 4),
                "distance": round(distance, 4) if distance is not None else None,
            })
        
        return formatted
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        if not self.client:
            return []
        return [c.name for c in self.client.list_collections()]
    