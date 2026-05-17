"""
Text Chunking Engine
--------------------
Multiple chunking strategies for creating embedding-ready text segments.
"""
import re
import hashlib
from typing import List, Optional, Iterator
from dataclasses import dataclass
from .models import Movie, TextChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingEngine:
    """
    Advanced text chunking with multiple strategies.
    
    Supports:
    - Fixed-size chunking with overlap
    - Semantic (paragraph-based) chunking
    - Movie-aware chunking (separates overview from metadata)
    - Hierarchical chunking (parent-child relationships)
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_length: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        logger.info(f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_movies(self, movies: List[Movie], 
                     strategy: str = "movie_aware") -> List[TextChunk]:
        """
        Chunk multiple movies into text segments.
        
        Args:
            movies: List of Movie objects
            strategy: Chunking strategy ('fixed', 'semantic', 'movie_aware', 'hierarchical')
        
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        for movie in movies:
            try:
                if strategy == "fixed":
                    chunks = self._fixed_chunks(movie)
                elif strategy == "semantic":
                    chunks = self._semantic_chunks(movie)
                elif strategy == "movie_aware":
                    chunks = self._movie_aware_chunks(movie)
                elif strategy == "hierarchical":
                    chunks = self._hierarchical_chunks(movie)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Chunking failed for '{movie.series_title}': {e}")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(movies)} movies "
                   f"using '{strategy}' strategy")
        return all_chunks
    
    def _fixed_chunks(self, movie: Movie) -> List[TextChunk]:
        """Simple fixed-size chunking with overlap."""
        text = movie.full_text
        chunks = []
        
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if len(chunk_text) >= self.min_chunk_length:
                chunk_id = f"{movie.doc_id}_fixed_{chunk_idx}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    doc_id=movie.doc_id,
                    content=chunk_text,
                    source_title=movie.series_title,
                    chunk_type="full_text",
                    token_estimate=self._estimate_tokens(chunk_text),
                    metadata={
                        "start_char": start,
                        "end_char": end,
                        "strategy": "fixed"
                    }
                ))
            
            start += self.chunk_size - self.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def _semantic_chunks(self, movie: Movie) -> List[TextChunk]:
        """Paragraph-aware semantic chunking."""
        paragraphs = re.split(r'\n\s*\n|\n', movie.full_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    chunk_id = f"{movie.doc_id}_sem_{chunk_idx}"
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        doc_id=movie.doc_id,
                        content=chunk_text,
                        source_title=movie.series_title,
                        chunk_type="semantic",
                        token_estimate=self._estimate_tokens(chunk_text),
                        metadata={"strategy": "semantic", "paragraphs": len(current_chunk)}
                    ))
                
                # Keep last paragraph for overlap
                current_chunk = [para]
                current_length = para_len
                chunk_idx += 1
            else:
                current_chunk.append(para)
                current_length += para_len
        
        # Add remaining
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                chunk_id = f"{movie.doc_id}_sem_{chunk_idx}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    doc_id=movie.doc_id,
                    content=chunk_text,
                    source_title=movie.series_title,
                    chunk_type="semantic",
                    token_estimate=self._estimate_tokens(chunk_text),
                    metadata={"strategy": "semantic", "paragraphs": len(current_chunk)}
                ))
        
        return chunks
    
    def _movie_aware_chunks(self, movie: Movie) -> List[TextChunk]:
        """
        Movie-aware chunking: creates structured chunks from different movie aspects.
        Separates overview, cast info, and metadata into distinct chunks.
        """
        chunks = []
        
        # Chunk 1: Overview (the plot summary)
        if movie.overview and len(movie.overview) >= 50:
            overview_chunks = self._chunk_text(
                movie.overview, movie, "overview", max_tokens=self.chunk_size
            )
            chunks.extend(overview_chunks)
        
        # Chunk 2: Cast and crew information
        cast_text = self._build_cast_text(movie)
        if cast_text:
            chunks.append(TextChunk(
                chunk_id=f"{movie.doc_id}_cast",
                doc_id=movie.doc_id,
                content=cast_text,
                source_title=movie.series_title,
                chunk_type="metadata",
                token_estimate=self._estimate_tokens(cast_text),
                metadata={
                    "director": movie.director,
                    "cast": movie.cast_list,
                    "strategy": "movie_aware"
                }
            ))
        
        # Chunk 3: Genre and classification
        genre_text = f"{movie.series_title} is a {', '.join(movie.genre_list)} movie. "
        genre_text += f"It was released in {movie.released_year} ({movie.era}). "
        genre_text += f"Rated {movie.imdb_rating}/10 on IMDb. "
        if movie.meta_score:
            genre_text += f"Metacritic score: {movie.meta_score}. "
        genre_text += f"Popularity: {movie.popularity_tier} ({movie.no_of_votes:,} votes)."
        
        chunks.append(TextChunk(
            chunk_id=f"{movie.doc_id}_genre",
            doc_id=movie.doc_id,
            content=genre_text,
            source_title=movie.series_title,
            chunk_type="metadata",
            token_estimate=self._estimate_tokens(genre_text),
            metadata={
                "genres": movie.genre_list,
                "year": movie.released_year,
                "rating": movie.imdb_rating,
                "strategy": "movie_aware"
            }
        ))
        
        # Chunk 4: Full combined text (for comprehensive retrieval)
        if len(movie.full_text) <= self.chunk_size * 2:
            chunks.append(TextChunk(
                chunk_id=f"{movie.doc_id}_full",
                doc_id=movie.doc_id,
                content=movie.full_text,
                source_title=movie.series_title,
                chunk_type="full_text",
                token_estimate=self._estimate_tokens(movie.full_text),
                metadata={"strategy": "movie_aware", "complete": True}
            ))
        
        return chunks
    
    def _hierarchical_chunks(self, movie: Movie) -> List[TextChunk]:
        """
        Hierarchical chunking: parent (summary) -> child (details).
        Creates chunks at different granularity levels.
        """
        chunks = []
        
        # Parent chunk: Movie summary
        summary = (f"{movie.series_title} ({movie.released_year}) - "
                  f"Directed by {movie.director}. "
                  f"Starring {', '.join(movie.cast_list[:2])}. "
                  f"Genres: {', '.join(movie.genre_list)}. "
                  f"IMDb: {movie.imdb_rating}/10. "
                  f"{movie.overview[:200]}...")
        
        chunks.append(TextChunk(
            chunk_id=f"{movie.doc_id}_parent",
            doc_id=movie.doc_id,
            content=summary,
            source_title=movie.series_title,
            chunk_type="summary",
            token_estimate=self._estimate_tokens(summary),
            metadata={"level": "parent", "strategy": "hierarchical"}
        ))
        
        # Child chunks: detailed sections
        child_chunks = self._movie_aware_chunks(movie)
        for chunk in child_chunks:
            chunk.chunk_id = f"{chunk.chunk_id}_child"
            chunk.chunk_type = f"child_{chunk.chunk_type}"
            chunk.metadata["level"] = "child"
            chunk.metadata["parent_id"] = f"{movie.doc_id}_parent"
        
        chunks.extend(child_chunks)
        return chunks
    
    def _chunk_text(self, text: str, movie: Movie, 
                    chunk_type: str, max_tokens: int = 512) -> List[TextChunk]:
        """Helper to chunk a long text field."""
        if len(text) <= max_tokens:
            return [TextChunk(
                chunk_id=f"{movie.doc_id}_{chunk_type}_0",
                doc_id=movie.doc_id,
                content=text,
                source_title=movie.series_title,
                chunk_type=chunk_type,
                token_estimate=self._estimate_tokens(text),
                metadata={"strategy": "movie_aware"}
            )]
        
        # Sentence-level splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        idx = 0
        
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > max_tokens and current:
                content = " ".join(current)
                chunks.append(TextChunk(
                    chunk_id=f"{movie.doc_id}_{chunk_type}_{idx}",
                    doc_id=movie.doc_id,
                    content=content,
                    source_title=movie.series_title,
                    chunk_type=chunk_type,
                    token_estimate=self._estimate_tokens(content),
                    metadata={"strategy": "movie_aware", "sentences": len(current)}
                ))
                current = [sent]
                current_len = sent_len
                idx += 1
            else:
                current.append(sent)
                current_len += sent_len
        
        if current:
            content = " ".join(current)
            chunks.append(TextChunk(
                chunk_id=f"{movie.doc_id}_{chunk_type}_{idx}",
                doc_id=movie.doc_id,
                content=content,
                source_title=movie.series_title,
                chunk_type=chunk_type,
                token_estimate=self._estimate_tokens(content),
                metadata={"strategy": "movie_aware"}
            ))
        
        return chunks
    
    def _build_cast_text(self, movie: Movie) -> str:
        """Build cast and crew descriptive text."""
        parts = [f"{movie.series_title} was directed by {movie.director}."]
        
        if movie.cast_list:
            if len(movie.cast_list) == 1:
                parts.append(f"The star is {movie.cast_list[0]}.")
            elif len(movie.cast_list) == 2:
                parts.append(f"Starring {movie.cast_list[0]} and {movie.cast_list[1]}.")
            else:
                stars = ", ".join(movie.cast_list[:-1]) + f" and {movie.cast_list[-1]}"
                parts.append(f"Starring {stars}.")
        
        return " ".join(parts)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ~ 4 chars for English)."""
        return len(text) // 4
    
    def chunk_stream(self, movies: List[Movie], 
                     strategy: str = "movie_aware") -> Iterator[TextChunk]:
        """
        Stream chunks one at a time (memory efficient for large datasets).
        
        Args:
            movies: List of movies to chunk
            strategy: Chunking strategy
        
        Yields:
            Individual TextChunk objects
        """
        for movie in movies:
            chunks = self.chunk_movies([movie], strategy)
            for chunk in chunks:
                yield chunk
