"""
Data Models for Entertainment Dataset
-------------------------------------
Pydantic models for type-safe data handling throughout the pipeline.
"""
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Movie:
    """Represents a movie entity from the IMDb dataset."""
    poster_link: str = ""
    series_title: str = ""
    released_year: int = 0
    certificate: str = ""
    runtime: str = ""
    genre: str = ""
    imdb_rating: float = 0.0
    overview: str = ""
    meta_score: Optional[float] = None
    director: str = ""
    star1: str = ""
    star2: str = ""
    star3: str = ""
    star4: str = ""
    no_of_votes: int = 0
    gross: Optional[str] = None
    
    # Derived fields (populated during preprocessing)
    doc_id: str = ""
    runtime_minutes: int = 0
    genre_list: List[str] = field(default_factory=list)
    cast_list: List[str] = field(default_factory=list)
    full_text: str = ""
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate document ID and derived fields after initialization."""
        if not self.doc_id:
            self.doc_id = self._generate_id()
        if not self.genre_list and self.genre:
            self.genre_list = [g.strip() for g in self.genre.split(",")]
        if not self.cast_list:
            stars = [self.star1, self.star2, self.star3, self.star4]
            self.cast_list = [s.strip() for s in stars if s and s.strip()]
        if not self.full_text:
            self.full_text = self._build_full_text()
        if not self.runtime_minutes and self.runtime:
            self.runtime_minutes = self._parse_runtime()
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        content = f"{self.series_title}_{self.released_year}_{self.director}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _build_full_text(self) -> str:
        """Build searchable full text representation."""
        parts = [
            f"Title: {self.series_title}",
            f"Year: {self.released_year}",
            f"Director: {self.director}",
            f"Cast: {', '.join(self.cast_list)}",
            f"Genre: {', '.join(self.genre_list)}",
            f"Rating: {self.imdb_rating}/10",
            f"Overview: {self.overview}",
        ]
        if self.meta_score:
            parts.append(f"Meta Score: {self.meta_score}")
        return "\n".join(parts)
    
    def _parse_runtime(self) -> int:
        """Parse runtime string to minutes."""
        try:
            return int(self.runtime.replace(" min", "").strip())
        except (ValueError, AttributeError):
            return 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "series_title": self.series_title,
            "released_year": self.released_year,
            "certificate": self.certificate,
            "runtime_minutes": self.runtime_minutes,
            "genre": self.genre,
            "genre_list": self.genre_list,
            "imdb_rating": self.imdb_rating,
            "overview": self.overview,
            "meta_score": self.meta_score,
            "director": self.director,
            "cast_list": self.cast_list,
            "star1": self.star1,
            "star2": self.star2,
            "star3": self.star3,
            "star4": self.star4,
            "no_of_votes": self.no_of_votes,
            "gross": self.gross,
            "full_text": self.full_text,
        }
    
    @property
    def era(self) -> str:
        """Determine movie era from release year."""
        if self.released_year >= 2020:
            return "2020s"
        elif self.released_year >= 2010:
            return "2010s"
        elif self.released_year >= 2000:
            return "2000s"
        elif self.released_year >= 1990:
            return "1990s"
        elif self.released_year >= 1980:
            return "1980s"
        elif self.released_year >= 1970:
            return "1970s"
        elif self.released_year >= 1960:
            return "1960s"
        else:
            return "Classic"
    
    @property
    def popularity_tier(self) -> str:
        """Classify movie by popularity."""
        if self.no_of_votes >= 500_000:
            return "Blockbuster"
        elif self.no_of_votes >= 100_000:
            return "Popular"
        elif self.no_of_votes >= 10_000:
            return "Moderate"
        else:
            return "Niche"


@dataclass
class TextChunk:
    """Represents a chunk of text for embedding."""
    chunk_id: str
    doc_id: str
    content: str
    source_title: str
    chunk_type: str  # 'overview', 'metadata', 'full_text'
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source_title": self.source_title,
            "chunk_type": self.chunk_type,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }
