"""
Entity Extraction
-----------------
Extract named entities and relationships from movie data.
Prepares structured data for graph construction.
"""
import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from .models import Movie
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    entity_type: str  # 'PERSON', 'MOVIE', 'GENRE', 'YEAR', 'ORG'
    source_doc: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.name.lower(), self.entity_type))
    
    def __eq__(self, other):
        return (isinstance(other, Entity) and 
                self.name.lower() == other.name.lower() and
                self.entity_type == other.entity_type)


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    source: Entity
    target: Entity
    relation_type: str  # 'DIRECTED', 'ACTED_IN', 'BELONGS_TO', 'RELEASED_IN'
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source.name,
            "source_type": self.source.entity_type,
            "target": self.target.name,
            "target_type": self.target.entity_type,
            "relation": self.relation_type,
            "weight": self.weight,
        }


class EntityExtractor:
    """
    Extract entities and relationships from movie dataset.
    Used for GraphRAG construction.
    """
    
    RELATION_TYPES = {
        "DIRECTED": "{director} directed {movie}",
        "STARRED_IN": "{actor} starred in {movie}",
        "BELONGS_TO_GENRE": "{movie} is a {genre} film",
        "RELEASED_IN": "{movie} was released in {year}",
        "SIMILAR_GENRE": "{movie1} and {movie2} share genre {genre}",
        "SAME_ERA": "{movie1} and {movie2} are from the {era}",
        "COLLABORATED": "{person1} and {person2} worked together",
    }
    
    def __init__(self):
        self.entities: Dict[str, Set[Entity]] = defaultdict(set)
        self.relationships: List[Relationship] = []
        logger.info("EntityExtractor initialized")
    
    def extract_from_movies(self, movies: List[Movie]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract all entities and relationships from movie list.
        
        Returns:
            Tuple of (entities, relationships)
        """
        logger.info(f"Extracting entities from {len(movies)} movies...")
        
        for movie in movies:
            self._extract_movie_entities(movie)
        
        # Cross-movie relationships
        self._extract_cross_movie_relationships(movies)
        
        all_entities = set()
        for entity_set in self.entities.values():
            all_entities.update(entity_set)
        
        logger.info(f"Extracted {len(all_entities)} entities, {len(self.relationships)} relationships")
        return list(all_entities), self.relationships
    
    def _extract_movie_entities(self, movie: Movie) -> None:
        """Extract entities from a single movie."""
        doc_id = movie.doc_id
        
        # Movie entity
        movie_entity = Entity(
            name=movie.series_title,
            entity_type="MOVIE",
            source_doc=doc_id,
            metadata={"rating": movie.imdb_rating, "year": movie.released_year}
        )
        self.entities["MOVIE"].add(movie_entity)
        
        # Director entity
        if movie.director:
            director = Entity(
                name=movie.director,
                entity_type="PERSON",
                source_doc=doc_id,
                metadata={"role": "director"}
            )
            self.entities["PERSON"].add(director)
            self.relationships.append(Relationship(
                source=director,
                target=movie_entity,
                relation_type="DIRECTED",
                weight=1.0
            ))
        
        # Cast entities
        for actor_name in movie.cast_list:
            if actor_name:
                actor = Entity(
                    name=actor_name,
                    entity_type="PERSON",
                    source_doc=doc_id,
                    metadata={"role": "actor"}
                )
                self.entities["PERSON"].add(actor)
                self.relationships.append(Relationship(
                    source=actor,
                    target=movie_entity,
                    relation_type="STARRED_IN",
                    weight=1.0
                ))
        
        # Genre entities
        for genre_name in movie.genre_list:
            if genre_name:
                genre = Entity(
                    name=genre_name,
                    entity_type="GENRE",
                    source_doc=doc_id
                )
                self.entities["GENRE"].add(genre)
                self.relationships.append(Relationship(
                    source=movie_entity,
                    target=genre,
                    relation_type="BELONGS_TO_GENRE",
                    weight=1.0
                ))
        
        # Year entity
        if movie.released_year > 0:
            year_entity = Entity(
                name=str(movie.released_year),
                entity_type="YEAR",
                source_doc=doc_id,
                metadata={"era": movie.era}
            )
            self.entities["YEAR"].add(year_entity)
            self.relationships.append(Relationship(
                source=movie_entity,
                target=year_entity,
                relation_type="RELEASED_IN",
                weight=0.8
            ))
    
    def _extract_cross_movie_relationships(self, movies: List[Movie]) -> None:
        """Extract relationships between different movies."""
        # Group by genre
        genre_movies: Dict[str, List[Movie]] = defaultdict(list)
        # Group by era
        era_movies: Dict[str, List[Movie]] = defaultdict(list)
        # Group by director
        director_movies: Dict[str, List[Movie]] = defaultdict(list)
        # Group by actor
        actor_movies: Dict[str, List[Movie]] = defaultdict(list)
        
        for movie in movies:
            for genre in movie.genre_list:
                genre_movies[genre].append(movie)
            era_movies[movie.era].append(movie)
            if movie.director:
                director_movies[movie.director].append(movie)
            for actor in movie.cast_list:
                actor_movies[actor].append(movie)
        
        # Genre similarity edges (movies sharing genres)
        for genre, movie_list in genre_movies.items():
            if len(movie_list) > 1:
                for i in range(min(len(movie_list), 50)):  # Limit to prevent explosion
                    for j in range(i + 1, min(len(movie_list), 50)):
                        m1, m2 = movie_list[i], movie_list[j]
                        self.relationships.append(Relationship(
                            source=Entity(name=m1.series_title, entity_type="MOVIE"),
                            target=Entity(name=m2.series_title, entity_type="MOVIE"),
                            relation_type="SIMILAR_GENRE",
                            weight=0.5,
                            metadata={"shared_genre": genre}
                        ))
        
        # Era similarity
        for era, movie_list in era_movies.items():
            if len(movie_list) > 1 and len(movie_list) <= 100:
                for i in range(min(len(movie_list), 30)):
                    for j in range(i + 1, min(len(movie_list), 30)):
                        m1, m2 = movie_list[i], movie_list[j]
                        self.relationships.append(Relationship(
                            source=Entity(name=m1.series_title, entity_type="MOVIE"),
                            target=Entity(name=m2.series_title, entity_type="MOVIE"),
                            relation_type="SAME_ERA",
                            weight=0.3,
                            metadata={"era": era}
                        ))
        
        # Director collaborations (same director = related movies)
        for director, movie_list in director_movies.items():
            if len(movie_list) > 1:
                for i in range(len(movie_list)):
                    for j in range(i + 1, len(movie_list)):
                        m1, m2 = movie_list[i], movie_list[j]
                        self.relationships.append(Relationship(
                            source=Entity(name=m1.series_title, entity_type="MOVIE"),
                            target=Entity(name=m2.series_title, entity_type="MOVIE"),
                            relation_type="SAME_DIRECTOR",
                            weight=0.7,
                            metadata={"director": director}
                        ))
    
    def get_entity_stats(self) -> Dict[str, int]:
        """Get counts of each entity type."""
        return {etype: len(entities) for etype, entities in self.entities.items()}
    
    def get_relationship_stats(self) -> Dict[str, int]:
        """Get counts of each relationship type."""
        counts = defaultdict(int)
        for rel in self.relationships:
            counts[rel.relation_type] += 1
        return dict(counts)
