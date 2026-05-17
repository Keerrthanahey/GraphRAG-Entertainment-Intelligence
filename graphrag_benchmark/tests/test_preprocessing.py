"""
Preprocessing Pipeline Tests
----------------------------
Unit tests for CSV loading, chunking, and entity extraction.
"""
import os
import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.csv_loader import CSVLoader
from src.preprocessing.chunker import ChunkingEngine
from src.preprocessing.entity_extractor import EntityExtractor
from src.preprocessing.models import Movie, TextChunk


class TestCSVLoader(unittest.TestCase):
    """Test CSV loading and validation."""
    
    def setUp(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.raw_dir = self.data_dir / "raw"
    
    def test_loader_initialization(self):
        """Test loader can be initialized."""
        csv_path = self.raw_dir / "test.csv"
        loader = CSVLoader(str(csv_path))
        self.assertEqual(loader.stats["total_rows"], 0)
    
    def test_movie_object_creation(self):
        """Test Movie dataclass creation."""
        movie = Movie(
            series_title="The Shawshank Redemption",
            released_year=1994,
            director="Frank Darabont",
            genre="Drama",
            imdb_rating=9.3,
            overview="Two imprisoned men bond over years.",
            star1="Tim Robbins",
            star2="Morgan Freeman"
        )
        
        self.assertEqual(movie.series_title, "The Shawshank Redemption")
        self.assertEqual(movie.released_year, 1994)
        self.assertEqual(movie.director, "Frank Darabont")
        self.assertEqual(len(movie.genre_list), 1)
        self.assertEqual(len(movie.cast_list), 2)
        self.assertTrue(len(movie.doc_id) > 0)
        self.assertTrue(len(movie.full_text) > 0)
    
    def test_movie_era(self):
        """Test era classification."""
        m1 = Movie(series_title="Old", released_year=1972)
        m2 = Movie(series_title="90s", released_year=1994)
        m3 = Movie(series_title="Recent", released_year=2023)
        
        self.assertEqual(m1.era, "1970s")
        self.assertEqual(m2.era, "1990s")
        self.assertEqual(m3.era, "2020s")


class TestChunkingEngine(unittest.TestCase):
    """Test text chunking strategies."""
    
    def setUp(self):
        self.chunker = ChunkingEngine(chunk_size=100, chunk_overlap=20, min_chunk_length=10)
        self.test_movie = Movie(
            series_title="Test Movie",
            released_year=2020,
            director="Test Director",
            genre="Action, Adventure",
            imdb_rating=8.5,
            overview="This is a test movie about testing. It has many exciting scenes and great acting.",
            star1="Actor One",
            star2="Actor Two"
        )
    
    def test_movie_aware_chunking(self):
        """Test movie-aware chunking produces multiple chunks."""
        chunks = self.chunker._movie_aware_chunks(self.test_movie)
        
        self.assertTrue(len(chunks) >= 2)
        
        # Check chunk types
        types = [c.chunk_type for c in chunks]
        self.assertIn("metadata", types)
        
        # Check all chunks have IDs
        for chunk in chunks:
            self.assertTrue(len(chunk.chunk_id) > 0)
            self.assertTrue(len(chunk.content) > 0)
            self.assertEqual(chunk.source_title, "Test Movie")
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        chunks = self.chunker._fixed_chunks(self.test_movie)
        
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 100 + 20)  # chunk_size + some tolerance
    
    def test_semantic_chunking(self):
        """Test semantic (paragraph) chunking."""
        chunks = self.chunker._semantic_chunks(self.test_movie)
        
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIsInstance(chunk.content, str)
            self.assertTrue(len(chunk.content) > 0)


class TestEntityExtractor(unittest.TestCase):
    """Test entity and relationship extraction."""
    
    def setUp(self):
        self.extractor = EntityExtractor()
        self.movies = [
            Movie(
                series_title="Movie A",
                released_year=2020,
                director="Director X",
                genre="Action",
                imdb_rating=8.0,
                overview="Great action movie.",
                star1="Actor 1",
                star2="Actor 2"
            ),
            Movie(
                series_title="Movie B",
                released_year=2021,
                director="Director X",  # Same director
                genre="Action",  # Same genre
                imdb_rating=7.5,
                overview="Another action film.",
                star1="Actor 2",  # Shared actor
                star2="Actor 3"
            ),
        ]
    
    def test_entity_extraction(self):
        """Test entities are extracted correctly."""
        entities, relationships = self.extractor.extract_from_movies(self.movies)
        
        self.assertTrue(len(entities) > 0)
        
        # Check entity types
        entity_types = set(e.entity_type for e in entities)
        self.assertIn("MOVIE", entity_types)
        self.assertIn("PERSON", entity_types)
        self.assertIn("GENRE", entity_types)
        self.assertIn("YEAR", entity_types)
    
    def test_relationship_extraction(self):
        """Test relationships are created."""
        entities, relationships = self.extractor.extract_from_movies(self.movies)
        
        self.assertTrue(len(relationships) > 0)
        
        # Check relationship types
        rel_types = set(r.relation_type for r in relationships)
        self.assertIn("DIRECTED", rel_types)
        self.assertIn("STARRED_IN", rel_types)
        self.assertIn("BELONGS_TO_GENRE", rel_types)
    
    def test_cross_movie_relationships(self):
        """Test cross-movie relationships (same director, genre)."""
        entities, relationships = self.extractor.extract_from_movies(self.movies)
        
        rel_types = [r.relation_type for r in relationships]
        self.assertIn("SAME_DIRECTOR", rel_types)
        self.assertIn("SIMILAR_GENRE", rel_types)


class TestDataSplitter(unittest.TestCase):
    """Test dataset splitting."""
    
    def test_split_ratios(self):
        """Test split produces correct ratios."""
        from src.preprocessing.csv_loader import DataSplitter
        
        movies = [
            Movie(series_title=f"Movie {i}", released_year=2000 + i)
            for i in range(100)
        ]
        
        train, val, test = DataSplitter.split(movies, train_ratio=0.7, val_ratio=0.15)
        
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
    
    def test_no_overlap(self):
        """Test splits don't overlap."""
        from src.preprocessing.csv_loader import DataSplitter
        
        movies = [
            Movie(series_title=f"Movie {i}", released_year=2000 + i)
            for i in range(50)
        ]
        
        train, val, test = DataSplitter.split(movies)
        
        train_ids = {m.doc_id for m in train}
        val_ids = {m.doc_id for m in val}
        test_ids = {m.doc_id for m in test}
        
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)


def create_test_csv():
    """Create a minimal test CSV file."""
    import csv
    
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = raw_dir / "test_movies.csv"
    
    sample_data = [
        ["Poster_Link", "Series_Title", "Released_Year", "Certificate", "Runtime", 
         "Genre", "IMDB_Rating", "Overview", "Meta_score", "Director", 
         "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross"],
        ["http://example.com/1.jpg", "The Shawshank Redemption", "1994", "R", "142 min",
         "Drama", "9.3", "Two imprisoned men bond over years.", "80", "Frank Darabont",
         "Tim Robbins", "Morgan Freeman", "Bob Gunton", "William Sadler", "2800000", "28341469"],
        ["http://example.com/2.jpg", "The Godfather", "1972", "R", "175 min",
         "Crime, Drama", "9.2", "The aging patriarch of an organized crime dynasty transfers control.", "100", "Francis Ford Coppola",
         "Marlon Brando", "Al Pacino", "James Caan", "Diane Keaton", "2000000", "134966411"],
        ["http://example.com/3.jpg", "The Dark Knight", "2008", "PG-13", "152 min",
         "Action, Crime, Drama", "9.0", "Batman faces the Joker.", "84", "Christopher Nolan",
         "Christian Bale", "Heath Ledger", "Aaron Eckhart", "Michael Caine", "2800000", "534858444"],
        ["http://example.com/4.jpg", "Pulp Fiction", "1994", "R", "154 min",
         "Crime, Drama", "8.9", "The lives of two mob hitmen, a boxer, and others intertwine.", "94", "Quentin Tarantino",
         "John Travolta", "Uma Thurman", "Samuel L. Jackson", "Bruce Willis", "2200000", "107928762"],
        ["http://example.com/5.jpg", "Inception", "2010", "PG-13", "148 min",
         "Action, Adventure, Sci-Fi", "8.8", "A thief who steals corporate secrets is given a final job.", "74", "Christopher Nolan",
         "Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Tom Hardy", "2500000", "292576195"],
    ]
    
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    
    return test_file


if __name__ == "__main__":
    # Create test data
    test_csv = create_test_csv()
    print(f"Created test CSV: {test_csv}")
    
    # Run tests
    unittest.main(verbosity=2)
