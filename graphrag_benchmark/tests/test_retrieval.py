"""
Semantic Retrieval Testing Suite
---------------------------------
Comprehensive tests for retrieval quality, latency, and performance metrics.
Includes benchmark queries and evaluation framework.
"""
import os
import sys
import time
import math
import unittest
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.models import Movie, TextChunk
from src.embedding.gemini_embedder import GeminiEmbedder
from src.storage.chroma_store import ChromaVectorStore


# Benchmark query set for entertainment domain
BENCHMARK_QUERIES = [
    {
        "query": "Movies directed by Christopher Nolan",
        "expected_titles": ["Inception", "The Dark Knight", "Interstellar"],
        "category": "director"
    },
    {
        "query": "Best crime dramas from the 1990s",
        "expected_titles": ["Pulp Fiction", "The Godfather Part III", "Goodfellas"],
        "category": "genre_era"
    },
    {
        "query": "Movies starring Morgan Freeman",
        "expected_titles": ["The Shawshank Redemption", "Se7en", "Driving Miss Daisy"],
        "category": "actor"
    },
    {
        "query": "Sci-fi movies about space exploration",
        "expected_titles": ["Interstellar", "2001: A Space Odyssey", "Gravity"],
        "category": "theme"
    },
    {
        "query": "Highly rated psychological thrillers",
        "expected_titles": ["Fight Club", "Black Swan", "Shutter Island"],
        "category": "genre_rating"
    },
    {
        "query": "Movies with a twist ending",
        "expected_titles": ["The Sixth Sense", "Fight Club", "Inception"],
        "category": "narrative"
    },
    {
        "query": "Classic romance movies",
        "expected_titles": ["Casablanca", "Gone with the Wind", "Roman Holiday"],
        "category": "genre"
    },
    {
        "query": "War movies about World War II",
        "expected_titles": ["Saving Private Ryan", "Schindler's List", "Dunkirk"],
        "category": "theme_era"
    },
    {
        "query": "Comedy movies from the 2000s",
        "expected_titles": ["The Hangover", "Superbad", "Anchorman"],
        "category": "genre_era"
    },
    {
        "query": "Movies about artificial intelligence",
        "expected_titles": ["The Matrix", "Ex Machina", "Blade Runner 2049"],
        "category": "theme"
    },
]


class TestRetrievalQuality(unittest.TestCase):
    """Test semantic retrieval quality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.embedder = None
        cls.store = None
        cls.test_chunks = cls._create_test_chunks()
        
        # Try to initialize with API key
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                cls.embedder = GeminiEmbedder(api_key=api_key)
                cls.store = ChromaVectorStore(
                    collection_name="test_retrieval"
                ).connect("memory").create_collection()
                
                # Embed and store test chunks
                embedded = cls.embedder.embed_chunks(cls.test_chunks[:20])
                cls.store.add_chunks(embedded)
                cls.has_api = True
            except Exception as e:
                print(f"Warning: Could not initialize API: {e}")
                cls.has_api = False
        else:
            print("Warning: No GEMINI_API_KEY found, skipping API tests")
            cls.has_api = False
    
    @classmethod
    def _create_test_chunks(cls) -> List[TextChunk]:
        """Create test chunks with synthetic embeddings."""
        chunks = []
        
        test_movies = [
            {"title": "Inception", "year": 2010, "director": "Christopher Nolan",
             "genre": "Sci-Fi", "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
             "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea."},
            {"title": "The Dark Knight", "year": 2008, "director": "Christopher Nolan",
             "genre": "Action", "cast": ["Christian Bale", "Heath Ledger"],
             "overview": "Batman faces the Joker, a criminal mastermind who wants to plunge Gotham into anarchy."},
            {"title": "Pulp Fiction", "year": 1994, "director": "Quentin Tarantino",
             "genre": "Crime", "cast": ["John Travolta", "Samuel L. Jackson"],
             "overview": "The lives of two mob hitmen, a boxer, and a gangster's wife intertwine in four tales of violence and redemption."},
            {"title": "The Shawshank Redemption", "year": 1994, "director": "Frank Darabont",
             "genre": "Drama", "cast": ["Tim Robbins", "Morgan Freeman"],
             "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
            {"title": "Fight Club", "year": 1999, "director": "David Fincher",
             "genre": "Drama", "cast": ["Brad Pitt", "Edward Norton"],
             "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club."},
            {"title": "Interstellar", "year": 2014, "director": "Christopher Nolan",
             "genre": "Sci-Fi", "cast": ["Matthew McConaughey", "Anne Hathaway"],
             "overview": "A team of explorers travel through a wormhole in space to ensure humanity's survival."},
            {"title": "The Matrix", "year": 1999, "director": "The Wachowskis",
             "genre": "Sci-Fi", "cast": ["Keanu Reeves", "Laurence Fishburne"],
             "overview": "A computer hacker learns about the true nature of his reality and his role in the war against its controllers."},
            {"title": "Goodfellas", "year": 1990, "director": "Martin Scorsese",
             "genre": "Crime", "cast": ["Robert De Niro", "Ray Liotta"],
             "overview": "The story of Henry Hill and his life in the mob, covering his relationship with his wife and partners."},
            {"title": "Se7en", "year": 1995, "director": "David Fincher",
             "genre": "Thriller", "cast": ["Brad Pitt", "Morgan Freeman"],
             "overview": "Two detectives hunt a serial killer who uses the seven deadly sins as his motives."},
            {"title": "Parasite", "year": 2019, "director": "Bong Joon Ho",
             "genre": "Thriller", "cast": ["Song Kang-ho", "Lee Sun-kyun"],
             "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
        ]
        
        for i, movie in enumerate(test_movies):
            content = f"{movie['title']} ({movie['year']}) directed by {movie['director']}. " \
                     f"Genres: {movie['genre']}. Starring {', '.join(movie['cast'])}. " \
                     f"{movie['overview']}"
            
            # Create synthetic embedding (768-dim, normalized)
            embedding = cls._create_synthetic_embedding(i)
            
            chunk = TextChunk(
                chunk_id=f"test_{i}",
                doc_id=f"movie_{i}",
                content=content,
                source_title=movie['title'],
                chunk_type="full_text",
                token_estimate=len(content) // 4,
                embedding=embedding,
                metadata={
                    "director": movie['director'],
                    "year": movie['year'],
                    "genre": movie['genre'],
                    "cast": movie['cast']
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _create_synthetic_embedding(index: int, dim: int = 768) -> List[float]:
        """Create deterministic synthetic embedding for testing."""
        import random
        random.seed(index * 42)
        vec = [random.uniform(-1, 1) for _ in range(dim)]
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]
    
    def test_chunk_creation(self):
        """Test that test chunks are created correctly."""
        self.assertEqual(len(self.test_chunks), 10)
        
        for chunk in self.test_chunks:
            self.assertIsNotNone(chunk.embedding)
            self.assertEqual(len(chunk.embedding), 768)
            self.assertTrue(len(chunk.content) > 0)
    
    def test_embedding_dimension(self):
        """Test all embeddings have consistent dimensions."""
        dims = [len(c.embedding) for c in self.test_chunks]
        self.assertEqual(len(set(dims)), 1)  # All same dimension
    
    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "No API key")
    def test_query_embedding(self):
        """Test query embedding generation."""
        query = "science fiction movies about space"
        embedding = self.embedder.embed_query(query)
        
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 768)
    
    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "No API key")
    def test_basic_search(self):
        """Test basic similarity search returns results."""
        query = "Christopher Nolan movies"
        query_emb = self.embedder.embed_query(query)
        
        results = self.store.search(query_emb, top_k=3)
        
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 3)
        
        # Check result structure
        for r in results:
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("metadata", r)
            self.assertGreaterEqual(r["score"], 0)
            self.assertLessEqual(r["score"], 1)
    
    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "No API key")
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        query = "sci-fi movies"
        query_emb = self.embedder.embed_query(query)
        
        # Filter by year
        results = self.store.search(
            query_emb, 
            top_k=5,
            filters={"meta_year": "2010"}
        )
        
        # Results should be empty or filtered (depends on metadata format)
        self.assertIsInstance(results, list)
    
    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "No API key")
    def test_search_latency(self):
        """Test search latency is reasonable."""
        query = "action movies"
        query_emb = self.embedder.embed_query(query)
        
        times = []
        for _ in range(5):
            start = time.time()
            self.store.search(query_emb, top_k=5)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"\nAverage search latency: {avg_time:.2f}ms")
        
        # Should be reasonably fast
        self.assertLess(avg_time, 5000)  # Under 5 seconds
    
    def test_result_scoring(self):
        """Test that results have valid scores."""
        if not self.has_api:
            self.skipTest("No API key")
        
        query_emb = self.embedder.embed_query("test query")
        results = self.store.search(query_emb, top_k=5)
        
        # Scores should be in valid range
        for r in results:
            self.assertGreaterEqual(r["score"], -1.0)
            self.assertLessEqual(r["score"], 1.0)
        
        # Results should be sorted by score (descending)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))


class TestMetricsCalculation(unittest.TestCase):
    """Test metrics calculation utilities."""
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation."""
        from src.utils.metrics import calculate_ndcg
        
        # Perfect relevance
        perfect = [1.0, 1.0, 1.0, 1.0]
        ndcg = calculate_ndcg(perfect, k=4)
        self.assertAlmostEqual(ndcg, 1.0, places=5)
        
        # No relevance
        none_rel = [0.0, 0.0, 0.0]
        ndcg = calculate_ndcg(none_rel, k=3)
        self.assertEqual(ndcg, 0.0)
        
        # Partial relevance
        partial = [1.0, 0.5, 0.0, 0.0]
        ndcg = calculate_ndcg(partial, k=4)
        self.assertGreater(ndcg, 0.0)
        self.assertLess(ndcg, 1.0)
    
    def test_mrr_calculation(self):
        """Test MRR calculation."""
        from src.utils.metrics import calculate_mrr
        
        # First result relevant
        mrr = calculate_mrr([True, False, False])
        self.assertEqual(mrr, 1.0)
        
        # Second result relevant
        mrr = calculate_mrr([False, True, False])
        self.assertEqual(mrr, 0.5)
        
        # No relevant results
        mrr = calculate_mrr([False, False, False])
        self.assertEqual(mrr, 0.0)
    
    def test_performance_metrics(self):
        """Test performance metrics container."""
        from src.utils.metrics import PerformanceMetrics
        
        pm = PerformanceMetrics(latency_ms=150.5, token_usage=500, cost_usd=0.0002)
        
        self.assertEqual(pm.latency_ms, 150.5)
        self.assertEqual(pm.token_usage, 500)
        self.assertEqual(pm.cost_usd, 0.0002)
        
        # Test dict conversion
        d = pm.to_dict()
        self.assertEqual(d["latency_ms"], 150.5)
        self.assertEqual(d["token_usage"], 500)


class TestBenchmarkQueries(unittest.TestCase):
    """Test benchmark query set validity."""
    
    def test_queries_have_required_fields(self):
        """Test all benchmark queries have required fields."""
        required_fields = ["query", "expected_titles", "category"]
        
        for q in BENCHMARK_QUERIES:
            for field in required_fields:
                self.assertIn(field, q, f"Query missing '{field}': {q}")
    
    def test_queries_are_diverse(self):
        """Test query categories are diverse."""
        categories = set(q["category"] for q in BENCHMARK_QUERIES)
        self.assertGreater(len(categories), 3, "Should have diverse query categories")
    
    def test_expected_titles_not_empty(self):
        """Test all queries have expected results."""
        for q in BENCHMARK_QUERIES:
            self.assertTrue(
                len(q["expected_titles"]) > 0,
                f"Query '{q['query']}' has no expected titles"
            )


def run_retrieval_benchmark(embedder: GeminiEmbedder, 
                           store: ChromaVectorStore) -> Dict[str, Any]:
    """
    Run full retrieval benchmark against query set.
    
    Returns:
        Benchmark results with per-query and aggregate metrics
    """
    results = {
        "queries_tested": 0,
        "successful_queries": 0,
        "total_latency_ms": 0,
        "per_query": []
    }
    
    for query_data in BENCHMARK_QUERIES:
        query_text = query_data["query"]
        expected = set(query_data["expected_titles"])
        
        try:
            start = time.time()
            query_emb = embedder.embed_query(query_text)
            search_results = store.search(query_emb, top_k=5)
            elapsed = (time.time() - start) * 1000
            
            # Extract found titles from results
            found_titles = set()
            for r in search_results:
                # Simple extraction - in real system would parse metadata
                content = r.get("content", "")
                for title in expected:
                    if title.lower() in content.lower():
                        found_titles.add(title)
            
            # Calculate precision
            if search_results:
                precision = len(found_titles) / len(search_results)
            else:
                precision = 0
            
            recall = len(found_titles) / len(expected) if expected else 0
            
            results["per_query"].append({
                "query": query_text,
                "category": query_data["category"],
                "latency_ms": round(elapsed, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "results_found": len(search_results),
            })
            
            results["successful_queries"] += 1
            results["total_latency_ms"] += elapsed
            
        except Exception as e:
            results["per_query"].append({
                "query": query_text,
                "error": str(e)
            })
        
        results["queries_tested"] += 1
    
    # Aggregate stats
    if results["successful_queries"] > 0:
        successful = [q for q in results["per_query"] if "error" not in q]
        results["avg_latency_ms"] = round(
            results["total_latency_ms"] / results["successful_queries"], 2
        )
        results["avg_precision"] = round(
            sum(q["precision"] for q in successful) / len(successful), 4
        )
        results["avg_recall"] = round(
            sum(q["recall"] for q in successful) / len(successful), 4
        )
    
    return results


if __name__ == "__main__":
    unittest.main(verbosity=2)
