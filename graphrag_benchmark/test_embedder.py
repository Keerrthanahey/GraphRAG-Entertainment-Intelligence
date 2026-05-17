# test_embedder.py

from src.embedding.gemini_embedder import GeminiEmbedder

embedder = GeminiEmbedder()

vec = embedder.embed_query(
    "best science fiction movies"
)

print(type(vec))
print(len(vec))
print(vec[:5])