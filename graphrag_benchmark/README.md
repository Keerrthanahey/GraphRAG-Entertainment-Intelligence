# GraphRAG Entertainment Intelligence Benchmarking System

> A production-quality, modular benchmarking system for comparing **Basic RAG**, **Hybrid RAG**, and **GraphRAG** pipelines on entertainment domain queries.

---

## Architecture Overview

```
User Query
    |
    v
+---------------------------------------------------+
|            Streamlit Dashboard                     |
|  (Query Input | Benchmarking | Visualization)      |
+---------------------------------------------------+
    |            |              |
    v            v              v
+-------+  +---------+  +-----------+
|Basic  |  | Hybrid  |  | GraphRAG  |
| RAG   |  |  RAG    |  |           |
+---+---+  +----+----+  +-----+-----+
    |           |             |
    v           v             v
+---+---+  +----+----+  +-----+-----+
|Vector |  | Vector  |  |  Vector   |
|Search |  | + Keyword|  |  Search   |
+---+---+  +----+----+  +-----+-----+
    |           |             |
    v           v             v
+---+-----------+-------------+-----+
|        ChromaDB Vector Store       |
+---+-----------+-------------+-----+
    |           |             |
    v           v             v
+---+---+  +----+----+  +-----+-----+
|Embed- |  |Embed-   |  |  Embed-   |
|dings  |  |dings    |  |  dings    |
+---+---+  +----+----+  +-----+-----+
    |           |             |
    v           v             v
+---+-----------+-------------+-----+
|       Gemini Embedding API         |
|     (models/embedding-001)         |
+------------------------------------+

GraphRAG Only:
    |
    v
+-------------------+
|  TigerGraph /     |
|  Local Graph      |
|  (Relationships)  |
+-------------------+
```

---

## Directory Structure

```
graphrag-benchmark/
├── config/
│   └── settings.yaml              # Central configuration
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── logger.py              # Structured logging
│   │   └── metrics.py             # Benchmark metrics
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── models.py              # Movie & TextChunk dataclasses
│   │   ├── csv_loader.py          # CSV loading & validation
│   │   ├── chunker.py             # Text chunking strategies
│   │   ├── entity_extractor.py    # Entity/relationship extraction
│   │   └── orchestrator.py        # Preprocessing orchestrator
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── gemini_embedder.py     # Gemini embedding provider
│   ├── storage/
│   │   ├── __init__.py
│   │   └── chroma_store.py        # ChromaDB vector store
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── base_pipeline.py       # Abstract base class
│   │   ├── basic_rag.py           # Basic RAG pipeline
│   │   ├── hybrid_rag.py          # Hybrid RAG pipeline
│   │   └── graph_rag.py           # GraphRAG pipeline
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── runner.py              # Benchmark execution engine
│   └── api/
│       ├── __init__.py
│       └── pipeline_api.py        # Unified API layer
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Preprocessing tests
│   └── test_retrieval.py          # Retrieval quality tests
├── data/
│   ├── raw/                       # Place IMDb CSV here
│   └── processed/                 # Processed artifacts
├── dashboard.py                    # Streamlit dashboard
├── main.py                         # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
cd graphrag-benchmark

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Get Gemini API key: https://makersuite.google.com/app/apikey
```

Your `.env` file should look like:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Prepare Dataset

Download the IMDb Top 1000 Movies dataset and place it in `data/raw/`:

```bash
mkdir -p data/raw

# Place your IMDb CSV file here:
# data/raw/imdb_top_1000.csv
```

Required columns: `Series_Title`, `Released_Year`, `Genre`, `IMDB_Rating`, `Overview`, `Director`, `Star1-Star4`

### 4. Run the Pipeline

```bash
# Full pipeline (preprocess + embed + store)
python main.py --csv data/raw/imdb_top_1000.csv --full-pipeline

# Or with sample for testing
python main.py --csv data/raw/imdb_top_1000.csv --full-pipeline --sample 50
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

---

## CLI Usage

### Preprocessing Only

```bash
python main.py --csv data/raw/imdb_top_1000.csv --preprocess-only
```

**What it does:**
- Loads and validates CSV
- Cleans data (handles missing values, type conversion)
- Splits into train/val/test (80/10/10)
- Generates text chunks using movie-aware strategy
- Extracts entities and relationships
- Saves processed artifacts to `data/processed/`

### Embedding & Storage Only

```bash
python main.py --embed-only
```

**What it does:**
- Loads preprocessed chunks from `data/processed/`
- Generates embeddings via Gemini API (`models/embedding-001`)
- Stores in ChromaDB with metadata
- Shows token usage and cost estimates

### Test Query

```bash
python main.py --test-query "Movies directed by Christopher Nolan"
```

### Run Benchmark

```bash
python main.py --benchmark --benchmark-queries 15
```

---

## Pipeline Details

### 1. Preprocessing Pipeline

**File:** `src/preprocessing/orchestrator.py`

```python
from src.preprocessing.orchestrator import PreprocessingOrchestrator

orchestrator = PreprocessingOrchestrator()
stats = orchestrator.run(
    csv_path="data/raw/imdb_top_1000.csv",
    sample_size=None,          # Use all rows
    chunk_strategy="movie_aware",  # or 'fixed', 'semantic', 'hierarchical'
    save_artifacts=True
)

print(stats)
```

**Chunking Strategies:**
- `fixed` - Fixed-size chunks with overlap
- `semantic` - Paragraph-aware splitting
- `movie_aware` - Structured chunks (overview, cast, genre, full text)
- `hierarchical` - Parent (summary) + child (detail) chunks

### 2. Embedding Pipeline

**File:** `src/embedding/gemini_embedder.py`

```python
from src.embedding.gemini_embedder import GeminiEmbedder

embedder = GeminiEmbedder()

# Embed chunks
embedded_chunks = embedder.embed_chunks(chunks)

# Embed query
query_embedding = embedder.embed_query("Best sci-fi movies")

# Stats
print(embedder.get_stats())
```

**Features:**
- Automatic batching (configurable batch size)
- Exponential backoff retry
- Token and cost tracking
- Query vs document embedding types

### 3. ChromaDB Storage

**File:** `src/storage/chroma_store.py`

```python
from src.storage.chroma_store import ChromaVectorStore

store = ChromaVectorStore()
store.connect(mode="persistent").create_collection()

# Add documents
store.add_chunks(embedded_chunks)

# Search
results = store.search(query_embedding, top_k=5)

# Search with filters
results = store.search(query_embedding, top_k=5, filters={"chunk_type": "overview"})

# MMR search for diversity
results = store.search_with_mmr(query_embedding, top_k=5)
```

**Features:**
- Persistent and in-memory modes
- Metadata filtering
- Multiple search modes (similarity, MMR)
- Collection management

---

## The Three RAG Pipelines

### Basic RAG (`src/pipelines/basic_rag.py`)

Simple vector similarity search.

```
Query → Embed → Vector Search → Return Top-K
```

**Strengths:** Fast, simple, good baseline

### Hybrid RAG (`src/pipelines/hybrid_rag.py`)

Combines semantic + keyword search with weighted fusion.

```
Query → Embed ─┬─→ Vector Search ──┐
               │                    ├──→ Fuse ──→ Rerank ──→ Top-K
               └──→ Keyword Search ─┘
```

**Strengths:** Better recall, handles exact match queries

### GraphRAG (`src/pipelines/graph_rag.py`)

Uses relationship graph for multi-hop reasoning.

```
Query → Embed → Vector Search → Seed Nodes → Graph Traversal → Enriched Results
```

**Strengths:** Relationship reasoning, discovers connections

---

## Streamlit Dashboard

### Features

| Page | Description |
|------|-------------|
| **Home** | System overview, pipeline status |
| **Query & Compare** | Query all pipelines simultaneously, compare latencies |
| **Benchmark** | Run systematic benchmarks, view visualizations |
| **Graph View** | Explore graph statistics and relationships |
| **System Config** | View configurations, manage data |

### Launch

```bash
streamlit run dashboard.py
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run preprocessing tests
python tests/test_preprocessing.py

# Run retrieval tests
python tests/test_retrieval.py
```

---

## Benchmarking

The system includes 25 benchmark queries covering:

| Category | Examples |
|----------|----------|
| Director queries | "Movies by Christopher Nolan" |
| Actor queries | "Films with Tom Hanks" |
| Genre queries | "Best horror movies" |
| Theme queries | "Movies about space exploration" |
| Relationship queries | "Movies similar to The Matrix" |
| Complex queries | "Best 1990s thrillers with high ratings" |

### Metrics Collected

| Metric | Description |
|--------|-------------|
| Latency | Query response time (ms) |
| P50/P95 Latency | Percentile latencies |
| Token Usage | API tokens consumed |
| Cost | Estimated API cost |
| Success Rate | Query success percentage |
| Error Rate | Failed query percentage |

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key |
| `TIGERGRAPH_HOST` | No | TigerGraph instance host |
| `TIGERGRAPH_PASSWORD` | No | TigerGraph password |

### Config File

Edit `config/settings.yaml` to customize:

```yaml
dataset:
  chunk_size: 512
  chunk_overlap: 50

embeddings:
  batch_size: 100
  max_retries: 3

pipelines:
  basic_rag:
    top_k: 5
  hybrid_rag:
    semantic_weight: 0.6
    keyword_weight: 0.4
  graphrag:
    traversal_depth: 2
    max_nodes: 20
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM/Embeddings | Google Gemini API |
| Vector DB | ChromaDB |
| Graph DB | TigerGraph Savanna (optional) |
| Frontend | Streamlit |
| Visualization | Plotly |
| Data Processing | Pandas |

---

## API Key Setup

### Gemini API Key (Required)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

### TigerGraph (Optional)

1. Sign up at [TigerGraph Savanna](https://savanna.tigergraph.com/)
2. Create a free instance
3. Set environment variables:

```bash
export TIGERGRAPH_HOST="your-instance.i.tgcloud.io"
export TIGERGRAPH_PASSWORD="your-password"
```

---

## Troubleshooting

### "GEMINI_API_KEY not set"

```bash
export GEMINI_API_KEY="your-key"
```

### "No module named 'src'"

Run from project root:

```bash
cd graphrag-benchmark
python main.py
```

### "ChromaDB collection not found"

The system auto-creates collections. If you see issues:

```bash
# Clear and recreate
python main.py --full-pipeline --clear-collection
```

### Rate Limiting

If you hit Gemini rate limits:

```yaml
# config/settings.yaml
embeddings:
  batch_size: 50      # Reduce batch size
  retry_delay: 5.0    # Increase retry delay
```

---

## Performance Tips

1. **Use `--sample` for testing** - Test with 50 movies before full dataset
2. **Persistent mode** - Use `persistent` for ChromaDB to avoid re-embedding
3. **Batch size** - Increase if you have higher rate limits
4. **Skip answers** - Use `--benchmark` without answer generation for faster testing

---

## License

MIT License - Built for hackathon purposes.

---

## Team

Built for hackathon - GraphRAG-powered entertainment relationship intelligence benchmarking system.
