# 🎬 GraphRAG Entertainment Intelligence System - CineGraphAI

> Advanced Retrieval-Augmented Generation (RAG) Benchmarking Platform using BasicRAG, HybridRAG, and GraphRAG pipelines.

Built for the **TigerGraph GraphRAG Hackathon 2025**.

---

# 🚀 Overview

GraphRAG Entertainment Intelligence is a complete AI benchmarking system designed to compare multiple retrieval architectures on entertainment datasets using:

- Semantic Search
- Hybrid Retrieval
- Graph-Based Reasoning
- Interactive Benchmark Dashboards
- Latency & Performance Analytics

The system uses the IMDb Top 1000 Movies dataset and evaluates how different RAG pipelines perform across retrieval quality, speed, and contextual understanding.

---

# ✨ Features

## 🔍 BasicRAG
- Semantic vector retrieval
- ChromaDB similarity search
- Fast baseline pipeline

## ⚡ HybridRAG
- Semantic + keyword search
- Weighted fusion retrieval
- Improved precision and recall

## 🕸️ GraphRAG
- Relationship-aware retrieval
- Graph traversal
- Multi-hop reasoning
- Contextual entity exploration

---

# 📊 Dashboard Features

- Interactive Query Playground
- Cross-pipeline comparison
- Benchmark analytics
- Latency visualization
- Graph statistics
- Entity relationship exploration
- Modern Streamlit UI

---

# 🛠️ Tech Stack

## Frontend
- Streamlit
- Plotly

## Backend / AI
- Python
- Gemini API
- ChromaDB

## Retrieval
- GraphRAG
- Hybrid Retrieval
- Vector Search

## Data Processing
- Pandas
- NumPy

---

# 🧠 Architecture

```text
IMDb Dataset
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Chunk Generation + Metadata Extraction
      │
      ▼
Gemini Embeddings
      │
      ▼
ChromaDB Vector Store
      │
      ├── BasicRAG
      ├── HybridRAG
      └── GraphRAG
              │
              ▼
Benchmark Dashboard

graphrag_benchmark/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── api/
│   ├── benchmark/
│   ├── embedding/
│   ├── pipelines/
│   ├── preprocessing/
│   ├── storage/
│   └── utils/
│
├── chroma_db/
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
