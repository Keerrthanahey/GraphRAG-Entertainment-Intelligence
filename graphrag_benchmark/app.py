from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.pipeline_api import PipelineAPI

app = FastAPI(title="GraphRAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API
api = PipelineAPI()

# IMPORTANT
api.initialize(chroma_mode="persistent")


@app.get("/")
def health():
    return {
        "status": "online",
        "message": "GraphRAG Backend Running"
    }


@app.get("/search")
def search(query: str):
    try:
        result = api.query_all(
            query_text=query,
            include_answer=False,
            top_k=5
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/graph")
def graph(query: str):
    try:
        result = api.query(
            query_text=query,
            pipeline="graph_rag",
            include_answer=False,
            top_k=5
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/benchmark")
def benchmark(n: int = 5):
    try:
        result = api.run_benchmark(
            num_queries=n,
            include_answers=False
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/status")
def status():
    return {
        "initialized": True,
        "pipelines": list(api.pipelines.keys())
    }