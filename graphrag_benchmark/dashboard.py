import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# ---------------------------------------------------------
# PATH FIX
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------
# IMPORT API
# ---------------------------------------------------------
from src.api.pipeline_api import PipelineAPI

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineGraphAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #070b14;
    color: white;
}

.main-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg,#ff4d4d,#7c4dff,#00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9aa4b2;
    margin-top: -10px;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
}

.metric-title {
    color: #9aa4b2;
    font-size: 0.85rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}

.pipeline-card {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

.pipeline-title {
    font-size: 1.3rem;
    font-weight: 800;
}

.answer-box {
    background: #101826;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #1d2a3a;
}

[data-testid="stSidebar"] {
    background: #0d1117;
}

.stButton>button {
    background: linear-gradient(135deg,#ff4d4d,#ff1744);
    border: none;
    color: white;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
    height: 3em;
}

.stTextInput>div>div>input {
    background: #101826;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHE API
# ---------------------------------------------------------
@st.cache_resource
def load_api():
    return PipelineAPI()

api = load_api()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
defaults = {
    "initialized": False,
    "last_results": None,
    "last_query": "",
    "init_error": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## 🎬 CineGraphAI")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", "")
    )

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.success("API Key Loaded")

    chroma_mode = st.selectbox(
        "Chroma Mode",
        ["persistent", "memory"],
        index=0
    )

    if st.button("🚀 Initialize System"):

        with st.spinner("Initializing pipelines..."):

            result = api.initialize(chroma_mode=chroma_mode)

            if result.get("ok"):

                st.session_state.initialized = True
                st.session_state.init_error = None

                st.success("Pipelines Initialized")

                try:
                    count = api.store.collection.count()
                    st.success(f"Documents Loaded: {count}")

                    if count == 0:
                        st.error("ChromaDB is EMPTY")
                        st.warning(
                            "Your deployment does not contain persisted embeddings."
                        )

                except Exception as e:
                    st.error(str(e))

            else:
                st.session_state.initialized = False
                st.session_state.init_error = result.get("error")

    st.markdown("---")

    if st.session_state.initialized:
        st.success("● System Online")
    else:
        st.error("● System Offline")

    if st.session_state.init_error:
        st.error(st.session_state.init_error)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">CineGraphAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">GraphRAG Entertainment Intelligence Benchmark</div>',
    unsafe_allow_html=True
)

st.markdown("")

# ---------------------------------------------------------
# STATS
# ---------------------------------------------------------
if st.session_state.initialized:

    try:
        stats = api.get_graph_stats()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Documents</div>
                <div class="metric-value">
                    {stats.get("document_count",0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Collection</div>
                <div class="metric-value">
                    {stats.get("collection_name","N/A")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Queries</div>
                <div class="metric-value">
                    {stats.get("total_queries",0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Query</div>
                <div class="metric-value">
                    {round(stats.get("avg_query_time_ms",0),2)}ms
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(str(e))

# ---------------------------------------------------------
# QUERY SECTION
# ---------------------------------------------------------
st.markdown("## 🔍 Ask CineGraphAI")

query = st.text_input(
    "",
    placeholder="e.g. Best sci-fi movies with AI themes"
)

col1, col2 = st.columns([1,1])

with col1:
    include_answer = st.checkbox(
        "Generate Answers",
        value=True
    )

with col2:
    top_k = st.slider(
        "Results Per Pipeline",
        1,
        10,
        5
    )

# ---------------------------------------------------------
# RUN QUERY
# ---------------------------------------------------------
if st.button("⚡ Run Query"):

    if not st.session_state.initialized:
        st.error("Initialize the system first")

    elif not query.strip():
        st.error("Enter a query")

    else:

        with st.spinner("Running pipelines..."):

            start = time.time()

            results = api.query_all(
                query=query,
                include_answer=include_answer,
                top_k=top_k
            )

            elapsed = round(time.time() - start, 2)

            st.session_state.last_results = results

        st.success(f"Completed in {elapsed}s")

# ---------------------------------------------------------
# SHOW RESULTS
# ---------------------------------------------------------
results = st.session_state.last_results

if results and "pipeline_results" in results:

    comparison = results.get("comparison", {})
    pipeline_results = results.get("pipeline_results", {})

    # -----------------------------------------------------
    # LATENCY GRAPH
    # -----------------------------------------------------
    if comparison.get("latencies"):

        df = pd.DataFrame({
            "Pipeline": list(comparison["latencies"].keys()),
            "Latency": list(comparison["latencies"].values())
        })

        fig = px.bar(
            df,
            x="Pipeline",
            y="Latency",
            color="Pipeline",
            title="Pipeline Latency Comparison"
        )

        fig.update_layout(
            plot_bgcolor="#070b14",
            paper_bgcolor="#070b14",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
    for name, res in pipeline_results.items():

        st.markdown(f"""
        <div class="pipeline-card">
            <div class="pipeline-title">{name}</div>
        </div>
        """, unsafe_allow_html=True)

        if "error" in res:
            st.error(res["error"])
            continue

        perf = res.get("performance", {})

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Latency",
            f'{perf.get("latency_ms",0)} ms'
        )

        c2.metric(
            "Results",
            res.get("num_results",0)
        )

        c3.metric(
            "Tokens",
            perf.get("token_usage",0)
        )

        # -------------------------------------------------
        # DOCUMENTS
        # -------------------------------------------------
        st.markdown("### 📄 Retrieved Documents")

        docs = res.get("results", [])

        if not docs:
            st.warning("No documents retrieved")

        for i, doc in enumerate(docs, 1):

            with st.expander(
                f"Document {i} • Score {round(doc.get('score',0),3)}"
            ):
                st.write(doc.get("content",""))

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------
        if res.get("answer"):

            st.markdown("### 🤖 Generated Answer")

            st.markdown(f"""
            <div class="answer-box">
            {res.get("answer")}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

# ---------------------------------------------------------
# DEBUG PANEL
# ---------------------------------------------------------
with st.expander("🛠 Debug Information"):

    st.write("Current Directory:")
    st.code(os.getcwd())

    st.write("Files:")
    st.write(os.listdir("."))

    chroma_path = BASE_DIR / "chroma_db"

    st.write("Chroma Path:")
    st.code(str(chroma_path))

    if chroma_path.exists():

        st.success("chroma_db exists")

        st.write(list(chroma_path.iterdir()))

    else:
        st.error("chroma_db missing")