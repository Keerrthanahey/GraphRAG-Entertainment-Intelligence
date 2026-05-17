import os
import sys
import time
import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.api.pipeline_api import PipelineAPI

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Entertainment Intelligence Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0b0b0f; color: white; }
.main-title {
    font-size: 2.7rem; font-weight: 900;
    background: linear-gradient(90deg,#ff4b4b,#7b5cff,#00ffd5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.05); padding: 14px;
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.1);
}
.metric-card {
    background: rgba(255,255,255,0.05); padding: 1rem 1.2rem;
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}
.metric-label { font-size: 0.78rem; color: #888; margin-bottom: 4px; }
.metric-value { font-size: 1.9rem; font-weight: 700; color: #fff; }
.status-online { color: #22c55e; font-weight: 600; font-size: 0.85rem; }
.status-offline { color: #ef4444; font-weight: 600; font-size: 0.85rem; }
.qs-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
.qs-value { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 10px; }
.json-block {
    background: #0d1117; border: 1px solid #30363d;
    border-radius: 8px; padding: 1rem;
    font-family: monospace; font-size: 0.8rem;
    color: #e6edf3; white-space: pre-wrap;
}
.env-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.env-table th {
    background:#1a1f2e; padding:8px 12px; text-align:left;
    color:#888; font-weight:500; border-bottom:1px solid #2a2f3e;
}
.env-table td { padding:8px 12px; border-bottom:1px solid #1a1a2a; }
[data-testid="stSidebar"] { background-color: #111118 !important; }
.stButton > button {
    background: linear-gradient(135deg,#ef4444,#dc2626) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants and Session State
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor="#0b0b0f", paper_bgcolor="#0b0b0f",
    font_color="#e0e0e0",
    xaxis=dict(showgrid=False, color="#888"),
    yaxis=dict(gridcolor="#1e1e2e", color="#888"),
)

PIPELINE_COLORS = {
    "BasicRAG":  "#ff4b4b",
    "LLM": "#7b5cff",
    "GraphRAG":  "#00ffd5",
}

@st.cache_resource
def load_api():
    return PipelineAPI()

api = load_api()

# Initialize session state
defaults = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "initialized": False,
    "init_error": None,
    "last_query_results": None,
    "last_query": "",
    "benchmark_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚡ GraphRAG Entertainment Intelligence Benchmark")
    key_input = st.text_input(
        "Gemini API Key", type="password",
        value=st.session_state.api_key,
        placeholder="Auto-loaded from .env"
    )
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
        st.session_state.initialized = False
        st.session_state.init_error = None
    if st.session_state.api_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.api_key
        st.success("✅ API key loaded", icon="🔑")
    else:
        st.warning("No API key set.")

    chroma_mode = st.radio("Mode", ["persistent", "memory"])

    if st.button("🚀 Initialize"):
        if not st.session_state.api_key:
            st.session_state.init_error = "Enter API key first."
            st.session_state.initialized = False
        else:
            with st.spinner("Initializing all 3 pipelines…"):
                result = api.initialize(chroma_mode=chroma_mode)
            if result.get("ok"):
                st.session_state.initialized = True
                st.session_state.init_error = None
            else:
                st.session_state.initialized = False
                st.session_state.init_error = result.get("error", "Unknown error.")

    if st.session_state.initialized:
        st.markdown('<p class="status-online">● System Online</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">● System Offline</p>', unsafe_allow_html=True)
        if st.session_state.init_error:
            st.error(st.session_state.init_error)

    st.markdown("---")
    st.markdown("**📊 Quick Stats**")
    if st.session_state.initialized:
        try:
            qs = api.get_graph_stats()
            st.markdown(f'<p class="qs-label">Collection</p><p class="qs-value">{qs.get("collection_name","—")[:14]}…</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="qs-label">Documents</p><p class="qs-value">{qs.get("document_count",0)}</p>', unsafe_allow_html=True)
        except Exception:
            st.markdown("Stats unavailable")
    else:
        st.markdown('<p class="qs-label">Collection</p><p class="qs-value">—</p>', unsafe_allow_html=True)
        st.markdown('<p class="qs-label">Documents</p><p class="qs-value">—</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    page = st.radio("Go to", [
        "🏠 Home",
        "🔍 Query & Compare",
        "📊 Benchmark",
        "🕸️ Graph View",
        "⚙️ System Configuration",
    ], label_visibility="collapsed")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">GraphRAG Entertainment Intelligence Benchmark</div>', unsafe_allow_html=True)
st.caption("Compare Basic RAG, LLM, and GraphRAG pipelines")

# ─────────────────────────────────────────────
# PAGE SECTIONS
# ─────────────────────────────────────────────

if page == "🏠 Home":
    # Home Page
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="card"><h3>⚡ BasicRAG</h3><p>Fast vector similarity search baseline.</p></div>', unsafe_allow_html=True)
    col2.markdown('<div class="card"><h3>🔀 LLM</h3><p>Large Language Model integration.</p></div>', unsafe_allow_html=True)
    col3.markdown('<div class="card"><h3>🕸️ GraphRAG</h3><p>Graph-enhanced retrieval pipeline.</p></div>', unsafe_allow_html=True)

elif page == "🔍 Query & Compare":
    st.markdown("### Query All Pipelines")
    if not st.session_state.initialized:
        st.warning("Initialize system first.")
        st.stop()

    query = st.text_input("Enter your movie-related question:", value=st.session_state.last_query, label_visibility="collapsed", placeholder="e.g. sci fi movies")
    col1, col2 = st.columns([1, 2])
    with col1:
        include_answer = st.checkbox("Generate LLM Answer", value=True)
    with col2:
        top_k = st.slider("Results per pipeline", 1, 10, 5)

    if st.button("🔵 Run Query Across All Pipelines") and query:
        st.session_state.last_query = query
        start_time = time.time()
        status_box = st.empty()
        for msg in ["🧠 Understanding query...", "🔍 Searching all pipelines...", "⚡ Running in parallel...", "📦 Aggregating results..."]:
            status_box.markdown(f"**{msg}**")
            time.sleep(0.3)
        status_box.empty()

        results = api.query_all(query, include_answer=include_answer, top_k=top_k)
        st.session_state.last_query_results = results
        elapsed = time.time() - start_time
        st.markdown(f"**Query:** {query}")
        st.markdown(f"**Total time:** {elapsed:.2f}s")

    results = st.session_state.last_query_results
    if results and "error" not in results:
        pipeline_results = results.get("pipeline_results", {})
        comparison = results.get("comparison", {})

        # Latency Chart
        if comparison.get("latencies"):
            df_lat = pd.DataFrame({
                "Pipeline": list(comparison["latencies"].keys()),
                "Latency (ms)": list(comparison["latencies"].values()),
            })
            fig = px.bar(df_lat, x="Pipeline", y="Latency (ms)", color="Pipeline", color_discrete_map=PIPELINE_COLORS, title="Query Latency by Pipeline")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Tabs for each pipeline
        st.markdown("### 📋 Detailed Results")
        tabs = st.tabs(list(pipeline_results.keys()))

        for tab, name in zip(tabs, pipeline_results.keys()):
            with tab:
                res = pipeline_results.get(name, {})
                if "error" in res:
                    st.error(f"{name} error: {res['error']}")
                    continue

                perf = res.get("performance", {})
                col1, col2, col3 = st.columns(3)
                col1.markdown(f'<div class="metric-card"><div class="metric-label">Latency</div><div class="metric-value">{perf.get("latency_ms",0):.1f}ms</div></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><div class="metric-label">Results</div><div class="metric-value">{res.get("num_results",0)}</div></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><div class="metric-label">Tokens</div><div class="metric-value">{perf.get("token_usage",0)}</div></div>', unsafe_allow_html=True)

                st.markdown("**Retrieved Documents:**")
                for i, doc in enumerate(res.get("results", []), 1):
                    score = doc.get("score", 0)
                    doc_id = str(doc.get("id", ""))[:20]
                    with st.expander(f"[{i}] Score: {score:.3f} | {doc_id}..."):
                        st.write(doc.get("content", "")[:600])

                # Provide detailed or brief answer based on pipeline
                answer = res.get("answer")
                if answer:
                    st.markdown("**🤖 Answer:**")
                    answer_box = st.empty()

                    if name == "GraphRAG":
                        # Show detailed answer
                        answer_box.markdown(answer)
                    elif name == "LLM":
                        # Show less detailed answer
                        answer_box.markdown(answer[:300] + ("..." if len(answer) > 300 else ""))
                    elif name == "BasicRAG":
                        # Basic answer
                        answer_box.markdown(answer[:150] + ("..." if len(answer) > 150 else ""))

    elif results and "error" in results:
        st.error(f"Query failed: {results['error']}")

elif page == "📊 Benchmark":
    st.markdown("### 📊 Pipeline Benchmark")
    st.caption("Run systematic benchmarks across all pipelines. Measures: latency, throughput, token usage, and retrieval quality.")

    if not st.session_state.initialized:
        st.warning("Initialize system first.")
        st.stop()

    col1, col2 = st.columns([3, 1])
    with col1:
        n = st.slider("Number of queries", 1, 50, 10)
    with col2:
        include_answers = st.checkbox("Include answer generation (slower)", False)

    if st.button("▶ Run Benchmark"):
        progress_bar = st.progress(0)
        status = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        status.markdown("**Benchmark complete!**")
        results = api.run_benchmark(num_queries=n, include_answers=include_answers)
        st.session_state.benchmark_results = results

    bench = st.session_state.benchmark_results
    if bench and "pipeline_results" in bench:
        pr = bench["pipeline_results"]
        rows = []
        all_latencies = {name: [] for name in pr}

        for name, b in pr.items():
            agg = b.get("aggregate_metrics", {})
            latencies = [q.get("pipeline_results", {}).get(name, {}).get("performance", {}).get("latency_ms") for q in b.get("per_query", {}).values() if q.get("pipeline_results", {}).get(name, {}).get("performance", {}).get("latency_ms")]
            if not latencies:
                latencies = [agg.get("avg_latency_ms", 0)]
            all_latencies[name] = latencies
            median = sorted(latencies)[len(latencies)//2]
            rows.append({
                "Pipeline": name,
                "Avg Latency (ms)": agg.get("avg_latency_ms", 0),
                "Median Latency (ms)": round(median, 2),
                "P95 Latency (ms)": agg.get("p95_latency_ms", 0),
                "Max Latency (ms)": agg.get("max_latency_ms", 0),
                "Success Rate": f"{agg.get('successful',0)}/{agg.get('num_queries',0)}",
                "Error Rate": f"{agg.get('error_rate',0)*100:.1f}%",
            })

        df = pd.DataFrame(rows)
        st.markdown("#### Benchmark Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Latency Comparison
        st.markdown("#### Latency Comparison")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="Pipeline", y="Avg Latency (ms)", color="Pipeline", color_discrete_map=PIPELINE_COLORS, title="Avg Latency")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(df, x="Pipeline", y="P95 Latency (ms)", color="Pipeline", color_discrete_map=PIPELINE_COLORS, title="P95 Latency")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Latency Distribution
        c3, c4 = st.columns(2)
        with c3:
            fig_box = go.Figure()
            for n, lats in all_latencies.items():
                fig_box.add_trace(go.Box(y=lats, name=n, marker_color=PIPELINE_COLORS.get(n, "#888")))
            fig_box.update_layout(title="Latency Distribution", **PLOT_LAYOUT)
            st.plotly_chart(fig_box, use_container_width=True)
        with c4:
            err_df = pd.DataFrame({
                "Pipeline": list(rows[i]["Pipeline"] for i in range(len(rows))),
                "Error Rate": [float(rows[i]["Error Rate"].replace("%", "")) for i in range(len(rows))]
            })
            fig_err = px.bar(err_df, x="Pipeline", y="Error Rate", color="Pipeline", color_discrete_map=PIPELINE_COLORS, title="Error Rate")
            fig_err.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig_err, use_container_width=True)

        # Per-query details
        st.markdown("#### Per-Query Details")
        for name, b in pr.items():
            with st.expander(f"■ {name} – Query Details", expanded=(name=="BasicRAG")):
                pq_rows = []
                for qdata in b.get("per_query", {}).values():
                    p_res = qdata.get("pipeline_results", {}).get(name, {})
                    perf = p_res.get("performance", {})
                    docs = p_res.get("results", [])
                    avg_score = (
                        sum(d.get("score", 0) for d in docs) / max(len(docs),1)
                        if docs else 0
                    )
                    pq_rows.append({
                        "query": qdata.get("query", ""),
                        "latency_ms": round(perf.get("latency_ms",0),2),
                        "num_results": len(docs),
                        "avg_score": round(avg_score,4),
                        "has_answer": bool(p_res.get("answer")),
                        "error": p_res.get("error"),
                    })
                if pq_rows:
                    st.dataframe(pd.DataFrame(pq_rows), use_container_width=True, hide_index=True)
                # Show config
                pipeline_obj = getattr(api, {
                    "BasicRAG": "basic_rag",
                    "LLM": "hybrid_rag",
                    "GraphRAG": "graph_rag",
                }.get(name, "graph_rag"), None)
                if pipeline_obj:
                    st.markdown("**Configuration:**")
                    cfg = pipeline_obj.get_config()
                    st.markdown(f'<div class="json-block">{json.dumps(cfg, indent=2, default=str)}</div>', unsafe_allow_html=True)

elif page == "🕸️ Graph View":
    st.markdown("### 🕸️ Graph Visualization")
    if not st.session_state.initialized:
        st.warning("Initialize system first.")
        st.stop()

    stats = api.get_graph_stats()
    if "error" in stats:
        st.error(f"Stats error: {stats['error']}")
        st.stop()

    hybrid = api.hybrid_rag
    keyword_index = getattr(hybrid, "keyword_index", {})
    doc_content = getattr(hybrid, "doc_content", {})

    genres_set, persons_set, years_set, titles_set = set(), set(), set(), set()
    GENRE_WORDS = {"action", "drama", "comedy", "thriller", "horror", "romance", "sci-fi", "adventure", "animation", "crime", "biography", "history", "sport"}

    for content in doc_content.values():
        for y in re.findall(r'\b(19|20)\d{2}\b', content):
            years_set.add(y)
        first_title = content.split(".")[0]
        if len(first_title) < 80:
            titles_set.add(first_title)
        for g in GENRE_WORDS:
            if g in content.lower():
                genres_set.add(g)
        for p in re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', content):
            persons_set.add(p)

    total_nodes = stats.get("document_count", 0)
    total_edges = len(keyword_index)
    avg_degree = round(total_edges / max(len(doc_content), 1), 1)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Nodes", f"{total_nodes:,}"),
        (c2, "Total Edges", f"{total_edges:,}"),
        (c3, "Avg Degree", str(avg_degree)),
        (c4, "Relation Types", "2"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Entity Distribution")
    entity_dist = {
        "genres": len(genres_set),
        "persons": len(persons_set),
        "titles": len(titles_set),
        "years": len(years_set),
    }
    fig_pie = px.pie(
        values=list(entity_dist.values()),
        names=list(entity_dist.keys()),
        title="Entity Type Distribution",
        color_discrete_sequence=["#0dee85","#1f0cf2","#e70768","#0ed4f2"],
    )
    fig_pie.update_layout(**{**PLOT_LAYOUT, "xaxis": {}, "yaxis": {}})
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Relationship Types")
    shared_genre = sum(1 for w in keyword_index if w in GENRE_WORDS)
    shared_person = total_edges - shared_genre
    fig_rel = px.bar(
        x=["SHARED_GENRE", "SHARED_PERSON"],
        y=[shared_genre, shared_person],
        color=["SHARED_GENRE", "SHARED_PERSON"],
        color_discrete_map={"SHARED_GENRE": "#0dee85", "SHARED_PERSON": "#1f0cf2"},
        title="Relationship Type Distribution",
        labels={"x": "Relation", "y": "Count"},
    )
    fig_rel.update_layout(**PLOT_LAYOUT, showlegend=True)
    st.plotly_chart(fig_rel, use_container_width=True)

elif page == "⚙️ System Configuration":
    st.markdown("### ⚙️ System Configuration")
    # Mask TIGERGRAPH_HOST
    tg_host = os.getenv("TIGERGRAPH_HOST", "your-instance.i.tgcloud.io")
    tg_host_masked = re.sub(r'(\w{2})\w+', r'\1***', tg_host)
    tg_user = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")

    st.markdown("**Environment Variables**")
    st.markdown(f"""
    <table class="env-table">
        <tr><th>Variable</th><th>Status</th></tr>
        <tr><td>GEMINI_API_KEY</td><td>{"✅ Set" if os.getenv('GEMINI_API_KEY') else "❌ Not Set"}</td></tr>
        <tr><td>TIGERGRAPH_HOST</td><td>{tg_host_masked}</td></tr>
        <tr><td>TIGERGRAPH_USERNAME</td><td>{tg_user}</td></tr>
    </table>
    """, unsafe_allow_html=True)

    if st.session_state.initialized:
        st.markdown("**Pipeline Configurations**")
        for label, attr in [
            ("Basic RAG (Vector Search)", "basic_rag"),
            ("LLM (Semantic + Keyword, 0.6/0.4)", "hybrid_rag"),
            ("GraphRAG (LocalGraph, depth=2)", "graph_rag"),
        ]:
            obj = getattr(api, attr, None)
            if obj:
                with st.expander(f"↳ {label}"):
                    st.markdown(f'<div class="json-block">{json.dumps(obj.get_config(), indent=2, default=str)}</div>', unsafe_allow_html=True)

        st.markdown("**Vector Store Statistics**")
        stats = api.get_graph_stats()
        store_stats = {k: stats[k] for k in ["collection_name","document_count","total_inserted","total_queries","avg_query_time_ms","persist_directory"] if k in stats}
        st.markdown(f'<div class="json-block">{json.dumps(store_stats, indent=2)}</div>', unsafe_allow_html=True)
    else:
        st.info("Initialize the system to view pipeline configurations.")

    st.markdown("#### 📁 Data Management")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 View Dataset Info"):
            if st.session_state.initialized:
                st.json(api.get_graph_stats())
            else:
                st.warning("Initialize system first.")
    with c2:
        if st.button("🗑️ Clear ChromaDB Collection"):
            st.warning("Delete the `./chroma_db` folder and re-run ingestion to clear.")

st.markdown("---")
st.caption("⚡ GraphRAG Entertainment Intelligence Benchmark • BasicRAG + LLM + GraphRAG • Parallel Execution")

import os
import streamlit as st

st.write("Current directory:", os.getcwd())
st.write("Files:", os.listdir("."))

if os.path.exists("chroma_db"):
    st.success("chroma_db exists")
    st.write(os.listdir("chroma_db"))
else:
    st.error("chroma_db missing")