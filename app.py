"""
app.py
======
NLP Semantic Similarity Explorer — Main Streamlit Application

Entry point for the entire app. Run with:
    streamlit run app.py

Requirements: see requirements.txt
Model used  : sentence-transformers/all-MiniLM-L6-v2 (free, Apache 2.0)
No preprocessing · No tokenization · No model training
"""

import numpy as np
import streamlit as st

# Local modules
from similarity  import (get_embeddings, cosine_similarity,
                          pairwise_similarity_matrix, get_top_similar,
                          similarity_label, MODEL_INFO, MODEL_DIM)
from graphs      import (bar_chart_top_similar, heatmap_similarity,
                          pca_scatter, radar_chart, gauge_chart,
                          distribution_chart)
from components  import (inject_css, render_hero_header, animated_divider,
                          render_metrics, render_top_results,
                          render_embedding_preview, render_critical_thinking_cards,
                          render_model_info, render_sidebar, render_footer,
                          render_success_banner, render_score_bar)
from utils       import (init_session_state, save_to_history, get_history_df,
                          clear_history, results_to_csv, history_to_csv,
                          compute_paul_scores, render_performance_stats,
                          clipboard_button, format_vector)


# ──────────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "NLP Similarity Explorer",
    page_icon    = "🔮",
    layout       = "wide",
    initial_sidebar_state = "expanded",
    menu_items   = {
        "Get Help":    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "Report a bug": None,
        "About": (
            "**NLP Semantic Similarity Explorer**\n\n"
            "Uses `sentence-transformers/all-MiniLM-L6-v2` to compute "
            "cosine similarity between texts.\n\n"
            "Built for BS Artificial Intelligence — Quiz Project."
        ),
    },
)

# ──────────────────────────────────────────────────────────────
#  Bootstrap
# ──────────────────────────────────────────────────────────────
inject_css("style.css")
init_session_state()

# ──────────────────────────────────────────────────────────────
#  Sidebar (returns selected page)
# ──────────────────────────────────────────────────────────────
page = render_sidebar("Main")
render_performance_stats()

# ══════════════════════════════════════════════════════════════
#  PAGE: MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════
if page == "Main":

    render_hero_header()
    animated_divider()

    # ── Input Section ─────────────────────────────────────────
    st.markdown("## ✍️ Enter Your Texts")
    st.caption(
        "Enter 2 to 10 texts (words, sentences, or paragraphs). "
        "The model computes cosine similarity between ALL pairs."
    )

    # Dynamic number of text inputs
    n_inputs = st.slider("Number of texts to compare", min_value=2, max_value=10, value=3)

    default_texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "The cat sat on the mat reading a book.",
        "Deep learning models are a subset of machine learning.",
        "Neural networks are inspired by the human brain.",
    ]

    texts = []
    cols_per_row = 2
    input_cols = [st.columns(cols_per_row) for _ in range((n_inputs + 1) // cols_per_row)]

    flat_cols = [c for row in input_cols for c in row]
    for i in range(n_inputs):
        with flat_cols[i]:
            default = default_texts[i] if i < len(default_texts) else ""
            txt = st.text_area(
                f"Text {i+1}",
                value=default,
                height=110,
                placeholder=f"Enter word, sentence, or paragraph #{i+1}…",
                key=f"text_input_{i}",
            )
            texts.append(txt)

    animated_divider()

    # ── Controls Row ──────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        top_k = st.slider("Top-K similar results to show", 2, min(9, n_inputs - 1) if n_inputs > 2 else 1, min(5, n_inputs - 1))

    with ctrl_col2:
        run_btn = st.button("🚀 Analyse Similarity", use_container_width=True)

    with ctrl_col3:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
        if reset_btn:
            for i in range(n_inputs):
                if f"text_input_{i}" in st.session_state:
                    del st.session_state[f"text_input_{i}"]
            st.rerun()

    # ── Run Analysis ──────────────────────────────────────────
    if run_btn:
        # Validate inputs
        clean_texts = [t.strip() for t in texts if t.strip()]
        if len(clean_texts) < 2:
            st.warning("⚠️ Please enter at least **2 non-empty texts** to compare.")
            st.stop()

        with st.spinner("🔮 Generating embeddings…"):
            embeddings, elapsed_ms = get_embeddings(clean_texts)

        # Primary score = similarity between first two texts
        primary_score = cosine_similarity(embeddings[0], embeddings[1])
        label, color  = similarity_label(primary_score)

        # Pairwise matrix
        matrix = pairwise_similarity_matrix(embeddings)

        # All unique pairwise scores (upper triangle)
        all_scores = []
        for i in range(len(clean_texts)):
            for j in range(i + 1, len(clean_texts)):
                all_scores.append(float(matrix[i, j]))

        # Top-K
        top_results = get_top_similar(
            embeddings[0], embeddings, clean_texts,
            top_k=min(top_k, len(clean_texts) - 1),
        )

        # Paul's scores
        paul_scores = compute_paul_scores(primary_score, all_scores, clean_texts)

        # Save to history
        save_to_history(clean_texts, all_scores, primary_score, elapsed_ms)

        # Persist results in session for Graphs page
        st.session_state["current"] = {
            "texts":         clean_texts,
            "embeddings":    embeddings,
            "matrix":        matrix,
            "all_scores":    all_scores,
            "top_results":   top_results,
            "paul_scores":   paul_scores,
            "primary_score": primary_score,
            "label":         label,
            "elapsed_ms":    elapsed_ms,
        }

        # ── Results ──────────────────────────────────────────
        render_success_banner(primary_score, label)

        # Metrics row
        render_metrics(primary_score, label, color,
                       len(clean_texts), elapsed_ms, MODEL_DIM)

        animated_divider()

        # Tabs for results
        tab1, tab2, tab3 = st.tabs(["📊 Results", "🧬 Embeddings", "📥 Download"])

        with tab1:
            col_g, col_m = st.columns([1.4, 1])

            with col_g:
                st.markdown("#### 🏆 Top Similar Results")
                render_top_results(top_results)

            with col_m:
                st.markdown("#### 🎯 Similarity Gauge")
                st.plotly_chart(gauge_chart(primary_score), use_container_width=True)

            st.markdown("#### 📊 Score Progress")
            for i in range(len(clean_texts)):
                for j in range(i + 1, len(clean_texts)):
                    sc = float(matrix[i, j])
                    lbl, _ = similarity_label(sc)
                    render_score_bar(
                        sc,
                        f"Text {i+1} ↔ Text {j+1}  ({lbl})"
                    )

        with tab2:
            st.markdown("#### 🧬 Embedding Vectors (first 16 dims)")
            for idx, (txt, emb) in enumerate(zip(clean_texts, embeddings)):
                render_embedding_preview(emb, label=f"Text {idx+1}: {txt[:50]}")
                st.markdown("")

            if st.checkbox("Show full first embedding as JSON"):
                clipboard_button(str(embeddings[0].tolist()), "📋 Copy Vector")
                st.json({"embedding_dim": len(embeddings[0]),
                         "first_16_values": embeddings[0][:16].tolist()})

        with tab3:
            csv_bytes = results_to_csv(clean_texts, top_results, matrix)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_bytes,
                file_name="similarity_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("CSV contains Top-K results and the full pairwise similarity matrix.")

    elif st.session_state.get("current"):
        # Show cached results without re-running
        cur = st.session_state["current"]
        st.info("💡 Showing results from the last analysis. Press **Analyse** to re-run.")
        render_success_banner(cur["primary_score"], cur["label"])
        render_metrics(cur["primary_score"], cur["label"], "#6C63FF",
                       len(cur["texts"]), cur["elapsed_ms"], MODEL_DIM)

    else:
        st.info("👆 Enter your texts above and click **Analyse Similarity** to begin.")


# ══════════════════════════════════════════════════════════════
#  PAGE: VISUALISATIONS
# ══════════════════════════════════════════════════════════════
elif page == "Graphs":

    st.markdown("## 📊 Interactive Visualisations")
    animated_divider()

    cur = st.session_state.get("current")
    if cur is None:
        st.warning("🔮 No analysis data yet. Go to **Main Analysis** and run a comparison first.")
    else:
        texts      = cur["texts"]
        embeddings = cur["embeddings"]
        matrix     = cur["matrix"]
        top_results= cur["top_results"]
        all_scores = cur["all_scores"]
        paul_scores= cur["paul_scores"]

        # ── 1. Animated Bar Chart ─────────────────────────────
        st.markdown("### 📊 Animated Bar Chart — Top Similar Results")
        fig_bar = bar_chart_top_similar(top_results)
        st.plotly_chart(fig_bar, use_container_width=True)

        animated_divider()

        # ── 2. Heatmap ────────────────────────────────────────
        st.markdown("### 🔥 Pairwise Similarity Heatmap")
        fig_heat = heatmap_similarity(matrix, texts)
        st.plotly_chart(fig_heat, use_container_width=True)

        animated_divider()

        # ── 3. PCA 2D Scatter ─────────────────────────────────
        st.markdown("### 🧠 2D PCA Embedding Projection")
        fig_pca = pca_scatter(embeddings, texts)
        st.plotly_chart(fig_pca, use_container_width=True)

        animated_divider()

        # ── Optional Charts ───────────────────────────────────
        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            st.markdown("### 🎯 Radar — Critical Thinking Standards")
            fig_radar = radar_chart(paul_scores)
            st.plotly_chart(fig_radar, use_container_width=True)

        with opt_col2:
            st.markdown("### 📈 Score Distribution")
            fig_dist = distribution_chart(all_scores)
            st.plotly_chart(fig_dist, use_container_width=True)

        animated_divider()

        # ── Gauge ─────────────────────────────────────────────
        st.markdown("### ⏱️ Primary Similarity Gauge")
        st.plotly_chart(gauge_chart(cur["primary_score"]), use_container_width=True)

        st.caption("All graphs are fully interactive — zoom, pan, hover for details.")


# ══════════════════════════════════════════════════════════════
#  PAGE: CRITICAL THINKING
# ══════════════════════════════════════════════════════════════
elif page == "Critical":

    st.markdown("## 🧪 Paul's Critical Thinking Standards")
    animated_divider()

    cur = st.session_state.get("current")
    scores = cur["paul_scores"] if cur else None

    render_critical_thinking_cards(scores)

    if cur:
        animated_divider()
        st.markdown("### 🎯 Standards Radar Chart")
        st.plotly_chart(radar_chart(cur["paul_scores"]), use_container_width=True)
        st.caption(
            "Scores above are heuristic proxies derived from the similarity "
            "analysis — not absolute measures of reasoning quality."
        )
    else:
        st.info("Run an analysis first to see quantitative standard scores.")


# ══════════════════════════════════════════════════════════════
#  PAGE: SESSION HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "History":

    st.markdown("## 📜 Session History")
    animated_divider()

    df_hist = get_history_df()

    if df_hist.empty:
        st.info("No history yet. Run some analyses and come back here.")
    else:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        col_dl, col_cl = st.columns(2)
        with col_dl:
            st.download_button(
                label="📥 Download History CSV",
                data=history_to_csv(),
                file_name="session_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_cl:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_history()
                st.rerun()

        # Score trend if enough runs
        if len(df_hist) >= 2:
            import plotly.graph_objects as go
            fig_trend = go.Figure(go.Scatter(
                x=df_hist["Run #"],
                y=df_hist["Primary Score"],
                mode="lines+markers",
                marker=dict(size=8, color="#6C63FF"),
                line=dict(color="#6C63FF", width=2),
                hovertemplate="Run #%{x}<br>Score: %{y:.4f}<extra></extra>",
                name="Primary Score",
            ))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#C0C0D0"),
                title=dict(text="📈 Score Trend Across Runs", font=dict(color="#fff")),
                xaxis=dict(title="Run #", gridcolor="rgba(255,255,255,0.07)"),
                yaxis=dict(title="Score", range=[0, 1], gridcolor="rgba(255,255,255,0.07)"),
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT & MODEL
# ══════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("## ℹ️ About & Model Information")
    animated_divider()

    st.markdown("### 🤖 Model Details")
    render_model_info(MODEL_INFO)

    animated_divider()

    st.markdown("### 🏗️ Application Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Project Structure**
        ```
        NLP_Similarity_App/
        ├── app.py          ← Main entry point
        ├── similarity.py   ← Model & cosine similarity
        ├── graphs.py       ← All Plotly charts
        ├── components.py   ← UI components (HTML/CSS)
        ├── utils.py        ← Helpers & session state
        ├── style.css       ← Global glassmorphism CSS
        ├── requirements.txt
        └── README.md
        ```
        """)

    with col2:
        st.markdown("""
        **Key Design Decisions**
        - ✅ Zero preprocessing — raw text → model
        - ✅ `@st.cache_resource` for model singleton
        - ✅ L2-normalised embeddings → dot product = cosine
        - ✅ Fully modular — each concern in its own file
        - ✅ PEP-8 compliant with docstrings
        - ✅ No paid APIs — 100% local inference
        - ✅ Works offline after initial model download
        """)

    animated_divider()

    st.markdown("### 📦 Dependencies")
    deps = {
        "streamlit":            "Core web framework",
        "sentence-transformers":"NLP embedding model",
        "torch":                "PyTorch backend",
        "scikit-learn":         "PCA reduction",
        "plotly":               "Interactive charts",
        "numpy":                "Numerical operations",
        "pandas":               "Data tables & CSV export",
    }
    dep_col1, dep_col2 = st.columns(2)
    items = list(deps.items())
    half  = len(items) // 2
    with dep_col1:
        for k, v in items[:half]:
            st.markdown(f"- **`{k}`** — {v}")
    with dep_col2:
        for k, v in items[half:]:
            st.markdown(f"- **`{k}`** — {v}")

    animated_divider()

    st.markdown("""
    ### 🎓 Quiz Requirements Checklist

    | Requirement | Status |
    |---|---|
    | One FREE pretrained NLP model | ✅ `all-MiniLM-L6-v2` |
    | No preprocessing / tokenization / stemming / lemmatization | ✅ Raw text → model |
    | No model training | ✅ Pre-trained weights only |
    | No paid APIs | ✅ 100% local / HuggingFace |
    | Cosine similarity calculation | ✅ Via dot product of normalised vectors |
    | Top similar results | ✅ Ranked table + bar chart |
    | Embedding vectors displayed | ✅ Preview + JSON |
    | Animated Bar Chart | ✅ `graphs.py` |
    | Interactive Heatmap | ✅ Pairwise similarity matrix |
    | 2D PCA Embedding Plot | ✅ With hover, zoom, pan, legend |
    | Radar Chart | ✅ Paul's standards |
    | Gauge Chart | ✅ Primary score |
    | Distribution Chart | ✅ All pair scores |
    | Paul's 7 Standards | ✅ Expandable cards + scores |
    | Glassmorphism UI | ✅ `style.css` |
    | Animated gradient background | ✅ CSS keyframes |
    | Floating blobs | ✅ CSS pseudo-elements |
    | Dark theme | ✅ Full dark palette |
    | Download CSV | ✅ Results + history |
    | Session History | ✅ Persistent in session |
    | Theme toggle | ✅ Sidebar control |
    | Reset button | ✅ Main page |
    | Copy result button | ✅ `utils.clipboard_button` |
    | Model information page | ✅ This page |
    | Sidebar with navigation | ✅ 5-page sidebar |
    | Professional footer | ✅ Bottom of every page |
    """)


# ──────────────────────────────────────────────────────────────
#  Footer (all pages)
# ──────────────────────────────────────────────────────────────
animated_divider()
render_footer()
