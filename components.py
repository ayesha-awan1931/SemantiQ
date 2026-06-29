"""
components.py
=============
Reusable Streamlit + HTML/CSS UI components for the NLP Similarity Explorer.
"""

import streamlit as st
import pandas as pd


# ──────────────────────────────────────────────────────────────
#  Load External CSS
# ──────────────────────────────────────────────────────────────
def inject_css(css_path: str = "style.css") -> None:
    """Inject the external CSS file into the Streamlit page."""
    try:
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Gracefully ignore if running from a different working directory


# ──────────────────────────────────────────────────────────────
#  Hero Header
# ──────────────────────────────────────────────────────────────
def render_hero_header() -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="badge">✦ AI-POWERED NLP TOOL</div>
        <h1>Semantic Similarity Explorer</h1>
        <p>
            Enter any words, sentences, or paragraphs and instantly discover how
            semantically similar they are using state-of-the-art sentence embeddings.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Animated Divider
# ──────────────────────────────────────────────────────────────
def animated_divider() -> None:
    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Metrics Row
# ──────────────────────────────────────────────────────────────
def render_metrics(score: float, label: str, colour: str,
                   n_texts: int, elapsed_ms: float, dim: int) -> None:
    """Four-column metrics row displayed as glass cards."""
    percent = int(score * 100)

    html = f"""
    <div class="metrics-row">
        <div class="glass-card">
            <h3>Similarity Score</h3>
            <div class="metric-value">{score:.4f}</div>
            <div style="margin-top:0.4rem;">
                <span class="tag-pill tag-{'green' if score>=0.75 else 'blue' if score>=0.55 else 'pink'}">{label}</span>
            </div>
        </div>
        <div class="glass-card">
            <h3>Confidence %</h3>
            <div class="metric-value">{percent}%</div>
            <div style="color:#9090B0;font-size:0.82rem;margin-top:0.3rem;">cosine similarity</div>
        </div>
        <div class="glass-card">
            <h3>Texts Analysed</h3>
            <div class="metric-value">{n_texts}</div>
            <div style="color:#9090B0;font-size:0.82rem;margin-top:0.3rem;">inputs compared</div>
        </div>
        <div class="glass-card">
            <h3>Inference Time</h3>
            <div class="metric-value">{elapsed_ms:.0f}<span style="font-size:1rem">ms</span></div>
            <div style="color:#9090B0;font-size:0.82rem;margin-top:0.3rem;">embedding dim: {dim}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Score Progress Bar
# ──────────────────────────────────────────────────────────────
def render_score_bar(score: float, label: str = "Similarity Score") -> None:
    st.caption(f"**{label}:** {score:.4f}")
    st.progress(int(score * 100))


# ──────────────────────────────────────────────────────────────
#  Top Similar Results Table
# ──────────────────────────────────────────────────────────────
def render_top_results(results: list[dict]) -> None:
    if not results:
        st.info("No similarity results to show. Run a comparison first.")
        return

    rows = []
    for r in results:
        label_color = "#43E97B" if r["score"] >= 0.75 else "#6C63FF" if r["score"] >= 0.5 else "#FF6584"
        rows.append({
            "Rank": f"#{r['rank']}",
            "Text Snippet": r["text"][:80] + ("…" if len(r["text"]) > 80 else ""),
            "Score": f"{r['score']:.6f}",
            "Level": "High" if r["score"] >= 0.75 else "Medium" if r["score"] >= 0.5 else "Low",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
#  Embedding Vector Display
# ──────────────────────────────────────────────────────────────
def render_embedding_preview(embedding, label: str = "Embedding", max_dims: int = 16) -> None:
    """Show the first `max_dims` dimensions of an embedding vector."""
    snippet = ", ".join(f"{v:.5f}" for v in embedding[:max_dims])
    full_dim = len(embedding)
    st.markdown(f"""
    <div class="vector-display">
<b style="color:#8B85FF">{label}</b>  (dim={full_dim})
[ {snippet}, … ]
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Paul's Critical Thinking Expandable Cards
# ──────────────────────────────────────────────────────────────
STANDARDS = {
    "Clarity": {
        "icon": "🔍",
        "desc": "The reasoning must be understandable and free from ambiguity.",
        "detail": (
            "Clarity asks: *Could you elaborate or give an example?* "
            "In the context of semantic similarity, clarity means the input texts "
            "should express ideas without vagueness so the model can form precise "
            "vector representations. Ambiguous or unclear language reduces "
            "embedding quality and lowers meaningful similarity scores."
        ),
    },
    "Accuracy": {
        "icon": "✅",
        "desc": "Information must be true and correct.",
        "detail": (
            "Accuracy asks: *Is this really true? How could we check?* "
            "Cosine similarity is mathematically accurate — it measures the angle "
            "between embedding vectors. The accuracy of the *semantic* result depends "
            "on whether the model has learned correct linguistic representations, "
            "which all-MiniLM-L6-v2 achieves through contrastive training on large corpora."
        ),
    },
    "Precision": {
        "icon": "🎯",
        "desc": "Information must be specific and detailed, not vague.",
        "detail": (
            "Precision asks: *Could you be more specific?* "
            "Similarity scores are expressed to six decimal places, giving precise "
            "numeric grounding. The 384-dimensional embedding space encodes fine-grained "
            "semantic distinctions between words like 'happy', 'joyful', and 'content' "
            "— each maps to a slightly different region."
        ),
    },
    "Relevance": {
        "icon": "🔗",
        "desc": "Information must relate directly to the question at hand.",
        "detail": (
            "Relevance asks: *How does this bear on the issue?* "
            "The similarity score is directly relevant to measuring semantic overlap. "
            "All graphs, metrics, and PCA plots are derived from the same embeddings, "
            "ensuring every visual element contributes relevant insight about the "
            "relationship between the input texts."
        ),
    },
    "Logic": {
        "icon": "⚙️",
        "desc": "Reasoning must follow from evidence and be internally consistent.",
        "detail": (
            "Logic asks: *Does this follow? Does it make sense together?* "
            "Cosine similarity follows logically from the dot product of unit vectors. "
            "Pre-normalising embeddings ensures that score = dot(a, b), which is "
            "bounded in [0,1] and monotonically reflects semantic closeness — a "
            "logically consistent and mathematically provable property."
        ),
    },
    "Significance": {
        "icon": "🌟",
        "desc": "The most important information must be prioritised.",
        "detail": (
            "Significance asks: *Is this the most important issue to consider?* "
            "Top-K ranking surfaces the most semantically significant matches, "
            "not just any match. PCA shows which variance is most significant in the "
            "embedding space, and the gauge chart visually emphasises whether the "
            "obtained similarity crosses practically meaningful thresholds."
        ),
    },
    "Fairness": {
        "icon": "⚖️",
        "desc": "Reasoning must be open-minded and free from bias.",
        "detail": (
            "Fairness asks: *Am I considering all relevant viewpoints?* "
            "The model treats all inputs equally — there is no weighting by text "
            "length, vocabulary richness, or author. Multiple texts are analysed "
            "in a full pairwise matrix so no text is privileged as the sole reference, "
            "giving a balanced view of the entire semantic landscape."
        ),
    },
}


def render_critical_thinking_cards(scores: dict | None = None) -> None:
    """
    Render Paul's Critical Thinking Standards as expandable glass cards.

    Parameters
    ----------
    scores : optional dict mapping standard name → float score in [0,1]
    """
    st.markdown("### 🧪 Paul's Critical Thinking Standards")
    st.caption(
        "Each intellectual standard is evaluated in the context of this "
        "NLP similarity analysis."
    )

    for name, info in STANDARDS.items():
        score_val = scores.get(name, None) if scores else None
        score_str = f"  —  **{score_val:.3f}**" if score_val is not None else ""

        with st.expander(f"{info['icon']}  **{name}**{score_str}  —  {info['desc']}"):
            st.markdown(info["detail"])
            if score_val is not None:
                st.progress(int(score_val * 100))


# ──────────────────────────────────────────────────────────────
#  Model Information Card
# ──────────────────────────────────────────────────────────────
def render_model_info(model_info: dict) -> None:
    st.markdown(f"""
    <div class="glass-card" style="padding:1.2rem 1.5rem;">
        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;">
            <span style="font-size:1.6rem;">🤖</span>
            <div>
                <div style="font-weight:700;font-size:1rem;color:#F0F0FF">
                    {model_info['name']}
                </div>
                <div style="font-size:0.8rem;color:#9090B0">{model_info['provider']}</div>
            </div>
        </div>
        <p style="color:#C0C0D0;font-size:0.88rem;line-height:1.65;margin:0 0 0.8rem">
            {model_info['description']}
        </p>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
            <span class="tag-pill tag-blue">dim={model_info['dimension']}</span>
            <span class="tag-pill tag-green">max {model_info['max_tokens']} tokens</span>
            <span class="tag-pill tag-cyan">{model_info['size']}</span>
            <span class="tag-pill tag-pink">{model_info['license']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Sidebar — Navigation & Tips
# ──────────────────────────────────────────────────────────────
def render_sidebar(page: str) -> str:
    """
    Render the sidebar with navigation and tips.
    Returns the selected page name.
    """
    with st.sidebar:
        # Logo / Title
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 0.5rem;">
            <div style="font-size:2.2rem;">🔮</div>
            <div style="font-weight:800;font-size:1.05rem;
                        background:linear-gradient(135deg,#8B85FF,#43E97B);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                NLP Similarity<br>Explorer
            </div>
            <div style="font-size:0.72rem;color:#6060A0;margin-top:0.2rem;">
                Powered by sentence-transformers
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "🏠  Main Analysis":   "Main",
            "📊  Visualisations":  "Graphs",
            "🧪  Critical Thinking": "Critical",
            "📜  Session History": "History",
            "ℹ️  About & Model":   "About",
        }

        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Tips
        st.markdown("### 💡 Tips")
        with st.expander("How to get best results"):
            st.markdown("""
            - Use **full sentences** for richer embeddings
            - Compare **3–5 texts** at once for a useful heatmap
            - Scores **> 0.75** indicate strong semantic overlap
            - Scores **< 0.35** suggest unrelated topics
            """)

        with st.expander("About Cosine Similarity"):
            st.markdown("""
            Cosine similarity measures the **angle** between two vectors.
            - **1.0** = identical direction (semantically equivalent)
            - **0.5** = 60° apart (related but different)
            - **0.0** = 90° apart (completely unrelated)
            """)

        st.markdown("---")

        # Model badge
        st.markdown("""
        <div style="background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.3);
                    border-radius:10px;padding:0.7rem;text-align:center;font-size:0.78rem;color:#9090B0;">
            🔬 <b style="color:#8B85FF">Model:</b><br>
            all-MiniLM-L6-v2<br>
            <span style="color:#43E97B">● Free  ● Local  ● 384-dim</span>
        </div>
        """, unsafe_allow_html=True)

        return pages[selected]


# ──────────────────────────────────────────────────────────────
#  Loading Spinner (HTML overlay)
# ──────────────────────────────────────────────────────────────
def render_loading_spinner(message: str = "Computing embeddings…") -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:2rem;">
        <div class="custom-spinner"></div>
        <div style="color:#9090B0;font-size:0.9rem;margin-top:0.8rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Animated Success Banner
# ──────────────────────────────────────────────────────────────
def render_success_banner(score: float, label: str) -> None:
    color = "#43E97B" if score >= 0.75 else "#6C63FF" if score >= 0.5 else "#FF6584"
    st.markdown(f"""
    <div style="background:rgba(67,233,123,0.08);border:1px solid rgba(67,233,123,0.3);
                border-radius:14px;padding:1rem 1.5rem;
                animation:fadeInUp 0.5s ease forwards;text-align:center;margin:1rem 0;">
        <span style="font-size:1.5rem;">✨</span>
        <span style="font-weight:700;font-size:1rem;color:{color};margin:0 0.5rem;">
            Analysis Complete!
        </span>
        <span style="color:#C0C0D0;font-size:0.9rem;">
            Similarity: <b style="color:{color}">{score:.4f}</b> — {label}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown("""
    <div class="custom-footer">
        <b>NLP Semantic Similarity Explorer</b> &nbsp;·&nbsp;
        Built with
        <a href="https://streamlit.io" target="_blank">Streamlit</a>,
        <a href="https://www.sbert.net" target="_blank">SentenceTransformers</a> &amp;
        <a href="https://plotly.com" target="_blank">Plotly</a>
        &nbsp;·&nbsp;
        Model: <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank">
        all-MiniLM-L6-v2</a>
        &nbsp;·&nbsp; Free &amp; Open Source
        <br><br>
        <span style="font-size:0.72rem;opacity:0.5;">
            © 2025 — BS Artificial Intelligence — Quiz Project
        </span>
    </div>
    """, unsafe_allow_html=True)
