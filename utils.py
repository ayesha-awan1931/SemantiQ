"""
utils.py
========
Utility helpers:
  - Session history management
  - CSV download generation
  - Paul's Critical Thinking score derivation
  - Copy-to-clipboard helper
  - Misc formatting helpers
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────────────────────
#  Session State Initialisation
# ──────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all required session-state keys on first run."""
    defaults = {
        "history":        [],          # list of result dicts
        "last_result":    None,        # latest analysis result
        "theme":          "dark",      # theme toggle (always dark for now)
        "run_count":      0,           # number of analyses run
        "total_ms":       0.0,         # cumulative inference time
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ──────────────────────────────────────────────────────────────
#  History
# ──────────────────────────────────────────────────────────────
def save_to_history(
    texts: list[str],
    scores: list[float],
    primary_score: float,
    elapsed_ms: float,
) -> None:
    """
    Append a result entry to the in-session history list.
    Keeps the most recent 50 entries.
    """
    entry = {
        "run":           st.session_state.run_count,
        "texts":         texts,
        "scores":        [round(s, 6) for s in scores],
        "primary_score": round(primary_score, 6),
        "elapsed_ms":    round(elapsed_ms, 2),
        "n_texts":       len(texts),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:50]
    st.session_state.run_count += 1
    st.session_state.total_ms += elapsed_ms
    st.session_state.last_result = entry


def get_history_df() -> pd.DataFrame:
    """Return session history as a tidy DataFrame."""
    if not st.session_state.history:
        return pd.DataFrame()

    rows = []
    for entry in st.session_state.history:
        rows.append({
            "Run #":          entry["run"],
            "Texts":          " | ".join(entry["texts"]),
            "Primary Score":  entry["primary_score"],
            "Avg Score":      round(float(np.mean(entry["scores"])), 6) if entry["scores"] else 0.0,
            "# Texts":        entry["n_texts"],
            "Inference (ms)": entry["elapsed_ms"],
        })
    return pd.DataFrame(rows)


def clear_history() -> None:
    """Clear all session history."""
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.run_count = 0
    st.session_state.total_ms = 0.0


# ──────────────────────────────────────────────────────────────
#  CSV Export
# ──────────────────────────────────────────────────────────────
def results_to_csv(
    texts: list[str],
    top_results: list[dict],
    matrix: np.ndarray | None,
) -> bytes:
    """
    Build a multi-sheet-like CSV for download.
    Returns UTF-8 encoded bytes.
    """
    buf = io.StringIO()

    # ── Section 1: Top Similar Results ──
    buf.write("# TOP SIMILAR RESULTS\n")
    if top_results:
        df_top = pd.DataFrame(top_results)[["rank", "text", "score"]]
        df_top.columns = ["Rank", "Text", "Score"]
        df_top.to_csv(buf, index=False)
    else:
        buf.write("No results available.\n")

    buf.write("\n\n# PAIRWISE SIMILARITY MATRIX\n")
    if matrix is not None and len(matrix) > 0:
        df_matrix = pd.DataFrame(
            matrix,
            index=[f"T{i+1}: {t[:30]}" for i, t in enumerate(texts)],
            columns=[f"T{i+1}: {t[:30]}" for i, t in enumerate(texts)],
        )
        df_matrix.to_csv(buf)
    else:
        buf.write("Matrix not available.\n")

    return buf.getvalue().encode("utf-8")


def history_to_csv() -> bytes:
    """Export the session history DataFrame as CSV bytes."""
    df = get_history_df()
    if df.empty:
        return b"No history available."
    return df.to_csv(index=False).encode("utf-8")


# ──────────────────────────────────────────────────────────────
#  Paul's Critical Thinking Score Derivation
# ──────────────────────────────────────────────────────────────
def compute_paul_scores(
    primary_score: float,
    all_scores: list[float],
    texts: list[str],
) -> dict[str, float]:
    """
    Derive proxy scores for each of Paul's seven intellectual standards
    from the NLP similarity results.  Returns values in [0, 1].

    These are heuristic proxies, not ground-truth measurements.
    """
    n   = len(texts)
    avg = float(np.mean(all_scores)) if all_scores else primary_score
    std = float(np.std(all_scores))  if len(all_scores) > 1 else 0.0

    # Average text length (normalised, capped at 1)
    avg_len_score = min(1.0, np.mean([len(t) for t in texts]) / 200)

    # Vocabulary richness proxy (unique words / total words, averaged)
    richness_scores = []
    for t in texts:
        words = t.lower().split()
        if words:
            richness_scores.append(len(set(words)) / len(words))
    richness = float(np.mean(richness_scores)) if richness_scores else 0.5

    scores = {
        # Clarity    — longer texts tend to be more explicit
        "Clarity":      round(min(1.0, avg_len_score * 1.2), 4),
        # Accuracy   — high similarity to query suggests accurate retrieval
        "Accuracy":     round(min(1.0, primary_score * 1.05), 4),
        # Precision  — low std means consistently precise matching
        "Precision":    round(max(0.0, 1.0 - std * 2), 4),
        # Relevance  — average score reflects overall relevance
        "Relevance":    round(min(1.0, avg * 1.1), 4),
        # Logic      — balanced scores across texts signal coherent logic
        "Logic":        round(max(0.0, 1.0 - std * 1.5), 4),
        # Significance — how much above random the score is (random ≈ 0.3)
        "Significance": round(min(1.0, max(0.0, (primary_score - 0.3) / 0.7)), 4),
        # Fairness   — vocabulary richness as proxy for balanced expression
        "Fairness":     round(min(1.0, richness * 1.3), 4),
    }
    return scores


# ──────────────────────────────────────────────────────────────
#  Performance Statistics
# ──────────────────────────────────────────────────────────────
def render_performance_stats() -> None:
    """Display running performance stats in the sidebar."""
    if st.session_state.run_count == 0:
        return

    avg_ms = st.session_state.total_ms / st.session_state.run_count
    st.sidebar.markdown("### 📈 Session Stats")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Runs", st.session_state.run_count)
    col2.metric("Avg (ms)", f"{avg_ms:.0f}")


# ──────────────────────────────────────────────────────────────
#  Copy-to-Clipboard Button
# ──────────────────────────────────────────────────────────────
def clipboard_button(text: str, label: str = "📋 Copy") -> None:
    """
    Render a button that copies `text` to the clipboard via JS.
    Falls back silently if clipboard API is unavailable.
    """
    escaped = text.replace("`", "\\`").replace("\\", "\\\\")
    button_id = f"clip_{abs(hash(text)) % 100000}"

    html = f"""
    <button
        id="{button_id}"
        onclick="navigator.clipboard.writeText(`{escaped}`)
                 .then(()=>{{
                     let b=document.getElementById('{button_id}');
                     b.innerText='✅ Copied!';
                     setTimeout(()=>b.innerText='{label}',2000);
                 }})"
        style="background:rgba(108,99,255,0.2);border:1px solid rgba(108,99,255,0.4);
               border-radius:50px;color:#8B85FF;padding:0.35rem 1rem;
               font-size:0.82rem;font-weight:600;cursor:pointer;
               transition:all 0.2s;font-family:Inter,sans-serif;">
        {label}
    </button>
    """
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Misc Helpers
# ──────────────────────────────────────────────────────────────
def truncate(text: str, max_len: int = 60) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text


def format_vector(vec: np.ndarray, n: int = 8) -> str:
    """Format a numpy vector to a compact string."""
    vals = ", ".join(f"{v:.5f}" for v in vec[:n])
    return f"[{vals}, … +{len(vec)-n} more]"
