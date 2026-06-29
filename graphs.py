"""
graphs.py
=========
All interactive Plotly visualisations for the NLP Similarity Explorer.
Includes:  Bar Chart, Heatmap, 2D PCA Scatter, Radar Chart, Gauge, Distribution.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# ── Shared palette ────────────────────────────────────────────
PALETTE      = ["#6C63FF", "#43E97B", "#FF6584", "#38F9D7", "#FF9A3C",
                 "#A855F7", "#EC4899", "#14B8A6", "#F59E0B", "#3B82F6"]
BG_COLOR     = "rgba(0,0,0,0)"
GRID_COLOR   = "rgba(255,255,255,0.07)"
TEXT_COLOR   = "#C0C0D0"
FONT_FAMILY  = "Inter, Space Grotesk, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(
        bgcolor="rgba(15,15,30,0.95)",
        bordercolor="rgba(108,99,255,0.5)",
        font=dict(family=FONT_FAMILY, color="#fff", size=12),
    ),
)


def _axis_style(**kwargs) -> dict:
    """Return a default axis style dict, merged with kwargs."""
    base = dict(
        gridcolor=GRID_COLOR,
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=11),
        zerolinecolor=GRID_COLOR,
    )
    base.update(kwargs)
    return base


# ──────────────────────────────────────────────────────────────
#  1. Animated Bar Chart — Top Similar Results
# ──────────────────────────────────────────────────────────────
def bar_chart_top_similar(results: list[dict]) -> go.Figure:
    """
    Animated horizontal bar chart showing Top-K similarity scores.

    Parameters
    ----------
    results : list[dict]  — output of similarity.get_top_similar()
    """
    if not results:
        return _empty_fig("No similarity results to display.")

    labels = [f"#{r['rank']} {r['text'][:40]}{'…' if len(r['text'])>40 else ''}"
              for r in results]
    scores = [r["score"] for r in results]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(results))]

    fig = go.Figure()

    # Animate bars from zero to final score
    fig.add_trace(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9,
        ),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.6f}<extra></extra>",
    ))

    # Animated frames: bars grow from 0 → final
    frames = []
    steps = 15
    for step in range(1, steps + 1):
        frac = step / steps
        frames.append(go.Frame(
            data=[go.Bar(x=[s * frac for s in scores], y=labels, orientation="h",
                         marker_color=colors, text=[f"{s*frac:.4f}" for s in scores],
                         textposition="outside")]
        ))
    fig.frames = frames

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🏆 Top Similar Results", font=dict(size=16, color="#fff"), x=0.02),
        xaxis=_axis_style(title="Cosine Similarity", range=[0, max(scores) * 1.18]),
        yaxis=_axis_style(autorange="reversed", tickfont=dict(size=10)),
        height=max(350, len(results) * 55 + 100),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12, x=0.98, xanchor="right",
            buttons=[dict(
                label="▶ Animate",
                method="animate",
                args=[None, dict(frame=dict(duration=60, redraw=True),
                                 fromcurrent=True, transition=dict(duration=0))]
            )],
            bgcolor="rgba(108,99,255,0.3)",
            bordercolor="rgba(108,99,255,0.6)",
            font=dict(color="#fff"),
        )],
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  2. Interactive Heatmap — Pairwise Similarity Matrix
# ──────────────────────────────────────────────────────────────
def heatmap_similarity(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    """
    Interactive heatmap of the N×N pairwise cosine similarity matrix.
    """
    if matrix is None or len(matrix) == 0:
        return _empty_fig("No matrix data available.")

    short_labels = [l[:25] + "…" if len(l) > 25 else l for l in labels]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=short_labels,
        y=short_labels,
        colorscale=[
            [0.0,  "rgba(10,10,20,1)"],
            [0.25, "rgba(108,99,255,0.4)"],
            [0.50, "rgba(108,99,255,0.75)"],
            [0.75, "rgba(67,233,123,0.85)"],
            [1.0,  "rgba(56,249,215,1)"],
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.4f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=9, color="rgba(255,255,255,0.8)"),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Similarity: %{z:.6f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Score", side="right"),
            tickfont=dict(color=TEXT_COLOR),
            thickness=14,
            outlinewidth=0,
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🔥 Pairwise Similarity Matrix", font=dict(size=16, color="#fff"), x=0.02),
        xaxis=_axis_style(tickangle=-40),
        yaxis=_axis_style(autorange="reversed"),
        height=max(400, len(labels) * 50 + 120),
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  3. 2D PCA Embedding Plot
# ──────────────────────────────────────────────────────────────
def pca_scatter(embeddings: np.ndarray, labels: list[str]) -> go.Figure:
    """
    Reduce embeddings to 2D with PCA and display as an interactive scatter.
    """
    if embeddings is None or len(embeddings) < 2:
        return _empty_fig("Need at least 2 embeddings for PCA.")

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

    x_vals = coords[:, 0]
    y_vals = coords[:, 1] if n_components > 1 else np.zeros(len(coords))

    short_labels = [l[:30] + "…" if len(l) > 30 else l for l in labels]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig = go.Figure()

    # Draw connecting lines (very faint)
    for i in range(len(x_vals)):
        for j in range(i + 1, len(x_vals)):
            fig.add_trace(go.Scatter(
                x=[x_vals[i], x_vals[j]],
                y=[y_vals[i], y_vals[j]],
                mode="lines",
                line=dict(color="rgba(108,99,255,0.12)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        text=short_labels,
        textposition="top center",
        textfont=dict(size=10, color=TEXT_COLOR),
        marker=dict(
            size=18,
            color=colors,
            line=dict(color="rgba(255,255,255,0.3)", width=1.5),
            symbol="circle",
            opacity=0.9,
        ),
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.4f}<br>PC2: %{y:.4f}<extra></extra>",
        name="Embeddings",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"🧠 2D PCA Embedding Space  "
                 f"<span style='font-size:12px;color:{TEXT_COLOR}'>"
                 f"(PC1={explained[0]:.1%}  "
                 f"PC2={explained[1]:.1%})</span>" if n_components > 1 else
                 "🧠 2D PCA Embedding Space",
            font=dict(size=15, color="#fff"), x=0.02,
        ),
        xaxis=_axis_style(title=f"PC1 ({explained[0]:.1%} variance)" if n_components > 1 else "PC1"),
        yaxis=_axis_style(title=f"PC2 ({explained[1]:.1%} variance)" if n_components > 1 else "PC2"),
        showlegend=False,
        height=480,
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  4. Radar Chart — Paul's Critical Thinking Dimensions
# ──────────────────────────────────────────────────────────────
def radar_chart(scores_dict: dict[str, float]) -> go.Figure:
    """
    Radar (spider) chart for Paul's intellectual standards scores.

    Parameters
    ----------
    scores_dict : {"Clarity": 0.85, "Accuracy": 0.72, ...}
    """
    categories = list(scores_dict.keys())
    values     = list(scores_dict.values())
    # Close the loop
    categories_loop = categories + [categories[0]]
    values_loop     = values     + [values[0]]

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=values_loop,
        theta=categories_loop,
        fill="toself",
        fillcolor="rgba(108,99,255,0.18)",
        line=dict(color="#6C63FF", width=2),
        marker=dict(size=8, color=PALETTE[:len(categories)]),
        name="Score",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🎯 Paul's Critical Thinking Standards", font=dict(size=15, color="#fff"), x=0.02),
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=GRID_COLOR,
                tickfont=dict(color=TEXT_COLOR, size=9),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR,
                tickfont=dict(color=TEXT_COLOR, size=11),
            ),
        ),
        showlegend=False,
        height=420,
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  5. Gauge Chart — Single Similarity Score
# ──────────────────────────────────────────────────────────────
def gauge_chart(score: float, title: str = "Similarity Score") -> go.Figure:
    """
    Gauge chart for a single cosine similarity value.
    """
    # Color zones
    steps = [
        dict(range=[0.00, 0.35], color="rgba(255,101,132,0.18)"),
        dict(range=[0.35, 0.55], color="rgba(255,154,60,0.18)"),
        dict(range=[0.55, 0.75], color="rgba(108,99,255,0.18)"),
        dict(range=[0.75, 0.90], color="rgba(56,249,215,0.18)"),
        dict(range=[0.90, 1.00], color="rgba(67,233,123,0.25)"),
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(suffix="", valueformat=".4f", font=dict(size=28, color="#fff")),
        delta=dict(reference=0.5, valueformat=".4f",
                   increasing=dict(color="#43E97B"),
                   decreasing=dict(color="#FF6584")),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor=TEXT_COLOR,
                      tickfont=dict(size=10, color=TEXT_COLOR)),
            bar=dict(color="rgba(108,99,255,0.85)", thickness=0.25),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=1,
            bordercolor=GRID_COLOR,
            steps=steps,
            threshold=dict(
                line=dict(color="rgba(67,233,123,0.9)", width=3),
                thickness=0.75,
                value=score,
            ),
        ),
        title=dict(text=title, font=dict(size=14, color=TEXT_COLOR)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  6. Distribution Chart — Score Histogram
# ──────────────────────────────────────────────────────────────
def distribution_chart(scores: list[float]) -> go.Figure:
    """
    Histogram + KDE of similarity score distribution.
    """
    if len(scores) < 2:
        return _empty_fig("Not enough scores for distribution.")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=max(5, len(scores) // 2),
        marker=dict(
            color="rgba(108,99,255,0.5)",
            line=dict(color="rgba(108,99,255,0.9)", width=1.5),
        ),
        name="Score Count",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    # Add mean line
    mean_val = float(np.mean(scores))
    fig.add_vline(
        x=mean_val,
        line=dict(color="#43E97B", width=2, dash="dash"),
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_font=dict(color="#43E97B", size=11),
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📊 Score Distribution", font=dict(size=15, color="#fff"), x=0.02),
        xaxis=_axis_style(title="Similarity Score"),
        yaxis=_axis_style(title="Count"),
        height=320,
        bargap=0.08,
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  Helper — Empty placeholder figure
# ──────────────────────────────────────────────────────────────
def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=TEXT_COLOR),
    )
    fig.update_layout(**LAYOUT_BASE, height=300)
    return fig
