"""
FoodGuard AI - Genomic Surveillance Dashboard
==============================================

A professional, interactive dashboard for visualizing whole-proteome ESM-2
embeddings and enabling geometry-aware triage of foodborne bacterial genomes.

This dashboard serves as an interactive companion to the manuscript:
"Whole-Proteome ESM-2 Embeddings Recover Taxonomy and Enable Geometry-Aware
Triage of Foodborne Bacterial Genomes"

Usage:
    streamlit run foodguard/dashboard.py

Authors: Jay Gutierrez & Javier Correa Alvarez
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# Configuration & Constants
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "foodguard" / "analysis"
ENRICHED_PARQUET = ANALYSIS_DIR / "genome_embeddings_enriched.parquet"
DIM_CACHE = ANALYSIS_DIR / "dim_reduction_cache.npz"

# Professional color palette - Food Safety themed
COLORS = {
    "primary": "#1B4D3E",        # Deep forest green (safety, trust)
    "secondary": "#2E7D5A",      # Medium green
    "accent": "#4CAF50",         # Bright green (success)
    "warning": "#FF9800",        # Amber (caution)
    "danger": "#D32F2F",         # Red (alert)
    "info": "#0288D1",           # Blue (information)
    "background": "#FAFAFA",     # Light gray
    "surface": "#FFFFFF",        # White
    "text_primary": "#212121",   # Dark gray
    "text_secondary": "#757575", # Medium gray
    "border": "#E0E0E0",         # Light border
}

# Species color mapping for consistent visualization
SPECIES_COLORS = {
    "Salmonella_enterica": "#E53935",      # Red
    "Listeria_monocytogenes": "#D81B60",   # Pink
    "E_coli_O157H7": "#8E24AA",            # Purple
    "E_coli_nonpathogenic": "#5E35B1",     # Deep Purple
    "Bacillus_subtilis": "#1E88E5",        # Blue
    "Citrobacter_koseri": "#00ACC1",       # Cyan
    "Listeria_innocua": "#43A047",         # Green
    "Escherichia_fergusonii": "#7CB342",   # Light Green
    "Citrobacter_freundii": "#FDD835",     # Yellow
}

# Pathogenicity colors
PATHOGEN_COLORS = {
    "pathogenic": "#D32F2F",
    "non-pathogenic": "#388E3C",
}

# Risk level configuration
RISK_CONFIG = {
    "high": {
        "color": "#D32F2F",
        "bg": "#FFEBEE",
        "icon": "🚨",
        "label": "HIGH RISK",
        "action": "ESCALATE FOR REVIEW"
    },
    "medium": {
        "color": "#FF9800",
        "bg": "#FFF3E0",
        "icon": "⚠️",
        "label": "MEDIUM RISK",
        "action": "ADDITIONAL ANALYSIS RECOMMENDED"
    },
    "low": {
        "color": "#388E3C",
        "bg": "#E8F5E9",
        "icon": "✓",
        "label": "LOW RISK",
        "action": "PROCEED WITH STANDARD PROTOCOL"
    }
}

# Manuscript key statistics
MANUSCRIPT_STATS = {
    "total_genomes": 21657,
    "total_taxa": 9,
    "pca_var_pc1": 76.56,
    "pca_var_pc2": 10.95,
    "pca_var_cumulative_3": 94.3,
    "mean_species_homophily": 0.9923,
    "mean_pathogen_homophily": 0.9929,
    "n_clusters": 34,
    "cluster_noise_pct": 35.6,
    "cluster_purity": 0.97,
    "silhouette": 0.555,
    "strict_outliers": 252,
    "strict_outlier_pct": 1.2,
    "ecoli_accuracy": 0.9842,
    "listeria_accuracy": 0.9992,
}

# =============================================================================
# Data Classes for Risk Assessment
# =============================================================================

@dataclass
class GeometryScores:
    """Geometry-derived confidence scores from embedding space."""
    homophily_confidence: float
    cluster_purity: float
    outlier_risk: float
    centroid_proximity: float
    k_neighbors: int = 20
    cluster_id: int = -1

@dataclass
class RiskAssessment:
    """Combined risk assessment result."""
    crs: float
    risk_level: str
    confidence: str
    decision: str
    reason: str
    components: dict

# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=3600)
def load_genome_data() -> tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load enriched genome embeddings with dimensionality reduction coordinates."""
    if not ENRICHED_PARQUET.exists():
        return create_demo_data()

    df = pd.read_parquet(ENRICHED_PARQUET)

    # Load dimension reduction cache if available
    pca_coords = None
    umap_coords = None
    if DIM_CACHE.exists():
        cache = np.load(DIM_CACHE)
        if "pca_result" in cache:
            pca_coords = cache["pca_result"]
        if "umap_result" in cache:
            umap_coords = cache["umap_result"]

    return df, pca_coords, umap_coords


def create_demo_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create demonstration data when real data is not available."""
    np.random.seed(42)
    n_samples = 500

    species_list = list(SPECIES_COLORS.keys())
    pathogenic_species = ["Salmonella_enterica", "Listeria_monocytogenes", "E_coli_O157H7"]

    data = {
        "genome_id": [f"GENOME_{i:05d}" for i in range(n_samples)],
        "species": np.random.choice(species_list, n_samples),
        "proteins": np.random.randint(3000, 6000, n_samples),
        "species_homophily": np.clip(np.random.beta(50, 1, n_samples), 0, 1),
        "pathogenicity_homophily": np.clip(np.random.beta(50, 1, n_samples), 0, 1),
        "cluster": np.random.randint(-1, 34, n_samples),
        "is_outlier": np.random.random(n_samples) < 0.05,
        "is_outlier_strict": np.random.random(n_samples) < 0.012,
    }

    df = pd.DataFrame(data)
    df["pathogenicity_label"] = df["species"].apply(
        lambda x: "pathogenic" if x in pathogenic_species else "non-pathogenic"
    )

    # Create synthetic PCA coordinates with species clustering
    pca_coords = np.zeros((n_samples, 2))
    for i, species in enumerate(species_list):
        mask = df["species"] == species
        n_sp = mask.sum()
        center = np.array([np.cos(2 * np.pi * i / len(species_list)) * 20,
                         np.sin(2 * np.pi * i / len(species_list)) * 10])
        pca_coords[mask] = center + np.random.randn(n_sp, 2) * 2

    # UMAP coordinates (similar structure, different scale)
    umap_coords = pca_coords * 0.5 + np.random.randn(n_samples, 2) * 0.5

    return df, pca_coords, umap_coords


# =============================================================================
# Risk Assessment Logic
# =============================================================================

def compute_risk_assessment(
    genome_row: pd.Series,
    posture: str = "recall_high"
) -> RiskAssessment:
    """
    Compute comprehensive risk assessment using geometry-gated scoring.

    This implements the hierarchical risk scoring from the manuscript:
    1. Geometry gate: Check homophily/cluster confidence
    2. If geometry passes: Weighted fusion of pathogenicity + evidence
    3. Decision based on posture-specific thresholds
    """
    # Extract geometry scores
    homophily = genome_row.get("species_homophily", 0.95)
    pathogen_homophily = genome_row.get("pathogenicity_homophily", 0.95)
    is_outlier = genome_row.get("is_outlier_strict", False)
    is_pathogenic = genome_row.get("pathogenicity_label", "") == "pathogenic"
    cluster_id = genome_row.get("cluster", -1)

    # Compute geometry confidence
    outlier_penalty = 0.3 if is_outlier else 0.0
    geometry_confidence = min(homophily, pathogen_homophily) * (1 - outlier_penalty)

    # Posture-specific thresholds
    thresholds = {
        "recall_high": {"geo_gate": 0.70, "high": 0.65, "medium": 0.35},
        "balanced": {"geo_gate": 0.75, "high": 0.75, "medium": 0.50},
        "precision_high": {"geo_gate": 0.85, "high": 0.85, "medium": 0.65},
    }
    t = thresholds.get(posture, thresholds["balanced"])

    # Stage 1: Geometry gate
    if geometry_confidence < t["geo_gate"]:
        if posture == "recall_high":
            return RiskAssessment(
                crs=0.80,
                risk_level="high",
                confidence="low",
                decision="defer_to_expert",
                reason="Low geometry confidence - requires expert review",
                components={
                    "geometry_confidence": geometry_confidence,
                    "homophily": homophily,
                    "outlier_risk": outlier_penalty,
                    "pathogenic_prior": 1.0 if is_pathogenic else 0.0
                }
            )
        else:
            return RiskAssessment(
                crs=0.50,
                risk_level="medium",
                confidence="low",
                decision="secondary_analysis",
                reason="Boundary case - additional analysis recommended",
                components={
                    "geometry_confidence": geometry_confidence,
                    "homophily": homophily,
                    "outlier_risk": outlier_penalty,
                    "pathogenic_prior": 1.0 if is_pathogenic else 0.0
                }
            )

    # Stage 2: Compute CRS with geometry-weighted fusion
    ps = 0.90 if is_pathogenic else 0.10  # Pathogenicity score from prior
    evidence_score = 0.5 if is_pathogenic else 0.2  # Placeholder

    # Weighted fusion with outlier boost
    base_crs = 0.6 * ps + 0.25 * geometry_confidence + 0.15 * evidence_score
    if is_outlier:
        base_crs = min(1.0, base_crs + 0.15)  # Outlier boost for caution

    crs = np.clip(base_crs, 0, 1)

    # Stage 3: Decision logic
    if crs >= t["high"]:
        risk_level = "high"
        decision = "escalate" if is_pathogenic else "review"
    elif crs >= t["medium"]:
        risk_level = "medium"
        decision = "review"
    else:
        risk_level = "low"
        decision = "proceed"

    # Confidence based on geometry
    if geometry_confidence >= 0.95:
        confidence = "high"
    elif geometry_confidence >= 0.85:
        confidence = "medium"
    else:
        confidence = "low"

    return RiskAssessment(
        crs=crs,
        risk_level=risk_level,
        confidence=confidence,
        decision=decision,
        reason=f"Geometry-validated {'pathogenic' if is_pathogenic else 'non-pathogenic'} assessment",
        components={
            "geometry_confidence": geometry_confidence,
            "homophily": homophily,
            "pathogenicity_score": ps,
            "evidence_score": evidence_score,
            "outlier_risk": outlier_penalty,
        }
    )


# =============================================================================
# Visualization Components
# =============================================================================

def create_pca_figure(
    df: pd.DataFrame,
    pca_coords: np.ndarray,
    color_by: str = "species",
    highlight_genome: Optional[str] = None
) -> go.Figure:
    """Create interactive PCA scatter plot."""
    plot_df = df.copy()
    plot_df["PC1"] = pca_coords[:, 0]
    plot_df["PC2"] = pca_coords[:, 1]

    if color_by == "species":
        color_map = SPECIES_COLORS
        color_col = "species"
    else:
        color_map = PATHOGEN_COLORS
        color_col = "pathogenicity_label"

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color=color_col,
        color_discrete_map=color_map,
        hover_data=["genome_id", "species", "pathogenicity_label", "species_homophily"],
        opacity=0.7,
    )

    # Highlight selected genome
    if highlight_genome:
        highlight_row = plot_df[plot_df["genome_id"] == highlight_genome]
        if not highlight_row.empty:
            fig.add_trace(go.Scatter(
                x=highlight_row["PC1"],
                y=highlight_row["PC2"],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#FFD700",
                    symbol="star",
                    line=dict(width=2, color="#000000")
                ),
                name="Selected Genome",
                hoverinfo="skip"
            ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"PC1 ({MANUSCRIPT_STATS['pca_var_pc1']:.1f}% variance)",
        yaxis_title=f"PC2 ({MANUSCRIPT_STATS['pca_var_pc2']:.1f}% variance)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=20, t=40, b=100),
        height=500,
    )

    return fig


def create_homophily_histogram(df: pd.DataFrame, metric: str = "species") -> go.Figure:
    """Create homophily distribution histogram."""
    col = f"{metric}_homophily"

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[col],
        nbinsx=50,
        marker_color=COLORS["primary"],
        opacity=0.8,
        name="All Genomes"
    ))

    # Add boundary case threshold line
    fig.add_vline(x=0.9, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Boundary threshold (0.9)")

    mean_val = df[col].mean()
    fig.add_vline(x=mean_val, line_dash="solid", line_color=COLORS["danger"],
                  annotation_text=f"Mean: {mean_val:.4f}")

    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"{metric.title()} Homophily (k=20)",
        yaxis_title="Count",
        showlegend=False,
        height=350,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig


def create_species_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create species distribution bar chart."""
    species_counts = df["species"].value_counts().sort_values(ascending=True)

    colors = [SPECIES_COLORS.get(sp, "#888888") for sp in species_counts.index]

    fig = go.Figure(go.Bar(
        y=species_counts.index.str.replace("_", " "),
        x=species_counts.values,
        orientation="h",
        marker_color=colors,
        text=species_counts.values,
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Number of Genomes",
        yaxis_title="",
        height=400,
        margin=dict(l=150, r=80, t=20, b=60),
    )

    return fig


def create_risk_gauge(crs: float, risk_level: str) -> go.Figure:
    """Create a professional gauge chart for risk score."""
    config = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=crs * 100,
        number={"suffix": "%", "font": {"size": 48, "color": config["color"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2},
            "bar": {"color": config["color"], "thickness": 0.75},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E0E0E0",
            "steps": [
                {"range": [0, 35], "color": "#E8F5E9"},
                {"range": [35, 65], "color": "#FFF3E0"},
                {"range": [65, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "#212121", "width": 3},
                "thickness": 0.8,
                "value": crs * 100
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_component_radar(components: dict) -> go.Figure:
    """Create radar chart for risk components."""
    categories = list(components.keys())
    values = list(components.values())

    # Close the radar
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[c.replace("_", " ").title() for c in categories],
        fill="toself",
        fillcolor=f"rgba(27, 77, 62, 0.3)",
        line=dict(color=COLORS["primary"], width=2),
        name="Risk Components"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        height=350,
        margin=dict(l=80, r=80, t=40, b=40),
    )

    return fig


def create_neighbor_network_viz(
    df: pd.DataFrame,
    genome_id: str,
    pca_coords: np.ndarray,
    k: int = 10
) -> go.Figure:
    """Create k-NN neighborhood visualization."""
    plot_df = df.copy()
    plot_df["PC1"] = pca_coords[:, 0]
    plot_df["PC2"] = pca_coords[:, 1]

    query_idx = plot_df[plot_df["genome_id"] == genome_id].index[0]
    query_point = pca_coords[query_idx]

    # Find k nearest neighbors
    distances = np.linalg.norm(pca_coords - query_point, axis=1)
    neighbor_indices = np.argsort(distances)[1:k+1]

    fig = go.Figure()

    # Draw lines to neighbors
    for idx in neighbor_indices:
        neighbor_point = pca_coords[idx]
        fig.add_trace(go.Scatter(
            x=[query_point[0], neighbor_point[0]],
            y=[query_point[1], neighbor_point[1]],
            mode="lines",
            line=dict(color="#BDBDBD", width=1),
            hoverinfo="skip",
            showlegend=False
        ))

    # Plot neighbors
    neighbor_df = plot_df.iloc[neighbor_indices]
    for pathogen_label, group in neighbor_df.groupby("pathogenicity_label"):
        color = PATHOGEN_COLORS.get(pathogen_label, "#888888")
        fig.add_trace(go.Scatter(
            x=group["PC1"],
            y=group["PC2"],
            mode="markers",
            marker=dict(size=12, color=color, line=dict(width=1, color="white")),
            name=pathogen_label,
            text=group["genome_id"],
            hovertemplate="<b>%{text}</b><br>Species: %{customdata}<extra></extra>",
            customdata=group["species"]
        ))

    # Plot query genome
    query_row = plot_df.iloc[query_idx]
    fig.add_trace(go.Scatter(
        x=[query_point[0]],
        y=[query_point[1]],
        mode="markers",
        marker=dict(
            size=20,
            color="#FFD700",
            symbol="star",
            line=dict(width=2, color="#000000")
        ),
        name="Query Genome",
        hovertemplate=f"<b>QUERY: {genome_id}</b><extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=20, t=20, b=80),
    )

    return fig


# =============================================================================
# Page Components
# =============================================================================

def render_header():
    """Render the professional header."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* ========================================
       GLOBAL STYLES - Academic/Professional
       ======================================== */

    :root {
        --primary-dark: #1B4D3E;
        --primary-medium: #2E7D5A;
        --primary-light: #4CAF50;
        --accent-gold: #C9A227;
        --danger: #C62828;
        --warning: #E65100;
        --success: #2E7D32;
        --neutral-900: #1a1a2e;
        --neutral-800: #16213e;
        --neutral-700: #2d3748;
        --neutral-600: #4a5568;
        --neutral-500: #718096;
        --neutral-400: #a0aec0;
        --neutral-300: #cbd5e0;
        --neutral-200: #e2e8f0;
        --neutral-100: #f7fafc;
        --neutral-50: #ffffff;
    }

    /* Main container adjustments */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* ========================================
       HEADER - Professional Banner
       ======================================== */

    .main-header {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-medium) 50%, #3d8b6e 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(27, 77, 62, 0.3);
        border-bottom: 3px solid var(--accent-gold);
    }

    .main-title {
        font-family: 'Source Sans Pro', -apple-system, sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .main-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        opacity: 0.92;
        margin-top: 0.4rem;
        letter-spacing: 0.2px;
    }

    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(4px);
        padding: 0.3rem 0.85rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* ========================================
       STAT CARDS - Clean Metrics Display
       ======================================== */

    .stat-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8faf9 100%);
        border-radius: 8px;
        padding: 1.25rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #e8ece9;
        text-align: center;
        transition: all 0.2s ease;
        border-left: 3px solid var(--primary-dark);
    }

    .stat-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(27, 77, 62, 0.12);
        border-left-color: var(--primary-light);
    }

    .stat-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.65rem;
        font-weight: 500;
        color: var(--primary-dark);
        margin: 0;
        line-height: 1.2;
    }

    .stat-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--neutral-500);
        margin-top: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ========================================
       RISK BANNERS - Alert Displays
       ======================================== */

    .risk-banner {
        border-radius: 8px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .risk-banner-high {
        background: linear-gradient(135deg, #C62828 0%, #B71C1C 100%);
        color: white;
        border-left: 4px solid #ff5252;
    }

    .risk-banner-medium {
        background: linear-gradient(135deg, #E65100 0%, #BF360C 100%);
        color: white;
        border-left: 4px solid #FF9800;
    }

    .risk-banner-low {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        border-left: 4px solid #69F0AE;
    }

    .risk-level-text {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .risk-action-text {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.92;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }

    /* ========================================
       INFO BOXES - Callouts (FIXED TEXT COLOR)
       ======================================== */

    .info-box {
        background: linear-gradient(135deg, #f0f7f4 0%, #e8f5e9 100%);
        border-left: 4px solid var(--primary-dark);
        padding: 1rem 1.25rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
        color: #1a1a2e !important;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .info-box strong {
        color: var(--primary-dark) !important;
        font-weight: 600;
    }

    .info-box-warning {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left-color: var(--warning);
    }

    .info-box-warning strong {
        color: var(--warning) !important;
    }

    .info-box-danger {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: var(--danger);
    }

    .info-box-danger strong {
        color: var(--danger) !important;
    }

    /* ========================================
       SECTION HEADERS - Academic Style
       ======================================== */

    .section-header {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--primary-dark);
        margin: 1.75rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--neutral-200);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header::before {
        content: '';
        width: 4px;
        height: 1.2rem;
        background: var(--primary-dark);
        border-radius: 2px;
    }

    /* ========================================
       METRIC PILLS - Status Badges
       ======================================== */

    .metric-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .metric-pill-success {
        background: #e8f5e9;
        color: #1B5E20 !important;
        border: 1px solid #a5d6a7;
    }

    .metric-pill-warning {
        background: #fff3e0;
        color: #E65100 !important;
        border: 1px solid #ffcc80;
    }

    .metric-pill-danger {
        background: #ffebee;
        color: #C62828 !important;
        border: 1px solid #ef9a9a;
    }

    /* ========================================
       DATA TABLES - Clean Academic Style
       ======================================== */

    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        margin: 1rem 0;
    }

    .data-table th {
        background: var(--primary-dark);
        color: white;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    .data-table td {
        padding: 0.65rem 1rem;
        border-bottom: 1px solid var(--neutral-200);
        color: var(--neutral-700) !important;
    }

    .data-table tr:hover td {
        background: #f0f7f4;
    }

    /* ========================================
       MANUSCRIPT CARD - For Key Findings
       ======================================== */

    .manuscript-card {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .manuscript-card h4 {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--primary-dark);
        margin: 0 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--neutral-200);
    }

    .manuscript-card p {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        color: var(--neutral-600) !important;
        line-height: 1.7;
        margin: 0;
    }

    /* ========================================
       FOOTER - Professional Attribution
       ======================================== */

    .footer-text {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.75rem;
        color: var(--neutral-500) !important;
        text-align: center;
        padding: 1.5rem 0;
        border-top: 1px solid var(--neutral-200);
        margin-top: 2rem;
        line-height: 1.8;
    }

    .footer-text strong {
        color: var(--primary-dark) !important;
    }

    /* ========================================
       STREAMLIT OVERRIDES
       ======================================== */

    /* Hide default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faf9 0%, #f0f4f2 100%);
        border-right: 1px solid var(--neutral-200);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--neutral-700) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--neutral-100);
        padding: 4px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--neutral-600);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--neutral-200);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-dark) !important;
        color: white !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace !important;
        color: var(--primary-dark) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: var(--neutral-500) !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.5px;
    }

    /* Button styling */
    .stButton > button {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        border-radius: 6px;
        border: 1px solid var(--neutral-300);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--primary-dark);
        color: var(--primary-dark);
    }

    /* Selectbox styling */
    [data-testid="stSelectbox"] label {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600;
        color: var(--neutral-600) !important;
        font-size: 0.85rem;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--neutral-200);
        border-radius: 8px;
        overflow: hidden;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">
            <span style="font-size: 1.5rem;">🛡️</span>
            FoodGuard AI
        </h1>
        <p class="main-subtitle">Genomic Surveillance Dashboard — Whole-Proteome ESM-2 Embedding Analysis</p>
        <span class="header-badge">GenomeTrakr • 21,657 Genomes • 9 Taxa • Public Health Defense</span>
    </div>
    """, unsafe_allow_html=True)


def render_overview_page(df: pd.DataFrame, pca_coords: np.ndarray):
    """Render the dataset overview page."""
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{len(df):,}</p>
            <p class="stat-label">Total Genomes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{df['species'].nunique()}</p>
            <p class="stat-label">Taxa</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{MANUSCRIPT_STATS['mean_species_homophily']:.3f}</p>
            <p class="stat-label">Mean Homophily</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{MANUSCRIPT_STATS['n_clusters']}</p>
            <p class="stat-label">Clusters</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{MANUSCRIPT_STATS['strict_outlier_pct']}%</p>
            <p class="stat-label">Outliers</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout for charts
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("**Embedding Space (PCA Projection)**")
        color_option = st.radio(
            "Color by:",
            ["Species", "Pathogenicity"],
            horizontal=True,
            key="overview_pca_color"
        )
        fig = create_pca_figure(
            df, pca_coords,
            color_by="species" if color_option == "Species" else "pathogenicity"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Species Distribution**")
        fig = create_species_distribution_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    # Homophily distributions
    st.markdown('<h2 class="section-header">📈 Geometry Diagnostics</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Species Homophily Distribution**")
        fig = create_homophily_histogram(df, "species")
        st.plotly_chart(fig, use_container_width=True)

        boundary_count = (df["species_homophily"] < 0.9).sum()
        st.markdown(f"""
        <div class="info-box">
            <strong>Boundary Cases:</strong> {boundary_count:,} genomes ({boundary_count/len(df)*100:.1f}%)
            have homophily < 0.9, indicating mixed neighborhoods requiring additional review.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Pathogenicity Homophily Distribution**")
        fig = create_homophily_histogram(df, "pathogenicity")
        st.plotly_chart(fig, use_container_width=True)

        high_conf = (df["pathogenicity_homophily"] >= 0.95).sum()
        st.markdown(f"""
        <div class="info-box">
            <strong>High Confidence:</strong> {high_conf:,} genomes ({high_conf/len(df)*100:.1f}%)
            have pathogenicity homophily ≥ 0.95, supporting confident triage decisions.
        </div>
        """, unsafe_allow_html=True)


def render_risk_assessment_page(df: pd.DataFrame, pca_coords: np.ndarray):
    """Render the interactive risk assessment page."""
    st.markdown('<h2 class="section-header">🔬 Genome Risk Assessment</h2>', unsafe_allow_html=True)

    # Sidebar-style controls in columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        posture = st.selectbox(
            "Operating Posture",
            ["recall_high", "balanced", "precision_high"],
            format_func=lambda x: {
                "recall_high": "🚨 Recall-High (Outbreak Investigation)",
                "balanced": "⚖️ Balanced (Routine Surveillance)",
                "precision_high": "🎯 Precision-High (Regulatory Action)"
            }.get(x, x),
            key="risk_posture"
        )

    with col2:
        species_filter = st.multiselect(
            "Filter by Species",
            options=sorted(df["species"].unique()),
            default=["E_coli_O157H7", "E_coli_nonpathogenic"] if "E_coli_O157H7" in df["species"].values else [],
            key="risk_species_filter"
        )

    filtered_df = df[df["species"].isin(species_filter)] if species_filter else df

    with col3:
        genome_id = st.selectbox(
            "Select Genome",
            options=filtered_df["genome_id"].tolist(),
            key="risk_genome_select"
        )

    # Quick action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    with qcol1:
        if st.button("🎲 Random Genome", use_container_width=True):
            genome_id = filtered_df.sample(1)["genome_id"].iloc[0]
            st.rerun()

    with qcol2:
        if st.button("⚠️ Random Boundary Case", use_container_width=True):
            boundary = filtered_df[filtered_df["species_homophily"] < 0.9]
            if not boundary.empty:
                genome_id = boundary.sample(1)["genome_id"].iloc[0]
                st.rerun()

    with qcol3:
        if st.button("🔴 Random Outlier", use_container_width=True):
            outliers = filtered_df[filtered_df["is_outlier_strict"] == True]
            if not outliers.empty:
                genome_id = outliers.sample(1)["genome_id"].iloc[0]
                st.rerun()

    with qcol4:
        if st.button("🦠 Random Pathogen", use_container_width=True):
            pathogens = filtered_df[filtered_df["pathogenicity_label"] == "pathogenic"]
            if not pathogens.empty:
                genome_id = pathogens.sample(1)["genome_id"].iloc[0]
                st.rerun()

    st.markdown("---")

    # Get genome data and compute assessment
    genome_row = df[df["genome_id"] == genome_id].iloc[0]
    assessment = compute_risk_assessment(genome_row, posture)

    # Risk banner
    risk_class = f"risk-banner-{assessment.risk_level}"
    config = RISK_CONFIG.get(assessment.risk_level, RISK_CONFIG["medium"])

    st.markdown(f"""
    <div class="risk-banner {risk_class}">
        <div class="risk-level-text">{config['icon']} {config['label']}</div>
        <div class="risk-action-text">{config['action']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main assessment display
    col1, col2 = st.columns([1, 1])

    with col1:
        # Genome info card
        st.markdown("**Genome Information**")

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Genome ID", genome_id)
            st.metric("Species", genome_row["species"].replace("_", " "))
        with info_col2:
            st.metric("Proteins", f"{int(genome_row['proteins']):,}")
            label_color = "🔴" if genome_row["pathogenicity_label"] == "pathogenic" else "🟢"
            st.metric("Label", f"{label_color} {genome_row['pathogenicity_label'].title()}")

        # Risk gauge
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Combined Risk Score (CRS)**")
        fig = create_risk_gauge(assessment.crs, assessment.risk_level)
        st.plotly_chart(fig, use_container_width=True)

        # Confidence and decision
        conf_class = {
            "high": "metric-pill-success",
            "medium": "metric-pill-warning",
            "low": "metric-pill-danger"
        }.get(assessment.confidence, "metric-pill-warning")

        st.markdown(f"""
        <div style="text-align: center; margin-top: -1rem;">
            <span class="metric-pill {conf_class}">Confidence: {assessment.confidence.upper()}</span>
            <span class="metric-pill metric-pill-warning">Decision: {assessment.decision.replace('_', ' ').upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Risk Component Analysis**")
        fig = create_component_radar(assessment.components)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>Assessment Rationale:</strong> {assessment.reason}
        </div>
        """, unsafe_allow_html=True)

        # Component breakdown table
        st.markdown("**Component Scores**")
        comp_df = pd.DataFrame([
            {"Component": k.replace("_", " ").title(), "Score": f"{v:.3f}"}
            for k, v in assessment.components.items()
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Neighborhood visualization
    st.markdown('<h2 class="section-header">🔗 Neighborhood Context</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("**k-Nearest Neighbors in Embedding Space**")
        fig = create_neighbor_network_viz(df, genome_id, pca_coords, k=15)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Neighbor Statistics**")

        # Get actual neighbors
        query_idx = df[df["genome_id"] == genome_id].index[0]
        query_point = pca_coords[query_idx]
        distances = np.linalg.norm(pca_coords - query_point, axis=1)
        neighbor_indices = np.argsort(distances)[1:21]
        neighbors = df.iloc[neighbor_indices]

        # Stats
        same_species = (neighbors["species"] == genome_row["species"]).mean()
        same_pathogen = (neighbors["pathogenicity_label"] == genome_row["pathogenicity_label"]).mean()

        st.metric("Species Agreement (k=20)", f"{same_species:.1%}")
        st.metric("Pathogenicity Agreement (k=20)", f"{same_pathogen:.1%}")

        # Neighbor table
        st.markdown("**Top 10 Neighbors**")
        neighbor_display = neighbors.head(10)[["genome_id", "species", "pathogenicity_label"]].copy()
        neighbor_display.columns = ["Genome", "Species", "Label"]
        neighbor_display["Species"] = neighbor_display["Species"].str.replace("_", " ")
        st.dataframe(neighbor_display, use_container_width=True, hide_index=True)


def render_manuscript_page(df: pd.DataFrame, pca_coords: np.ndarray):
    """Render the manuscript key findings page."""
    st.markdown('<h2 class="section-header">📄 Manuscript Key Findings</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Paper:</strong> "Whole-Proteome ESM-2 Embeddings Recover Taxonomy and Enable
        Geometry-Aware Triage of Foodborne Bacterial Genomes"<br>
        <strong>Authors:</strong> Jay Gutierrez & Javier Correa Alvarez
    </div>
    """, unsafe_allow_html=True)

    # Key findings tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧬 Embedding Geometry",
        "📊 Clustering Analysis",
        "🎯 Within-Genus Tests",
        "🛡️ Robustness"
    ])

    with tab1:
        st.markdown("### Low-Dimensional Structure")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            The embedding space shows strong low-dimensional structure dominated by taxonomic signals:

            - **PC1** explains **76.56%** of variance
            - **PC2** explains **10.95%** of variance
            - Only **3 PCs** needed for 90% cumulative variance

            This indicates that whole-proteome ESM-2 embeddings capture broad taxonomic
            and proteome-level signals, enabling rapid retrieval-based triage.
            """)

            # Variance explained chart
            pcs = ["PC1", "PC2", "PC3", "PC4", "PC5"]
            variances = [76.56, 10.95, 6.82, 2.14, 1.08]
            cumulative = [76.56, 87.51, 94.33, 96.47, 97.55]

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(x=pcs, y=variances, name="Individual", marker_color=COLORS["primary"]),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(x=pcs, y=cumulative, name="Cumulative",
                          line=dict(color=COLORS["danger"], width=3),
                          mode="lines+markers"),
                secondary_y=True
            )

            fig.update_layout(
                template="plotly_white",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=60, t=40, b=60),
            )
            fig.update_yaxes(title_text="Variance Explained (%)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Interactive PCA Visualization**")
            fig = create_pca_figure(df, pca_coords, color_by="species")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### HDBSCAN Clustering Results")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            Density-based clustering reveals species-coherent structure:

            | Metric | Value |
            |--------|-------|
            | **Clusters** | {MANUSCRIPT_STATS['n_clusters']} |
            | **Noise Points** | {MANUSCRIPT_STATS['cluster_noise_pct']}% |
            | **Silhouette Score** | {MANUSCRIPT_STATS['silhouette']:.3f} |
            | **Species Purity** | {MANUSCRIPT_STATS['cluster_purity']:.0%} |
            | **Bootstrap Stability** | 0.81 (mean Jaccard) |

            **Key Finding:** 33/34 clusters (97%) achieve ≥90% pathogenicity purity,
            supporting cluster-based confidence layers for operational triage.
            """)

        with col2:
            # Cluster distribution
            if "cluster" in df.columns:
                cluster_counts = df[df["cluster"] >= 0]["cluster"].value_counts().head(20)
                fig = go.Figure(go.Bar(
                    x=[f"C{c}" for c in cluster_counts.index],
                    y=cluster_counts.values,
                    marker_color=COLORS["secondary"]
                ))
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Cluster ID",
                    yaxis_title="Genome Count",
                    height=300,
                    margin=dict(l=60, r=20, t=20, b=60),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Within-Genus Discrimination")

        st.markdown("""
        Critical stress tests validate discrimination within closely related taxa:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### *E. coli* O157:H7 vs Non-pathogenic

            | Metric | Value |
            |--------|-------|
            | **Sample Size** | 5,812 genomes |
            | **kNN Accuracy (k=20)** | **98.42%** |
            | **Balanced Accuracy** | 98.30% |
            | **Mean Homophily** | 0.9752 |
            | **Silhouette** | 0.392 |

            Even among closely related *E. coli* strains, the embedding geometry
            preserves pathotype annotations with high fidelity.
            """)

            # E. coli accuracy visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=98.42,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["primary"]},
                    "steps": [
                        {"range": [0, 80], "color": "#FFEBEE"},
                        {"range": [80, 95], "color": "#FFF3E0"},
                        {"range": [95, 100], "color": "#E8F5E9"},
                    ],
                },
                title={"text": "E. coli Accuracy"}
            ))
            fig.update_layout(height=250, margin=dict(l=30, r=30, t=80, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            #### *Listeria monocytogenes* vs *L. innocua*

            | Metric | Value |
            |--------|-------|
            | **Sample Size** | 4,949 genomes |
            | **kNN Accuracy (k=20)** | **99.92%** |
            | **Balanced Accuracy** | 99.55% |
            | **Mean Homophily** | 0.9986 |
            | **Silhouette** | 0.180 |

            The *Listeria* pair shows near-perfect separation despite
            being closely related species.
            """)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=99.92,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["primary"]},
                    "steps": [
                        {"range": [0, 80], "color": "#FFEBEE"},
                        {"range": [80, 95], "color": "#FFF3E0"},
                        {"range": [95, 100], "color": "#E8F5E9"},
                    ],
                },
                title={"text": "Listeria Accuracy"}
            ))
            fig.update_layout(height=250, margin=dict(l=30, r=30, t=80, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Robustness Validation")

        st.markdown("""
        Stress tests under proteome incompleteness and contamination demonstrate operational resilience:
        """)

        # Robustness results table
        robustness_data = {
            "Condition": [
                "Protein Dropout 90%", "Protein Dropout 75%", "Protein Dropout 50%",
                "Contig Dropout 90%", "Contig Dropout 75%", "Contig Dropout 50%",
                "Contamination 5%", "Contamination 10%", "Contamination 20%"
            ],
            "Cosine Similarity": [
                "0.9999", "0.9999", "0.9999",
                "0.9999", "0.9999", "0.9999",
                "0.9999", "0.9998", "0.9995"
            ],
            "kNN Jaccard": [
                "0.739", "0.667", "0.481",
                "0.818", "0.481", "0.143",
                "0.667", "0.538", "0.290"
            ],
            "Outlier Rate": [
                "3.1%", "4.6%", "26.4%",
                "7.8%", "21.1%", "53.8%",
                "3.1%", "15.7%", "56.4%"
            ]
        }

        st.dataframe(pd.DataFrame(robustness_data), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
            <strong>Key Insight:</strong> Global embeddings remain highly stable (cosine similarity ~1.0)
            even under severe perturbation, while local neighborhoods degrade predictably. This enables
            calibrated "trust vs defer" rules tied to assembly quality and contamination risk.
        </div>
        """, unsafe_allow_html=True)


def render_about_page():
    """Render the about/methodology page."""
    st.markdown('<h2 class="section-header">ℹ️ About FoodGuard AI</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### Mission

    FoodGuard AI is a **defensive tool for public health** designed to protect food safety by
    identifying potential contamination in the food supply chain. Think of it as an
    **antivirus for food** — we study the digital signatures of harmful bacteria to create
    a system that detects and flags them, preventing harm.

    ---

    ### Methodology

    #### Whole-Proteome ESM-2 Embeddings

    Each bacterial genome is represented by:
    1. **Protein Extraction**: All proteins from GenBank annotations
    2. **ESM-2 Embedding**: Each protein → 480-dimensional vector via ESM-2 (`esm2_t12_35M_UR50D`)
    3. **Mean Pooling**: Genome embedding = mean of all protein embeddings

    This "bag-of-proteins" representation captures broad taxonomic and functional signals
    while enabling fast, cache-friendly computation.

    #### Geometry-Aware Risk Scoring

    The risk scoring framework uses embedding geometry as a confidence layer:

    | Signal | Description | Use |
    |--------|-------------|-----|
    | **Homophily** | k-NN label agreement | Confidence indicator |
    | **Cluster Purity** | Dominant label fraction in cluster | Triage reliability |
    | **Outlier Risk** | Distance-based anomaly score | Novelty/QC flag |

    When geometry confidence is low, the system **defers to expert review** rather than
    returning brittle high-confidence labels.

    ---

    ### Dataset

    **21,657 bacterial genomes** from FDA's GenomeTrakr surveillance network spanning 9 taxa:

    | Taxon | Count | Label |
    |-------|-------|-------|
    | *Salmonella enterica* | 7,000 | Pathogenic |
    | *Listeria monocytogenes* | 4,500 | Pathogenic |
    | *E. coli* (non-pathogenic) | 4,312 | Non-pathogenic |
    | *Bacillus subtilis* | 2,361 | Non-pathogenic |
    | *E. coli* O157:H7 | 1,500 | Pathogenic |
    | *Citrobacter koseri* | 897 | Non-pathogenic |
    | *Listeria innocua* | 449 | Non-pathogenic |
    | *Escherichia fergusonii* | 438 | Non-pathogenic |
    | *Citrobacter freundii* | 200 | Non-pathogenic |

    ---

    ### Technology Stack

    - **Foundation Model**: ESM-2 protein language model (Meta AI)
    - **Genomic Context**: Bacformer contextual genome language model
    - **Visualization**: Plotly, Streamlit
    - **Compute**: Cache-first execution on EAFIT Apolo-3 HPC (NVIDIA H100)

    ---

    ### Citation

    ```
    Gutierrez J, Correa Alvarez J. Whole-Proteome ESM-2 Embeddings Recover Taxonomy
    and Enable Geometry-Aware Triage of Foodborne Bacterial Genomes. 2025.
    ```

    ---

    ### Contact

    - **Jay Gutierrez** — jg@graphoflife.com
    - **Javier Correa Alvarez, PhD** — jcorre38@eafit.edu.co
    """)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="FoodGuard AI — Genomic Surveillance",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Render header
    render_header()

    # Load data
    df, pca_coords, umap_coords = load_genome_data()

    if pca_coords is None:
        st.error("PCA coordinates not available. Using synthetic demo data.")
        df, pca_coords, umap_coords = create_demo_data()

    # Navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h3 style="color: #1B4D3E; margin: 0;">Navigation</h3>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dataset Overview", "🔬 Risk Assessment", "📄 Manuscript Findings", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Dataset stats in sidebar
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Total Genomes", f"{len(df):,}")
    st.sidebar.metric("Mean Homophily", f"{df['species_homophily'].mean():.4f}")

    outlier_count = df["is_outlier_strict"].sum() if "is_outlier_strict" in df.columns else 0
    st.sidebar.metric("Outliers", f"{outlier_count:,}")

    pathogen_count = (df["pathogenicity_label"] == "pathogenic").sum()
    st.sidebar.metric("Pathogenic", f"{pathogen_count:,} ({pathogen_count/len(df)*100:.1f}%)")

    # Render selected page
    if page == "📊 Dataset Overview":
        render_overview_page(df, pca_coords)
    elif page == "🔬 Risk Assessment":
        render_risk_assessment_page(df, pca_coords)
    elif page == "📄 Manuscript Findings":
        render_manuscript_page(df, pca_coords)
    elif page == "ℹ️ About":
        render_about_page()

    # Footer
    st.markdown("""
    <div class="footer-text">
        <strong>FoodGuard AI</strong> v1.0 — A defensive tool for public health<br>
        Powered by ESM-2 & Bacformer | Data: GenomeTrakr (21,657 genomes)<br>
        © 2025 Jay Gutierrez & Javier Correa Alvarez
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
