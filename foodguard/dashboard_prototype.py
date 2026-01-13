"""
FoodGuard Surveillance Dashboard (Streamlit Prototype)

A minimal viable dashboard for demonstrating the risk scoring framework
to food safety domain experts. Implements the Level 1 "Traffic Light Summary"
from the strategic recommendations.

Usage:
    streamlit run foodguard/dashboard_prototype.py

Requirements:
    pip install streamlit plotly pandas numpy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Conditional imports for production environment
try:
    from foodguard.risk_v2 import (
        CombinedRiskAssessment,
        GeometryScores,
        ModelScores,
        compute_crs_gated,
    )

    RISK_V2_AVAILABLE = True
except ImportError:
    RISK_V2_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "foodguard" / "analysis"
ENRICHED_PARQUET = ANALYSIS_DIR / "genome_embeddings_enriched.parquet"
DIM_CACHE = ANALYSIS_DIR / "dim_reduction_cache.npz"

POSTURE_OPTIONS = {
    "Recall-High (Outbreak Investigation)": "recall_high",
    "Balanced (Routine Surveillance)": "balanced",
    "Precision-High (Regulatory Action)": "precision_high",
}

RISK_COLORS = {
    "low": "#28a745",  # Green
    "medium": "#ffc107",  # Yellow
    "high": "#dc3545",  # Red
}

DECISION_ICONS = {
    "proceed": "\u2705",  # Check mark
    "review": "\u26a0\ufe0f",  # Warning
    "defer_to_expert": "\ud83d\uded1",  # Stop sign
    "escalate": "\ud83d\udea8",  # Police siren
    "require_secondary_analysis": "\ud83d\udd0d",  # Magnifying glass
}


# ============================================================================
# Data Loading
# ============================================================================


@st.cache_data
def load_data():
    """Load enriched genome embeddings and metadata."""
    if not ENRICHED_PARQUET.exists():
        st.error(f"Data file not found: {ENRICHED_PARQUET}")
        st.stop()

    df = pd.read_parquet(ENRICHED_PARQUET)

    if DIM_CACHE.exists():
        dim_cache = np.load(DIM_CACHE)
        pca_coords = dim_cache["pca_result"]
    else:
        pca_coords = None

    return df, pca_coords


# ============================================================================
# Risk Assessment (Mock or Real)
# ============================================================================


def assess_genome_risk(
    genome_row: pd.Series,
    posture: str = "recall_high",
) -> dict:
    """
    Compute risk assessment for a single genome.

    In production, this would call the actual pipeline. For this prototype,
    we use pre-computed columns from the enriched parquet.
    """
    if RISK_V2_AVAILABLE:
        # Use real risk_v2 implementation
        geom = GeometryScores(
            homophily_confidence=float(genome_row["species_homophily"]),
            cluster_purity=1.0,  # Placeholder (would compute from cluster stats)
            outlier_risk=1.0 if genome_row["is_outlier_strict"] else 0.2,
            centroid_proximity=0.8,  # Placeholder
            k_neighbors=20,
            cluster_id=int(genome_row["cluster"]),
        )
        model = ModelScores(
            pathogenicity_score=0.85 if genome_row["pathogenicity_label"] == "pathogenic" else 0.15,
            evidence_score=0.5,  # Placeholder
            novelty_score=0.3 if genome_row["is_outlier"] else 0.1,
            calibration_method="isotonic",
        )
        assessment = compute_crs_gated(geom, model, posture=posture)
        return {
            "crs": assessment.crs,
            "risk_level": assessment.risk_level,
            "confidence": assessment.confidence,
            "decision": assessment.decision,
            "reason": assessment.reason,
            "components": assessment.components,
        }
    else:
        # Mock assessment using pre-computed signals
        homophily = genome_row["species_homophily"]
        is_pathogenic = genome_row["pathogenicity_label"] == "pathogenic"
        is_outlier = genome_row["is_outlier_strict"]

        if homophily > 0.95 and is_pathogenic and not is_outlier:
            return {
                "crs": 0.88,
                "risk_level": "high",
                "confidence": "high",
                "decision": "review",
                "reason": "high_confidence_pathogenic_match",
                "components": {"ps": 0.90, "hcs": homophily},
            }
        elif homophily > 0.95 and not is_pathogenic:
            return {
                "crs": 0.12,
                "risk_level": "low",
                "confidence": "high",
                "decision": "proceed",
                "reason": "high_confidence_non_pathogenic",
                "components": {"ps": 0.10, "hcs": homophily},
            }
        elif homophily < 0.85:
            return {
                "crs": 0.75,
                "risk_level": "medium_uncertain",
                "confidence": "low",
                "decision": "defer_to_expert",
                "reason": "boundary_case_low_homophily",
                "components": {"ps": 0.60, "hcs": homophily},
            }
        else:
            return {
                "crs": 0.55,
                "risk_level": "medium",
                "confidence": "medium",
                "decision": "review",
                "reason": "moderate_risk_requires_confirmation",
                "components": {"ps": 0.65, "hcs": homophily},
            }


# ============================================================================
# Visualization Components
# ============================================================================


def render_traffic_light(risk_level: str, crs: float, confidence: str):
    """Render traffic light risk indicator."""
    color = RISK_COLORS.get(risk_level.replace("_uncertain", ""), "#6c757d")

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
        ">
            RISK: {risk_level.upper().replace('_', ' ')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Combined Risk Score (CRS)", f"{crs:.2f}")
    with col2:
        st.metric("Confidence", confidence.upper())


def render_knn_neighbors(df: pd.DataFrame, genome_id: str, k: int = 20):
    """Render table of k-nearest neighbors."""
    st.subheader("Nearest Neighbors (k=20)")

    # In production, compute actual k-NN. For prototype, sample similar species.
    query_row = df[df["genome_id"] == genome_id].iloc[0]
    same_species = df[df["species"] == query_row["species"]]

    if len(same_species) > k:
        neighbors = same_species.sample(min(k, len(same_species)))
    else:
        neighbors = same_species

    neighbor_table = neighbors[["genome_id", "species", "pathogenicity_label", "species_homophily"]].copy()
    neighbor_table = neighbor_table.rename(
        columns={
            "genome_id": "Genome ID",
            "species": "Species",
            "pathogenicity_label": "Label",
            "species_homophily": "Homophily",
        }
    )

    st.dataframe(neighbor_table, use_container_width=True)

    # Label mismatch warning
    query_label = query_row["pathogenicity_label"]
    mismatch_count = (neighbor_table["Label"] != query_label).sum()
    if mismatch_count > 0:
        st.warning(f"Warning: {mismatch_count}/{k} neighbors have different labels (explains medium confidence)")


def render_embedding_plot(df: pd.DataFrame, pca_coords: np.ndarray | None, genome_id: str):
    """Render 2D PCA embedding plot with query genome highlighted."""
    st.subheader("Embedding Space Context")

    if pca_coords is None:
        st.warning("PCA coordinates not available")
        return

    plot_df = df.copy()
    plot_df["PC1"] = pca_coords[:, 0]
    plot_df["PC2"] = pca_coords[:, 1]
    plot_df["is_query"] = plot_df["genome_id"] == genome_id

    # Sample for visualization (Plotly struggles with 21k points)
    query_row = plot_df[plot_df["is_query"]]
    other_sample = plot_df[~plot_df["is_query"]].sample(min(2000, len(plot_df) - 1))
    plot_sample = pd.concat([query_row, other_sample])

    fig = px.scatter(
        plot_sample,
        x="PC1",
        y="PC2",
        color="species",
        symbol="is_query",
        symbol_map={True: "star", False: "circle"},
        size="is_query",
        size_max=15,
        hover_data=["genome_id", "pathogenicity_label", "species_homophily"],
        title="PCA Projection (Query Genome Highlighted)",
        labels={"PC1": "PC1 (76.6% var)", "PC2": "PC2 (11.0% var)"},
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_risk_breakdown(components: dict):
    """Render bar chart of risk score components."""
    st.subheader("Risk Score Breakdown")

    component_names = {
        "ps": "Pathogenicity Score",
        "es": "Evidence Score",
        "ns": "Novelty Score",
        "ors": "Outlier Risk",
        "hcs": "Homophily Confidence",
        "cps": "Cluster Purity",
    }

    plot_data = []
    for key, value in components.items():
        if key in component_names:
            plot_data.append({"Component": component_names[key], "Score": value})

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        fig = px.bar(
            plot_df,
            x="Component",
            y="Score",
            title="Component Scores (0-1 scale)",
            labels={"Score": "Score", "Component": ""},
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Dashboard
# ============================================================================


def main():
    st.set_page_config(
        page_title="FoodGuard Surveillance Dashboard",
        page_icon="\ud83d\udd2c",
        layout="wide",
    )

    st.title("\ud83d\udd2c FoodGuard Surveillance Dashboard")
    st.markdown("**Whole-Proteome ESM-2 Embedding Risk Assessment System**")

    # Load data
    df, pca_coords = load_data()

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        # Posture selection
        posture_label = st.selectbox(
            "Operating Posture",
            options=list(POSTURE_OPTIONS.keys()),
            index=0,
        )
        posture = POSTURE_OPTIONS[posture_label]

        st.markdown("---")

        # Genome selection
        st.header("Genome Selection")
        species_filter = st.multiselect(
            "Filter by Species",
            options=sorted(df["species"].unique()),
            default=["E_coli_O157H7", "E_coli_nonpathogenic"],
        )

        if species_filter:
            filtered_df = df[df["species"].isin(species_filter)]
        else:
            filtered_df = df

        genome_id = st.selectbox(
            "Select Genome",
            options=filtered_df["genome_id"].tolist(),
            index=0,
        )

        # Quick filters
        st.markdown("---")
        st.header("Quick Filters")
        if st.button("Show Random Boundary Case"):
            boundary_cases = df[df["species_homophily"] < 0.90]
            if not boundary_cases.empty:
                genome_id = boundary_cases.sample(1)["genome_id"].iloc[0]
                st.rerun()

        if st.button("Show Random Outlier"):
            outliers = df[df["is_outlier_strict"] == True]  # noqa: E712
            if not outliers.empty:
                genome_id = outliers.sample(1)["genome_id"].iloc[0]
                st.rerun()

    # Main content
    genome_row = df[df["genome_id"] == genome_id].iloc[0]

    # Header info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Genome ID", genome_id)
    with col2:
        st.metric("Species", genome_row["species"].replace("_", " "))
    with col3:
        st.metric("Proteins", int(genome_row["proteins"]))

    st.markdown("---")

    # Level 1: Traffic Light Summary
    assessment = assess_genome_risk(genome_row, posture=posture)

    render_traffic_light(
        risk_level=assessment["risk_level"],
        crs=assessment["crs"],
        confidence=assessment["confidence"],
    )

    # Decision recommendation
    decision_icon = DECISION_ICONS.get(assessment["decision"], "\u2753")
    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #007bff;
            margin: 20px 0;
        ">
            <h3>{decision_icon} Recommended Action: {assessment['decision'].replace('_', ' ').title()}</h3>
            <p><strong>Reason:</strong> {assessment['reason'].replace('_', ' ')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable details
    with st.expander("View Detailed Diagnostics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            render_risk_breakdown(assessment["components"])

        with col2:
            st.subheader("Geometry Diagnostics")
            st.metric("Species Homophily", f"{genome_row['species_homophily']:.3f}")
            st.metric("Pathogenicity Homophily", f"{genome_row['pathogenicity_homophily']:.3f}")
            st.metric("Cluster ID", int(genome_row["cluster"]))
            st.metric("Outlier Status", "Yes" if genome_row["is_outlier_strict"] else "No")

        st.markdown("---")
        render_knn_neighbors(df, genome_id)

        st.markdown("---")
        render_embedding_plot(df, pca_coords, genome_id)

    # Footer
    st.markdown("---")
    st.caption(
        f"FoodGuard Dashboard v0.1 | Operating Posture: {posture_label} | "
        f"Dataset: {len(df):,} genomes from 9 taxa"
    )


if __name__ == "__main__":
    main()
