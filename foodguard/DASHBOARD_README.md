# FoodGuard AI Dashboard

A professional, interactive dashboard for visualizing whole-proteome ESM-2 embeddings and enabling geometry-aware triage of foodborne bacterial genomes.

![FoodGuard Dashboard](https://img.shields.io/badge/FoodGuard-AI%20Dashboard-1B4D3E?style=for-the-badge)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit)

## Overview

This dashboard serves as an **interactive companion to the manuscript**:

> *"Whole-Proteome ESM-2 Embeddings Recover Taxonomy and Enable Geometry-Aware Triage of Foodborne Bacterial Genomes"*
>
> Jay Gutierrez & Javier Correa Alvarez

### Features

- **📊 Dataset Overview**: Explore the 21,657 genome embedding space with interactive PCA/UMAP visualizations
- **🔬 Risk Assessment**: Geometry-aware risk scoring with confidence signals for individual genomes
- **📄 Manuscript Findings**: Interactive visualization of key results from the paper
- **🛡️ Professional UI**: Public health-themed design with intuitive navigation

---

## Quick Start

### 1. Install Dependencies

```bash
# From the repository root
pip install streamlit plotly pandas numpy scipy pyarrow

# Or with the full development environment
pip install -e ".[dev]"
```

### 2. Generate Demo Data (Optional)

If you don't have the actual GenomeTrakr embeddings, generate demonstration data:

```bash
python foodguard/generate_demo_data.py
```

This creates realistic synthetic data matching the manuscript's dataset structure.

### 3. Launch the Dashboard

```bash
streamlit run foodguard/dashboard.py
```

The dashboard will open at `http://localhost:8501`

---

## Dashboard Pages

### 📊 Dataset Overview

- **Key Metrics**: Total genomes, taxa, mean homophily, clusters, outliers
- **Embedding Space**: Interactive PCA scatter plot colored by species or pathogenicity
- **Species Distribution**: Bar chart of genome counts per taxon
- **Geometry Diagnostics**: Homophily distribution histograms with boundary case identification

### 🔬 Risk Assessment

- **Operating Posture**: Select recall-high (outbreak), balanced, or precision-high (regulatory)
- **Genome Selection**: Filter by species, select specific genomes, or use quick filters
- **Risk Banner**: Traffic-light style risk indicator (HIGH/MEDIUM/LOW)
- **CRS Gauge**: Combined Risk Score visualization
- **Component Radar**: Breakdown of geometry confidence, homophily, pathogenicity score
- **Neighborhood Context**: k-NN visualization showing neighbor labels and agreement

### 📄 Manuscript Findings

Interactive exploration of key results:

1. **Embedding Geometry**: PCA variance explained, low-dimensional structure
2. **Clustering Analysis**: HDBSCAN results, cluster purity, bootstrap stability
3. **Within-Genus Tests**: E. coli O157:H7 vs non-pathogenic (98.4% accuracy), Listeria stress test (99.9%)
4. **Robustness**: Protein dropout, contig dropout, contamination mixing validation

### ℹ️ About

- Methodology explanation
- Dataset description
- Technology stack
- Citation information

---

## Data Requirements

The dashboard expects data files in `foodguard/analysis/`:

| File | Description |
|------|-------------|
| `genome_embeddings_enriched.parquet` | Genome metadata with homophily scores |
| `dim_reduction_cache.npz` | PCA and UMAP coordinates |

### Expected Columns in Parquet

```python
{
    "genome_id": str,            # Unique identifier
    "species": str,              # Species name (e.g., "Salmonella_enterica")
    "pathogenicity_label": str,  # "pathogenic" or "non-pathogenic"
    "proteins": int,             # Number of proteins
    "species_homophily": float,  # Species-level k-NN agreement (0-1)
    "pathogenicity_homophily": float,  # Pathogenicity k-NN agreement (0-1)
    "cluster": int,              # HDBSCAN cluster ID (-1 for noise)
    "is_outlier": bool,          # Distance outlier flag
    "is_outlier_strict": bool,   # Strict outlier (distance + noise)
}
```

### NPZ Cache Structure

```python
{
    "pca_result": np.ndarray,   # Shape: (n_genomes, 2) - PC1, PC2
    "umap_result": np.ndarray,  # Shape: (n_genomes, 2) - UMAP coords
}
```

---

## Risk Scoring Logic

The dashboard implements **geometry-gated risk scoring** from `risk_v2.py`:

### Stage 1: Geometry Gate

```
geometry_confidence = min(homophily, pathogen_homophily) × (1 - outlier_penalty)
```

If `geometry_confidence < threshold`:
- **recall_high**: Return CRS=0.80, decision="defer_to_expert"
- **balanced/precision**: Return CRS=0.50, decision="secondary_analysis"

### Stage 2: Weighted Fusion

```
CRS = 0.6 × PS + 0.25 × geometry_confidence + 0.15 × evidence_score
```

With outlier boost: `CRS = min(1.0, CRS + 0.15)` if outlier

### Stage 3: Decision Logic

| CRS Range | Risk Level | Decision |
|-----------|------------|----------|
| ≥ 0.65 (recall_high) | HIGH | Escalate/Review |
| ≥ 0.35 | MEDIUM | Review |
| < 0.35 | LOW | Proceed |

---

## Development

### Running with Hot Reload

```bash
streamlit run foodguard/dashboard.py --server.runOnSave true
```

### Customization

Key configuration in `dashboard.py`:

```python
# Color palette
COLORS = {
    "primary": "#1B4D3E",    # Forest green
    "secondary": "#2E7D5A",  # Medium green
    ...
}

# Species colors for consistent visualization
SPECIES_COLORS = {
    "Salmonella_enterica": "#E53935",
    ...
}

# Risk configuration
RISK_CONFIG = {
    "high": {"color": "#D32F2F", "icon": "🚨", ...},
    ...
}
```

---

## Troubleshooting

### Dashboard shows demo data

The actual data files are not present. Either:
1. Run `python foodguard/generate_demo_data.py` to create synthetic data
2. Or ensure `genome_embeddings_enriched.parquet` and `dim_reduction_cache.npz` exist in `foodguard/analysis/`

### Streamlit not found

```bash
pip install streamlit>=1.28.0
```

### Slow loading

The dashboard uses `@st.cache_data` for performance. First load may be slower while caching.

### Port already in use

```bash
streamlit run foodguard/dashboard.py --server.port 8502
```

---

## Citation

```bibtex
@article{gutierrez2025foodguard,
  title={Whole-Proteome ESM-2 Embeddings Recover Taxonomy and Enable
         Geometry-Aware Triage of Foodborne Bacterial Genomes},
  author={Gutierrez, Jay and Correa Alvarez, Javier},
  year={2025},
  note={FoodGuard AI Dashboard}
}
```

---

## License

This dashboard is part of the FoodGuard AI project — a defensive tool for public health.

**Contact:**
- Jay Gutierrez — jg@graphoflife.com
- Javier Correa Alvarez, PhD — jcorre38@eafit.edu.co
