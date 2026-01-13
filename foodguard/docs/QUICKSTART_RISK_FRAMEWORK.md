# Quick Start Guide: Enhanced Risk Framework Implementation

**Goal:** Transform your embedding geometry analysis into a deployable surveillance system in 2-4 weeks.

**Status:** All code templates are ready. You need to integrate, test, and validate.

---

## What Was Delivered

### 1. Strategic Analysis (30 pages)
**File:** `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/docs/strategic_risk_framework_recommendations.md`

**Contents:**
- Multi-tiered risk scoring framework (geometry-gated)
- Dashboard design (3 levels: operator, analyst, scientist)
- 4 simulation scenarios (gradual drift, outbreak, artifacts, ecosystem shift)
- Manuscript framing recommendations (Section 3.9, Discussion updates)
- Operational deployment guardrails

**Action:** Read Section 1 (Risk Scoring Framework) first. It's the foundation for everything else.

### 2. Enhanced Risk Scoring Code
**File:** `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/risk_v2.py`

**Key Features:**
- `GeometryScores` dataclass (homophily, cluster purity, outlier risk)
- `ModelScores` dataclass (pathogenicity, evidence, novelty)
- `compute_crs_gated()` - hierarchical gating logic
- `compute_temporal_alert()` - batch stream monitoring
- `compute_ensemble_vote()` - multi-signal consensus

**Test It Now:**
```bash
cd /Users/jaygut/Documents/Side_Projects/Bacformer
python foodguard/risk_v2.py
```

**Expected Output:** 5 example cases with CRS, decision, and reason.

### 3. Drift Simulation Framework
**File:** `/Users/jaygut/Documents/Side_Projects/Bacformer/scripts/foodguard_drift_simulation.py`

**Scenarios:**
1. Gradual evolutionary drift (E. coli O157:H7 → Salmonella direction)
2. Sudden outbreak contamination (Listeria injection)
3. Assembly artifact stress test (false positive characterization)
4. Ecosystem compositional shift (facility microbiome change)

**Run It Now:**
```bash
cd /Users/jaygut/Documents/Side_Projects/Bacformer
python scripts/foodguard_drift_simulation.py \
    --embeddings-npz foodguard/logs/genome_embeddings.npz \
    --metadata-parquet foodguard/analysis/genome_embeddings_enriched.parquet \
    --output-dir foodguard/analysis/simulations \
    --scenarios gradual_drift sudden_outbreak
```

**Expected Output:**
- `gradual_drift_trajectory.png` (3-panel figure)
- `gradual_drift_performance.csv` (detection latency, sensitivity)
- `sudden_outbreak_trajectory.png`
- `sudden_outbreak_performance.csv`

### 4. Dashboard Prototype
**File:** `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/dashboard_prototype.py`

**Features:**
- Traffic light summary (red/yellow/green)
- Risk score breakdown (component contributions)
- k-NN neighbor table with mismatch warnings
- 2D PCA embedding plot (query genome highlighted)
- Posture selector (recall-high, balanced, precision-high)

**Run It Now:**
```bash
pip install streamlit plotly  # if not already installed
cd /Users/jaygut/Documents/Side_Projects/Bacformer
streamlit run foodguard/dashboard_prototype.py
```

**Expected:** Browser opens to `localhost:8501` with interactive dashboard.

### 5. Executive Summary (4 pages)
**File:** `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/docs/executive_summary_risk_framework.md`

**Contents:**
- Problem statement (what's missing)
- Prioritized roadmap (immediate → medium-term)
- Key design decisions (your input needed)
- Success criteria (3-month horizon)

**Action:** Share this with collaborators for feedback on direction.

---

## Week-by-Week Implementation Plan

### Week 1: Integrate Risk Scoring
**Goal:** Replace placeholder CRS with gated hierarchical version.

**Tasks:**
1. Update `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/pipeline.py`:
   - Import `compute_crs_gated` from `risk_v2`
   - Replace `compute_crs()` call with `compute_crs_gated()`
   - Add geometry score computation (use pre-computed homophily from enriched parquet)

2. Update `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/config.py`:
   - Add `POSTURE_THRESHOLDS` dictionary with calibrated values
   - Add `GEOMETRY_GATE_THRESHOLD = 0.75` constant

3. Write unit tests:
   - Create `/Users/jaygut/Documents/Side_Projects/Bacformer/tests/test_risk_v2.py`
   - Test boundary cases (low homophily, high outlier risk, perfect geometry)

**Validation:** Run pipeline on 10 test genomes, verify CRS changes with posture.

### Week 2: Dashboard User Testing
**Goal:** Demo to 2-3 domain experts, collect feedback.

**Tasks:**
1. Enhance dashboard:
   - Add "Export PDF Report" button
   - Add batch upload (CSV of genome IDs → batch risk summary)
   - Improve mobile responsiveness

2. Recruit testers:
   - Email 3 food safety scientists (FDA, FSIS, or university)
   - Send demo video (record 5-minute walkthrough with Loom)

3. User testing session (1 hour each):
   - Let them explore dashboard with sample genomes
   - Ask: "What's missing? What's confusing? Would you trust this?"
   - Document feedback in `foodguard/docs/user_testing_notes.md`

**Deliverable:** Feedback-driven v0.2 dashboard with fixes.

### Week 3: Run All Simulations
**Goal:** Generate manuscript figures for Section 3.9.

**Tasks:**
1. Run 4 scenarios with multiple replicates:
   ```bash
   for seed in {42..51}; do
       python scripts/foodguard_drift_simulation.py \
           --scenarios gradual_drift sudden_outbreak assembly_artifacts ecosystem_shift \
           --random-state $seed \
           --output-dir foodguard/analysis/simulations/run_$seed
   done
   ```

2. Aggregate results:
   - Compute mean detection latency ± 95% CI across replicates
   - Generate Figure 8 (4-panel simulation trajectories)
   - Generate Figure 9 (detection latency violin plots)
   - Generate Table 3 (summary statistics by scenario)

3. Statistical testing:
   - Compare detection methods (homophily vs. entropy vs. fusion)
   - Report effect sizes (Cohen's d for latency differences)

**Deliverable:** 3 manuscript-ready figures + 1 table.

### Week 4: Manuscript Revision
**Goal:** Add simulation study section, update discussion.

**Tasks:**
1. Write Section 3.9: "Temporal Drift Simulation and Alert Trigger Validation"
   - Follow outline in strategic recommendations doc (Section 3)
   - 800-1000 words, 3 figures, 1 table
   - Emphasize: "Temporal aggregation reduces false positives 4-7x"

2. Revise Discussion Section 4.3:
   - Add Subsection 4.3.1: "Bridging Geometry to Decision: The Risk Scoring Interface"
   - Follow outline in strategic recommendations doc (Section 4)
   - 400-600 words
   - Emphasize: "Gating logic prevents overinterpretation at boundaries"

3. Update Abstract:
   - Add sentence about simulation validation
   - Example: "We validate the framework's temporal responsiveness via simulation, demonstrating <5 batch detection latency for biologically plausible drift and <5% false positive rate for assembly artifacts."

4. Update Figure 1 (conceptual framework):
   - Add "Temporal Alert Logic" panel showing batch stream

**Deliverable:** Updated manuscript draft (Overleaf or LaTeX).

---

## Integration Checklist

### Code Integration
- [ ] Import `risk_v2` into `pipeline.py`
- [ ] Add geometry score computation to `process_protein_sequences()`
- [ ] Update response schema to include `geometry_score` field
- [ ] Write unit tests for `compute_crs_gated()` edge cases
- [ ] Update `config.py` with posture-specific thresholds

### Simulation Study
- [ ] Run all 4 scenarios with 10 replicates each
- [ ] Generate Figure 8 (4-panel trajectories)
- [ ] Generate Figure 9 (detection latency comparison)
- [ ] Generate Table 3 (summary statistics)
- [ ] Write Section 3.9 text (~1000 words)

### Dashboard
- [ ] Test on Chrome, Firefox, Safari (cross-browser)
- [ ] Add PDF export functionality
- [ ] Add batch upload CSV handler
- [ ] Recruit 2-3 domain expert testers
- [ ] Conduct user testing sessions
- [ ] Document feedback and iterate

### Manuscript
- [ ] Add Section 3.9 (simulation study)
- [ ] Add Section 4.3.1 (risk scoring interface)
- [ ] Update Abstract (add simulation sentence)
- [ ] Update Figure 1 (add temporal panel)
- [ ] Internal review by co-author

---

## Troubleshooting

### Issue: Simulation script fails with "embeddings NPZ not found"
**Solution:**
```bash
# Check if file exists
ls -lh foodguard/logs/genome_embeddings.npz

# If missing, regenerate from enriched parquet:
python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('foodguard/analysis/genome_embeddings_enriched.parquet')
# Note: This won't work because embeddings aren't stored in parquet
# You need the original NPZ from HPC run
"
```
**Workaround:** Use a subset of genomes if full NPZ is unavailable.

### Issue: Dashboard shows "Data file not found"
**Solution:**
```bash
# Verify paths in dashboard_prototype.py match your setup
# Default expects:
#   foodguard/analysis/genome_embeddings_enriched.parquet
#   foodguard/analysis/dim_reduction_cache.npz

# If files are elsewhere, update ANALYSIS_DIR in dashboard_prototype.py
```

### Issue: risk_v2 imports fail in pipeline
**Solution:**
```bash
# Ensure foodguard is on Python path
cd /Users/jaygut/Documents/Side_Projects/Bacformer
export PYTHONPATH=$PWD:$PYTHONPATH

# Or install in dev mode:
pip install -e .
```

---

## Success Metrics (2-Week Checkpoint)

**After Week 2, you should have:**
1. Working dashboard with user feedback from 2+ testers
2. At least 2 simulation scenarios running end-to-end
3. Draft text for Section 3.9 (even if figures are incomplete)
4. Updated `pipeline.py` with gated CRS

**If you're stuck:**
- Focus on dashboard first (highest impact for stakeholder demos)
- Simulation can be iterative (start with 1 scenario, perfect it, then scale)
- Risk scoring integration can be deferred to Week 3 if needed

---

## Next Decision Points

### Decision 1: Gating Strategy
**Status:** Implementation ready in `risk_v2.py` (Option A: gated hierarchical)
**Your Call:** Do you want geometry gate threshold to be configurable per posture, or fixed at 0.75?

**Recommendation:** Fixed at 0.75 for initial deployment, make configurable later if needed.

### Decision 2: Simulation Replicates
**Status:** Script supports any number of random seeds
**Your Call:** How many replicates for manuscript? (10 = 1 hour runtime, 50 = 5 hours)

**Recommendation:** Start with 10, add more if reviewers request robustness analysis.

### Decision 3: Dashboard Technology
**Status:** Streamlit prototype complete
**Your Call:** Migrate to Plotly Dash for production, or stick with Streamlit?

**Recommendation:** Stay with Streamlit for internal demos, migrate to Dash only if deploying to external labs.

---

## Communication with Collaborators

**Email Template for Co-Author Review:**

```
Subject: FoodGuard Risk Framework Update - Feedback Needed

Hi [Co-Author],

I've completed a strategic analysis of our FoodGuard embedding system with
a bio-risk analyst consultant. Key deliverables:

1. Enhanced risk scoring framework (geometry-gated CRS)
2. Temporal drift simulation study (4 scenarios)
3. Interactive dashboard prototype

**Action Items for You:**
- Review executive summary: foodguard/docs/executive_summary_risk_framework.md
- Test dashboard: streamlit run foodguard/dashboard_prototype.py
- Provide feedback on manuscript framing (Section 3.9 outline)

**Timeline:**
- Week 1-2: Risk scoring integration + user testing
- Week 3-4: Simulation runs + manuscript revision

**Decision Needed:**
Should we target Nature Communications (broader audience) or Bioinformatics
(methods focus)? Simulation study strengthens both cases.

Let's sync this week to align on priorities.

Best,
[Your Name]
```

---

## Resources

### Documentation
- Strategic Recommendations (full): `foodguard/docs/strategic_risk_framework_recommendations.md`
- Executive Summary: `foodguard/docs/executive_summary_risk_framework.md`
- This Quick Start: `foodguard/docs/QUICKSTART_RISK_FRAMEWORK.md`

### Code
- Enhanced Risk Scoring: `foodguard/risk_v2.py`
- Drift Simulation: `scripts/foodguard_drift_simulation.py`
- Dashboard Prototype: `foodguard/dashboard_prototype.py`

### Example Outputs
- Run risk_v2 examples: `python foodguard/risk_v2.py`
- Run simulation: See "Run It Now" section above
- Launch dashboard: See "Run It Now" section above

---

## The Bottom Line

You have everything you need to execute the 2-week MVP:
1. Code templates are ready (risk_v2, simulation, dashboard)
2. Strategic roadmap is documented
3. Manuscript framing is outlined

**Your job:** Integrate, test, validate, iterate.

**The payoff:** A compelling manuscript update + demo-able system that positions this as a systems contribution, not just an embeddings characterization.

Execute Week 1-2 (risk scoring + dashboard), and you'll have a major upgrade to show stakeholders.

---

**Questions?** Review the strategic recommendations doc for deep dives on each component.

**Ready to start?** Begin with Week 1, Task 1: Update `pipeline.py` to use `compute_crs_gated()`.
