"""
Enhanced risk scoring framework for FoodGuard surveillance system.

This module implements the multi-tiered Combined Risk Score (CRS) with
geometry-aware gating, as outlined in the strategic recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeometryScores:
    """Geometry-derived confidence signals from embedding space."""

    homophily_confidence: float  # HCS: k-NN label agreement (0-1)
    cluster_purity: float  # CPS: Dominant label fraction in cluster (0-1)
    outlier_risk: float  # ORS: Distance-based outlier score (0-1)
    centroid_proximity: float  # CenPS: Normalized distance to nearest species centroid (0-1)
    k_neighbors: int  # Number of neighbors used for homophily
    cluster_id: int  # Cluster assignment (-1 if noise)


@dataclass
class ModelScores:
    """Model-derived signals (Bacformer + contextual)."""

    pathogenicity_score: float  # PS: Calibrated pathogenicity probability (0-1)
    evidence_score: float  # ES: Attention-weighted virulence factor score (0-1)
    novelty_score: float  # NS: Embedding deviation from threat library (0-1)
    calibration_method: str  # e.g., "isotonic", "platt", "identity"


@dataclass
class CombinedRiskAssessment:
    """Final risk assessment with decision logic and provenance."""

    crs: float  # Combined Risk Score (0-1)
    confidence: str  # "high", "medium", "low"
    decision: str  # "proceed", "review", "defer_to_expert", "escalate"
    risk_level: str  # "low", "medium", "high"
    reason: str  # Human-readable explanation
    posture: str  # Operating posture (recall_high, balanced, precision_high)
    geometry_score: float  # Composite geometry confidence
    components: Dict[str, float]  # Breakdown of all scores
    nearest_neighbors: Optional[List[Dict]] = None  # k-NN provenance (optional)
    alert_metadata: Optional[Dict] = None  # Temporal/batch context (optional)


def clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def compute_geometry_confidence(geom: GeometryScores) -> float:
    """
    Compute composite geometry confidence score.

    Combines homophily, cluster purity, and outlier risk into a single
    confidence measure. Lower outlier risk → higher confidence.
    """
    base_confidence = min(geom.homophily_confidence, geom.cluster_purity)

    # Penalize outliers
    outlier_penalty = 1.0 - geom.outlier_risk

    # Weight cluster quality by whether it's noise
    cluster_weight = 0.0 if geom.cluster_id == -1 else 1.0

    geometry_confidence = base_confidence * outlier_penalty * (0.5 + 0.5 * cluster_weight)

    return clamp01(geometry_confidence)


def compute_crs_gated(
    geom: GeometryScores,
    model: ModelScores,
    posture: str = "recall_high",
    geometry_threshold: float = 0.75,
) -> CombinedRiskAssessment:
    """
    Compute Combined Risk Score with geometry-aware gating.

    This implements the hierarchical gating strategy:
    1. Geometry gate: Check if embedding neighborhood is reliable
    2. If geometry fails → defer to expert or return conservative high risk
    3. If geometry passes → weighted fusion of PS, ES, NS with outlier boost

    Args:
        geom: Geometry-derived scores from embedding space
        model: Model-derived scores (pathogenicity, evidence, novelty)
        posture: Operating posture ("recall_high", "balanced", "precision_high")
        geometry_threshold: Minimum geometry confidence to proceed (default: 0.75)

    Returns:
        CombinedRiskAssessment with CRS, confidence, decision, and provenance
    """
    # Stage 1: Compute geometry confidence
    geometry_confidence = compute_geometry_confidence(geom)

    # Stage 2: Geometry gate
    if geometry_confidence < geometry_threshold:
        # Low geometric confidence → conservative handling
        if posture == "recall_high":
            # Precautionary principle: flag as high risk when uncertain
            return CombinedRiskAssessment(
                crs=0.80,
                confidence="low",
                decision="defer_to_expert",
                risk_level="high_uncertain",
                reason="boundary_case_low_homophily",
                posture=posture,
                geometry_score=geometry_confidence,
                components={
                    "ps": model.pathogenicity_score,
                    "es": model.evidence_score,
                    "ns": model.novelty_score,
                    "ors": geom.outlier_risk,
                    "hcs": geom.homophily_confidence,
                    "cps": geom.cluster_purity,
                },
            )
        else:
            # Balanced/precision posture: require secondary analysis
            return CombinedRiskAssessment(
                crs=0.50,
                confidence="low",
                decision="require_secondary_analysis",
                risk_level="medium_uncertain",
                reason="ambiguous_embedding_neighborhood",
                posture=posture,
                geometry_score=geometry_confidence,
                components={
                    "ps": model.pathogenicity_score,
                    "es": model.evidence_score,
                    "ns": model.novelty_score,
                    "ors": geom.outlier_risk,
                    "hcs": geom.homophily_confidence,
                    "cps": geom.cluster_purity,
                },
            )

    # Stage 3: Pathogenicity-weighted fusion (geometry passed gate)
    weights = {
        "recall_high": {"ps": 0.50, "es": 0.30, "ns": 0.20},
        "balanced": {"ps": 0.40, "es": 0.35, "ns": 0.25},
        "precision_high": {"ps": 0.35, "es": 0.40, "ns": 0.25},
    }
    w = weights.get(posture, weights["recall_high"])

    # Base score from model components
    base_score = w["ps"] * model.pathogenicity_score + w["es"] * model.evidence_score + w["ns"] * model.novelty_score

    # Outlier boost: increase CRS if genome is geometric outlier (potential novel threat)
    outlier_boost = 0.15 * geom.outlier_risk if geom.outlier_risk > 0.7 else 0.0

    crs = clamp01(base_score + outlier_boost)

    # Stage 4: Confidence calibration and decision logic
    confidence = "high" if geometry_confidence > 0.90 else "medium"

    # Risk level and decision based on CRS thresholds
    if crs >= 0.85:
        risk_level = "high"
        decision = "escalate" if confidence == "medium" else "review"
        reason = "high_pathogenicity_score_with_evidence"
    elif crs >= 0.75:
        risk_level = "high"
        decision = "review"
        reason = "moderate_pathogenicity_elevated_by_context"
    elif crs >= 0.50:
        risk_level = "medium"
        decision = "review"
        reason = "borderline_pathogenicity_requires_confirmation"
    else:
        risk_level = "low"
        decision = "proceed" if confidence == "high" else "review"
        reason = "low_risk_high_confidence_neighborhood" if confidence == "high" else "low_risk_medium_confidence"

    # Special case: high outlier risk overrides low CRS
    if geom.outlier_risk > 0.8 and crs < 0.6:
        risk_level = "medium"
        decision = "review"
        reason = "geometric_outlier_requires_qc_or_novelty_assessment"

    return CombinedRiskAssessment(
        crs=crs,
        confidence=confidence,
        decision=decision,
        risk_level=risk_level,
        reason=reason,
        posture=posture,
        geometry_score=geometry_confidence,
        components={
            "ps": model.pathogenicity_score,
            "es": model.evidence_score,
            "ns": model.novelty_score,
            "ors": geom.outlier_risk,
            "hcs": geom.homophily_confidence,
            "cps": geom.cluster_purity,
            "cenps": geom.centroid_proximity,
            "base_score": base_score,
            "outlier_boost": outlier_boost,
        },
    )


def compute_temporal_alert(
    batch_scores: List[float],
    window_size: int = 5,
    threshold: float = 0.75,
    min_consecutive: int = 2,
) -> Dict[str, float | bool | str]:
    """
    Temporal alert trigger using sliding window majority vote with exponential decay.

    Triggers alert if:
    - At least `min_consecutive` batches exceed threshold consecutively, OR
    - Exponentially weighted moving average (EWMA) exceeds threshold

    Args:
        batch_scores: CRS values for recent batches (time-ordered)
        window_size: Number of recent batches to consider
        threshold: CRS threshold for alert
        min_consecutive: Minimum consecutive exceedances to trigger

    Returns:
        Alert status, metrics, and trigger reason
    """
    if not batch_scores:
        return {
            "alert": False,
            "consecutive_exceedances": 0,
            "ewma": 0.0,
            "trigger_reason": "no_data",
        }

    recent = batch_scores[-window_size:]

    # Count consecutive exceedances (from most recent backward)
    consecutive_count = 0
    for score in reversed(recent):
        if score >= threshold:
            consecutive_count += 1
        else:
            break

    # Exponentially weighted moving average (more weight on recent batches)
    weights = np.exp(np.linspace(-1, 0, len(recent)))
    weights /= weights.sum()
    ewma = float(np.dot(recent, weights))

    # Trigger logic
    alert = (consecutive_count >= min_consecutive) or (ewma >= threshold)
    trigger_reason = "consecutive" if consecutive_count >= min_consecutive else ("ewma" if alert else "none")

    return {
        "alert": alert,
        "consecutive_exceedances": consecutive_count,
        "ewma": ewma,
        "trigger_reason": trigger_reason,
        "window_size": len(recent),
        "threshold": threshold,
    }


def compute_ensemble_vote(
    knn_vote: float,  # k-NN majority vote fraction
    cluster_vote: float,  # Cluster dominant label score
    bacformer_prob: float,  # Bacformer calibrated probability
    centroid_vote: float,  # Nearest centroid assignment confidence
    consensus_threshold: int = 3,
) -> Tuple[str, float, List[str]]:
    """
    Ensemble voting across multiple signals for high-stakes decisions.

    Requires consensus (≥3 out of 4 signals) to assign "pathogenic" label.

    Args:
        knn_vote: Fraction of k-NN neighbors labeled pathogenic (0-1)
        cluster_vote: Fraction of cluster labeled pathogenic (0-1)
        bacformer_prob: Bacformer pathogenicity probability (0-1)
        centroid_vote: Confidence that nearest centroid is pathogenic (0-1)
        consensus_threshold: Minimum number of signals that must agree

    Returns:
        (predicted_label, confidence, supporting_signals)
    """
    # Binary threshold for each signal (>0.5 = pathogenic)
    signals = {
        "knn": knn_vote > 0.5,
        "cluster": cluster_vote > 0.5,
        "bacformer": bacformer_prob > 0.5,
        "centroid": centroid_vote > 0.5,
    }

    votes_pathogenic = sum(signals.values())
    supporting = [name for name, vote in signals.items() if vote]

    if votes_pathogenic >= consensus_threshold:
        # Consensus for pathogenic
        confidence = min(knn_vote, cluster_vote, bacformer_prob, centroid_vote)
        return "pathogenic", confidence, supporting
    elif votes_pathogenic <= (4 - consensus_threshold):
        # Consensus for non-pathogenic
        confidence = min(1 - knn_vote, 1 - cluster_vote, 1 - bacformer_prob, 1 - centroid_vote)
        return "non_pathogenic", confidence, supporting
    else:
        # No consensus
        return "uncertain", 0.5, supporting


def calibrate_thresholds_from_cv(
    cv_predictions: np.ndarray,  # CV predicted probabilities
    cv_labels: np.ndarray,  # True binary labels
    target_sensitivity: float = 0.95,
) -> Dict[str, float]:
    """
    Calibrate decision thresholds from cross-validation results.

    Finds threshold that achieves target sensitivity (recall) and reports
    corresponding precision, specificity.

    Args:
        cv_predictions: Predicted probabilities from CV (e.g., PS or CRS)
        cv_labels: True binary labels (0/1)
        target_sensitivity: Desired sensitivity (recall) for threshold

    Returns:
        Calibrated thresholds and performance metrics
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    # Precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(cv_labels, cv_predictions)

    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(recall - target_sensitivity))
    threshold_recall_target = pr_thresholds[min(idx, len(pr_thresholds) - 1)]
    precision_at_target = precision[idx]

    # ROC curve for specificity
    fpr, tpr, roc_thresholds = roc_curve(cv_labels, cv_predictions)
    roc_idx = np.argmin(np.abs(tpr - target_sensitivity))
    specificity_at_target = 1 - fpr[roc_idx]

    return {
        "threshold": float(threshold_recall_target),
        "sensitivity": float(recall[idx]),
        "precision": float(precision_at_target),
        "specificity": float(specificity_at_target),
        "target_sensitivity": target_sensitivity,
    }


# Example usage and test cases
if __name__ == "__main__":
    # Example 1: High-confidence pathogenic genome
    geom_high = GeometryScores(
        homophily_confidence=0.95,
        cluster_purity=0.98,
        outlier_risk=0.10,
        centroid_proximity=0.90,
        k_neighbors=20,
        cluster_id=5,
    )
    model_high = ModelScores(
        pathogenicity_score=0.88,
        evidence_score=0.65,
        novelty_score=0.12,
        calibration_method="isotonic",
    )
    assessment_high = compute_crs_gated(geom_high, model_high, posture="recall_high")
    print("High-confidence pathogenic case:")
    print(f"  CRS: {assessment_high.crs:.3f}")
    print(f"  Risk Level: {assessment_high.risk_level}")
    print(f"  Decision: {assessment_high.decision}")
    print(f"  Reason: {assessment_high.reason}\n")

    # Example 2: Boundary case (low homophily)
    geom_boundary = GeometryScores(
        homophily_confidence=0.72,  # Below threshold
        cluster_purity=0.88,
        outlier_risk=0.35,
        centroid_proximity=0.65,
        k_neighbors=20,
        cluster_id=12,
    )
    model_boundary = ModelScores(
        pathogenicity_score=0.75,
        evidence_score=0.40,
        novelty_score=0.25,
        calibration_method="isotonic",
    )
    assessment_boundary = compute_crs_gated(geom_boundary, model_boundary, posture="recall_high")
    print("Boundary case (low homophily):")
    print(f"  CRS: {assessment_boundary.crs:.3f}")
    print(f"  Risk Level: {assessment_boundary.risk_level}")
    print(f"  Decision: {assessment_boundary.decision}")
    print(f"  Reason: {assessment_boundary.reason}\n")

    # Example 3: Geometric outlier (potential novelty)
    geom_outlier = GeometryScores(
        homophily_confidence=0.85,
        cluster_purity=0.60,
        outlier_risk=0.88,  # High outlier score
        centroid_proximity=0.30,
        k_neighbors=20,
        cluster_id=-1,  # HDBSCAN noise
    )
    model_outlier = ModelScores(
        pathogenicity_score=0.45,
        evidence_score=0.20,
        novelty_score=0.82,  # High novelty
        calibration_method="isotonic",
    )
    assessment_outlier = compute_crs_gated(geom_outlier, model_outlier, posture="balanced")
    print("Geometric outlier (potential novel threat):")
    print(f"  CRS: {assessment_outlier.crs:.3f}")
    print(f"  Risk Level: {assessment_outlier.risk_level}")
    print(f"  Decision: {assessment_outlier.decision}")
    print(f"  Reason: {assessment_outlier.reason}\n")

    # Example 4: Temporal alert trigger
    batch_crs = [0.45, 0.52, 0.68, 0.77, 0.81, 0.79, 0.82]
    alert_result = compute_temporal_alert(batch_crs, window_size=5, threshold=0.75, min_consecutive=2)
    print("Temporal alert analysis:")
    print(f"  Alert triggered: {alert_result['alert']}")
    print(f"  Consecutive exceedances: {alert_result['consecutive_exceedances']}")
    print(f"  EWMA: {alert_result['ewma']:.3f}")
    print(f"  Trigger reason: {alert_result['trigger_reason']}\n")

    # Example 5: Ensemble voting
    label, conf, support = compute_ensemble_vote(
        knn_vote=0.85,
        cluster_vote=0.92,
        bacformer_prob=0.78,
        centroid_vote=0.88,
        consensus_threshold=3,
    )
    print("Ensemble voting result:")
    print(f"  Predicted label: {label}")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Supporting signals: {support}")
