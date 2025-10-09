from __future__ import annotations

from foodguard import FoodGuardPipeline
from foodguard.config import PipelineConfig


def test_pipeline_stub_response_schema():
    cfg = PipelineConfig(use_stub=True)
    pipe = FoodGuardPipeline(cfg)
    out = pipe.process_genome("files/pao1.gbff")

    # Top-level keys
    assert set(out.keys()) == {
        "classification",
        "novelty",
        "evidence",
        "risk",
        "embeddings",
        "similar_genomes",
        "calibration",
        "timing_sec",
    }

    # Classification
    assert "ps" in out["classification"] and "calibrated" in out["classification"]
    assert isinstance(out["classification"]["ps"], float)

    # Novelty
    assert out["novelty"]["status"] == "beta"

    # Risk
    assert "crs" in out["risk"] and "posture" in out["risk"]

    # Embeddings
    assert "genome" in out["embeddings"] and "per_protein" in out["embeddings"]

    # Calibration meta
    assert "version" in out["calibration"] and "thresholds" in out["calibration"]

