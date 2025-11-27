from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for FoodGuardPipeline.

    Keep this minimal and environment-agnostic. Real models and catalogs
    are injected by the caller (paths or in-memory objects).
    """

    # Model identifiers or objects are passed to the pipeline at runtime.
    model_path: Optional[str] = None
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto

    # Operational posture / metadata
    posture_id: str = "recall_high"
    calibration_version: str = "dev"
    thresholds_version: str = "fg-food-v1"

    # Controls
    use_stub: bool = False  # when True, skip model I/O and return synthetic outputs (for tests)
    return_embeddings: bool = False  # include genome embedding in responses

    # ESM-2 protein embedding cache controls
    cache_dir: Optional[str] = None  # directory containing prot_emb_*.pt cache files
    cache_max_prot_seq_len: int = 1024  # aligns with ESM-2 token limit; adjust to match cache
    cache_log_metrics: bool = True  # emit cache hit/miss timing to the logger
