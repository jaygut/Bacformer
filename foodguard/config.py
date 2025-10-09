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

