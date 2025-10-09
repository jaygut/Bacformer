from __future__ import annotations

from typing import Optional

import numpy as np


def embedding_deviation(
    genome_embedding: Optional[np.ndarray],
    threat_library: Optional[np.ndarray] = None,
) -> float:
    """Compute a simple novelty proxy based on embedding deviation.

    If a threat library of reference embeddings is provided, compute the
    minimum cosine distance to the library and convert to a 0-1 novelty score.
    If not available, return a conservative default.
    """
    if genome_embedding is None or threat_library is None or len(threat_library) == 0:
        return 0.0

    ge = genome_embedding / (np.linalg.norm(genome_embedding) + 1e-8)
    tl = threat_library / (np.linalg.norm(threat_library, axis=1, keepdims=True) + 1e-8)
    cos_sims = tl @ ge  # shape: [N]
    # novelty increases as similarity decreases
    min_dist = 1.0 - float(cos_sims.max())
    # clamp to [0,1]
    return max(0.0, min(1.0, min_dist))

