"""High-level cache affordances for ESM-2 protein embeddings.

This module provides thin wrappers and naming guidance around the
existing cache helpers in `bacformer.pp.embed_prot_seqs`:

- `_make_cache_key(protein_sequences, model_id, model_type, max_prot_seq_len, genome_pooling_method)`
- `_cache_load(cache_dir, key)` / `_cache_save(cache_dir, key, obj)`

FoodGuard’s pipeline should use these helpers (or call the higher-level
`add_protein_embeddings`) to avoid recomputing per-protein ESM-2 means.

Implementation deferred: we will wire to these utilities directly in the
pipeline to keep a single source of truth for cache format and keys.
"""

from __future__ import annotations

from typing import Any, Optional

from bacformer.pp.embed_prot_seqs import _cache_load, _cache_save, _make_cache_key


def load(cache_dir: str, key: str) -> Optional[Any]:
    return _cache_load(cache_dir, key)


def save(cache_dir: str, key: str, obj: Any) -> None:
    _cache_save(cache_dir, key, obj)


def make_key(
    protein_sequences,
    model_id: str,
    model_type: str,
    max_prot_seq_len: int,
    genome_pooling_method: str | None,
) -> str:
    return _make_cache_key(
        protein_sequences=protein_sequences,
        model_id=model_id,
        model_type=model_type,
        max_prot_seq_len=max_prot_seq_len,
        genome_pooling_method=genome_pooling_method,
    )

