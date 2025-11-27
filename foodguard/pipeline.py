from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PipelineConfig
from .novelty import embedding_deviation
from .risk import compute_crs
from .utils import cache as cache_utils

logger = logging.getLogger(__name__)


class FoodGuardPipeline:
    """End-to-end orchestration for FoodGuard AI MVP.

    This pipeline operates in two modes:
    - Real mode: Uses repository preprocessing and embedding utilities plus a
      Bacformer classification head to produce PS; computes NS/ES placeholders.
    - Stub mode: Returns deterministic synthetic outputs for offline testing.

    Real mode performs lazy imports to avoid import-time failures when optional
    dependencies are missing in CI.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        model: Any = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_stub: Optional[bool] = None,
        return_embeddings: Optional[bool] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        # Allow kwargs override for convenience
        if model_path is not None:
            self.config.model_path = model_path
        if device is not None:
            self.config.device = device
        if use_stub is not None:
            self.config.use_stub = use_stub
        if return_embeddings is not None:
            self.config.return_embeddings = return_embeddings

        self._clf = model  # Bacformer classification model (optional)
        self._plm = None  # ESM-2 model
        self._tok = None  # ESM-2 tokenizer
        self._plm_model_id = "facebook/esm2_t12_35M_UR50D"
        self._last_cache_info: Optional[Dict[str, Any]] = None

    # ------------------------------ Public API ------------------------------ #
    def process_genome(self, genome_path: str) -> Dict[str, Any]:
        start = time.perf_counter()
        if self.config.use_stub:
            out = self._stub_response()
            out["timing_sec"] = float(time.perf_counter() - start)
            return out

        proteins = self._preprocess_genome(genome_path)
        return self.process_protein_sequences(proteins)

    def process_protein_sequences(self, protein_sequences: List[List[str]] | List[str]) -> Dict[str, Any]:
        start = time.perf_counter()
        if self.config.use_stub:
            out = self._stub_response()
            out["timing_sec"] = float(time.perf_counter() - start)
            return out

        # Lazy imports to keep CI light when running tests
        from bacformer.pp.embed_prot_seqs import (
            load_plm,
            compute_genome_protein_embeddings,
            protein_embeddings_to_inputs,
        )
        from bacformer.modeling import BacformerForGenomeClassification
        import torch

        # Load pLM if needed
        if self._plm is None or self._tok is None:
            self._plm, self._tok = load_plm(self._plm_model_id, model_type="esm2")

        prot_means, cache_info = self._get_protein_embeddings(
            protein_sequences=protein_sequences,
            compute_fn=compute_genome_protein_embeddings,
            tokenizer=self._tok,
            batch_size=64,
            max_prot_seq_len=self.config.cache_max_prot_seq_len,
        )
        self._last_cache_info = cache_info

        # Convert proteins to Bacformer inputs
        inputs = protein_embeddings_to_inputs(
            protein_embeddings=prot_means,
            max_n_proteins=6000,
            max_n_contigs=1000,
        )

        # Load classifier lazily
        if self._clf is None:
            if not self.config.model_path:
                raise ValueError("model_path must be set or a model provided to FoodGuardPipeline")
            self._clf = BacformerForGenomeClassification.from_pretrained(self.config.model_path).eval()

        device = (
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.bfloat16 if device.startswith("cuda") and hasattr(torch, "bfloat16") else torch.float32
        self._clf.to(device)

        # Forward pass
        with torch.no_grad():
            out = self._clf(
                protein_embeddings=inputs["protein_embeddings"].to(device=device, dtype=dtype),
                special_tokens_mask=inputs["special_tokens_mask"].to(device),
                token_type_ids=inputs["token_type_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                return_dict=True,
            )
            logits = out.logits  # [1, num_labels] or [1]
            if logits.shape[-1] == 1:
                ps = float(torch.sigmoid(logits).cpu().item())
            else:
                ps = float(torch.softmax(logits, dim=-1)[0].max().cpu().item())

        # Calibrated PS (identity placeholder)
        ps_cal = ps

        # Genome embedding for retrieval / novelty (mean pooling)
        genome_emb: Optional[np.ndarray] = None
        if self.config.return_embeddings:
            hs = out.last_hidden_state  # [1, L, H]
            genome_emb = hs.mean(dim=1).squeeze(0).float().cpu().numpy()

        # Novelty proxy based on embedding deviation (optional library later)
        ns = embedding_deviation(genome_emb, threat_library=None)
        # Evidence score placeholder (e.g., attention strength on VFDB/CARD regions)
        es = 0.0

        crs_info = compute_crs(ps=ps_cal, ns=ns, es=es, posture=self.config.posture_id)

        timing = float(time.perf_counter() - start)
        response = self._build_response(
            ps=ps,
            ps_cal=ps_cal,
            ns=ns,
            es=es,
            crs_info=crs_info,
            genome_embedding=(genome_emb.tolist() if genome_emb is not None else None),
            evidence=[],
            similar_genomes=[],
            timing=timing,
        )
        if cache_info.get("enabled"):
            response["cache"] = cache_info
        return response

    # ------------------------------ Internals ------------------------------- #
    def _preprocess_genome(self, genome_path: str) -> List[List[str]]:
        from bacformer.pp.preprocess import preprocess_genome_assembly

        pre = preprocess_genome_assembly(genome_path)
        # `protein_sequence` may be nested by contig (preferred)
        return pre["protein_sequence"]

    def _stub_response(self) -> Dict[str, Any]:
        # Deterministic synthetic baseline
        ps = 0.75
        ns = 0.10
        es = 0.40
        crs_info = compute_crs(ps=ps, ns=ns, es=es, posture=self.config.posture_id)
        return self._build_response(
            ps=ps,
            ps_cal=ps,  # identity calibration placeholder
            ns=ns,
            es=es,
            crs_info=crs_info,
            genome_embedding=None,
            evidence=[],
            similar_genomes=[],
            timing=0.0,
        )

    def _build_response(
        self,
        *,
        ps: float,
        ps_cal: float,
        ns: float,
        es: float,
        crs_info: Dict[str, float | str],
        genome_embedding: Optional[List[float]],
        evidence: List[Dict[str, Any]],
        similar_genomes: List[Dict[str, Any]],
        timing: float,
    ) -> Dict[str, Any]:
        return {
            "classification": {"ps": ps, "calibrated": ps_cal},
            "novelty": {"ns": ns, "status": "beta", "threshold": None},
            "evidence": evidence,
            "risk": {"es": es, **crs_info},
            "embeddings": {"genome": genome_embedding, "per_protein": False},
            "similar_genomes": similar_genomes,
            "calibration": {
                "version": self.config.calibration_version,
                "thresholds": self.config.thresholds_version,
            },
            "timing_sec": timing,
        }

    def _get_protein_embeddings(
        self,
        protein_sequences: List[List[str]] | List[str],
        *,
        compute_fn,
        tokenizer,
        batch_size: int,
        max_prot_seq_len: int,
    ) -> tuple[list[list[np.ndarray]] | list[np.ndarray], Dict[str, Any]]:
        """Fetch mean ESM-2 embeddings, preferring the cache when configured."""
        cache_dir = self.config.cache_dir
        cache_enabled = bool(cache_dir)
        genome_pooling_method = None
        cache_hit = False
        cache_lookup_sec = 0.0
        compute_sec = 0.0
        key: Optional[str] = None
        prot_means = None

        if cache_enabled:
            key = cache_utils.make_key(
                protein_sequences=protein_sequences,
                model_id=self._plm_model_id,
                model_type="esm2",
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=genome_pooling_method,
            )
            cache_start = time.perf_counter()
            prot_means = cache_utils.load(cache_dir, key)
            cache_lookup_sec = time.perf_counter() - cache_start
            cache_hit = prot_means is not None

        if prot_means is None:
            compute_start = time.perf_counter()
            prot_means = compute_fn(
                model=self._plm,
                tokenizer=tokenizer,
                protein_sequences=protein_sequences,
                model_type="esm2",
                batch_size=batch_size,
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=genome_pooling_method,
            )
            compute_sec = time.perf_counter() - compute_start
            if cache_enabled and key is not None:
                try:
                    cache_utils.save(cache_dir, key, prot_means)
                except Exception as exc:  # pragma: no cover - best-effort cache write
                    logger.warning("Failed to cache embeddings for key=%s: %s", key, exc)
        protein_count = self._count_proteins(protein_sequences)

        cache_info: Dict[str, Any] = {
            "enabled": cache_enabled,
            "hit": cache_hit if cache_enabled else None,
            "lookup_sec": cache_lookup_sec if cache_enabled else None,
            "compute_sec": compute_sec,
            "proteins": protein_count,
        }

        if cache_enabled and self.config.cache_log_metrics:
            log_level = logging.INFO if cache_hit else logging.WARNING
            key_suffix = (key[-12:] if key else "n/a")
            logger.log(
                log_level,
                "esm2_cache_%s key_suffix=%s proteins=%d lookup=%.4fs compute=%.4fs",
                "hit" if cache_hit else "miss",
                key_suffix,
                protein_count,
                cache_lookup_sec,
                compute_sec,
            )

        return prot_means, cache_info

    @staticmethod
    def _count_proteins(protein_sequences: List[List[str]] | List[str]) -> int:
        if not protein_sequences:
            return 0
        if isinstance(protein_sequences[0], list):
            return sum(len(contig) for contig in protein_sequences)  # type: ignore[arg-type]
        return len(protein_sequences)  # type: ignore[arg-type]

    @property
    def last_cache_info(self) -> Optional[Dict[str, Any]]:
        """Expose metrics from the most recent cache lookup (for diagnostics/tests)."""
        return self._last_cache_info
