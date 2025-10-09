import logging
import time
from datasets import Dataset

from bacformer.pp.preprocess import preprocess_genome_assembly
from bacformer.pp.embed_prot_seqs import embed_dataset_col


def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Preprocess PAO1 GBFF to extract protein sequences (nested by contig)
    gbff_path = "files/pao1.gbff"
    row = preprocess_genome_assembly(gbff_path)

    # Flatten to count proteins and take only ~20 for a quick smoke test
    # We keep the nested structure (contigs), but truncate total proteins to ~20
    nested = row["protein_sequence"]  # List[List[str]]
    contigs = []
    total = 0
    limit = 20
    for contig in nested:
        if total >= limit:
            break
        take = contig[: max(0, min(len(contig), limit - total))]
        if take:
            contigs.append(take)
            total += len(take)

    dataset = Dataset.from_list([
        {
            "protein_sequence": contigs,  # List[List[str]]
            "contig_name": list(range(len(contigs))),
        }
    ])

    # 2) Run ESM-2 embeddings with caching enabled
    cache_dir = ".cache/esm2"

    t0 = time.time()
    ds1 = embed_dataset_col(
        dataset=dataset,
        model_path="facebook/esm2_t12_35M_UR50D",
        model_type="esm2",
        batch_size=8,
        max_prot_seq_len=1024,
        output_col="embeddings",
        genome_pooling_method="mean",  # pool to a single genome vector for speed
        cache_dir=cache_dir,
        cache_overwrite=False,
    )
    t1 = time.time()

    print("First run embedding shape:", ds1[0]["embeddings"].shape)
    print(f"First run time: {t1 - t0:.2f}s")

    # 3) Run again; should trigger a cache hit (see INFO logs)
    t2 = time.time()
    ds2 = embed_dataset_col(
        dataset=dataset,
        model_path="facebook/esm2_t12_35M_UR50D",
        model_type="esm2",
        batch_size=8,
        max_prot_seq_len=1024,
        output_col="embeddings",
        genome_pooling_method="mean",
        cache_dir=cache_dir,
        cache_overwrite=False,
    )
    t3 = time.time()

    print("Second run embedding shape:", ds2[0]["embeddings"].shape)
    print(f"Second run time: {t3 - t2:.2f}s (should be faster due to cache)")


if __name__ == "__main__":
    main()
