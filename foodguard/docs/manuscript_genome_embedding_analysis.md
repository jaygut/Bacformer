# Whole-Proteome ESM-2 Embeddings Recover Taxonomy and Enable Geometry-Aware Triage of Foodborne Bacterial Genomes

**Jay Gutierrez**\* & **Javier Correa Alvarez**\*

\* Corresponding authors

---

## Abstract

Whole-genome sequencing (WGS) has transformed foodborne pathogen surveillance [2], yet time-sensitive decision-making
remains constrained by computationally expensive alignment-centric workflows. Here we evaluate a cache-first,
protein-language-model (PLM) embedding strategy for rapid, alignment-free triage of bacterial genomes. Using 21,657
GenomeTrakr-derived assemblies spanning nine food safety–relevant taxa [2], we represent each genome by mean-pooling
per-protein embeddings from ESM-2 (`esm2_t12_35M_UR50D`, 480 dimensions) [5]. The resulting embedding space is dominated
by taxonomic structure, exhibiting near-perfect neighborhood consistency for both species and a coarse
species/pathotype-derived pathogenicity prior (mean homophily >0.99). Density-based clustering recovers species-coherent
structure with high purity and bootstrap stability, while external agreement with the binary pathogenicity prior is only
moderate—consistent with phylogenetic entanglement by design rather than embedding failure. As a within-genus stress
test, kNN separates *E. coli* O157:H7 from non-pathogenic *E. coli* with ~98% accuracy (5-fold CV), demonstrating that
known pathotype annotations are preserved in the embedding geometry even among closely related genomes. We position this
mean-pooling baseline relative to contextual genome language models that retain protein order or operon-scale context,
and outline how embedding geometry (homophily, purity, outliers) can serve as a principled confidence layer in
surveillance-oriented triage pipelines.

**Keywords:** protein language model, ESM-2, genome embedding, GenomeTrakr, clustering, homophily, outlier detection,
food safety surveillance

---

## 1. Introduction

Foodborne diseases remain a major global burden, with WHO estimates of ~600 million illnesses and ~420,000 deaths per
year attributable to unsafe food [1]. Modern surveillance increasingly relies on WGS, and networks such as FDA’s
GenomeTrakr have enabled more precise traceback and outbreak investigation at scale [2]. Despite this progress, typical
WGS workflows still depend on computationally expensive similarity search, alignment, and phylogenetic reconstruction,
creating latency bottlenecks for time-sensitive triage.

Alignment-free approaches (e.g., MinHash sketches as implemented in Mash) reduce the cost of large-scale similarity
search [3], but primarily capture nucleotide-level similarity and may not explicitly encode functional signals relevant
to virulence. In parallel, PLMs such as ESM have shown that structure and function can emerge from self-supervised
learning on large protein sequence corpora [4,5]. This has motivated a wave of genome-scale modeling efforts that treat
genomic content as language [6], either at the nucleotide level (DNABERT [10], DNABERT-2 [11], Nucleotide Transformer
[9], HyenaDNA [12], GenSLMs [13]) or at the protein/gene level (gLM [7], Protein Set Transformer [8], BacPT [18],
Bacformer [19]).

In this landscape, a practical question remains: how much can we gain from the simplest possible genome representation
built from PLMs—mean pooling per-protein embeddings—when scaled to tens of thousands of genomes under realistic
surveillance constraints? And can the geometry of the embedding space be turned into actionable signals for operational
systems (confidence, novelty, drift, anomaly triage), rather than treated as mere visualization?

### Contributions

This manuscript provides a large-scale, geometry-first characterization of whole-proteome ESM-2 embeddings for foodborne
surveillance (Figure 1). Specifically, we (i) implement a cache-first embedding pipeline that enables reproducible reuse
across analyses and downstream models; (ii) quantify the embedding space's intrinsic dimensionality and clustering
geometry; (iii) introduce homophily, purity, and outlier scores as operationally relevant confidence signals; and (iv)
position this mean-pooling baseline honestly relative to contextual genome language models designed to model operon- and
genome-scale dependencies.

---

## 2. Materials and Methods

### 2.1 Dataset and Labeling

We analyze 21,657 bacterial genomes drawn from a GenomeTrakr-derived collection and grouped into nine taxa [2]:

| Taxon | n | Label |
| --- | ---: | --- |
| *Salmonella enterica* | 7,000 | pathogenic |
| *Listeria monocytogenes* | 4,500 | pathogenic |
| *E. coli* (non-pathogenic group) | 4,312 | non-pathogenic |
| *Bacillus subtilis* | 2,361 | non-pathogenic |
| *E. coli* O157:H7 | 1,500 | pathogenic |
| *Citrobacter koseri* | 897 | non-pathogenic |
| *Listeria innocua* | 449 | non-pathogenic |
| *Escherichia fergusonii* | 438 | non-pathogenic |
| *Citrobacter freundii* | 200 | non-pathogenic |

Because isolate-level virulence phenotypes are not uniformly available, we use a **coarse pathogenicity prior** derived
from species/pathotype assignments. This label is intentionally treated as a *surveillance prior*, not as ground truth
for every individual isolate; it is suitable for studying embedding geometry, retrieval consistency, and coarse triage,
but it cannot resolve within-species virulence variation.

Accordingly, when we report within-genus stress tests (e.g., *E. coli* O157:H7 vs non-pathogenic *E. coli*), we interpret
performance as pathotype-level discriminability under known annotations—not as discovery of novel virulent isolates in
the absence of prior labeling.

### 2.2 Whole-Proteome Embeddings (ESM-2 Mean Pooling)

Protein sequences are extracted from GenBank annotations. Each protein is embedded with ESM-2
(`facebook/esm2_t12_35M_UR50D`) [5] and reduced to a single 480-dimensional vector by mean pooling over token
embeddings. Genome embeddings are then computed as mean pooling across all proteins in the genome.

This “bag-of-proteins” genome representation is deliberately simple, and corresponds to the context-free baselines used
in operon-/contig-scale genomic language modeling (e.g., mean pooling of ESM-2 protein embeddings prior to contextual
modeling in gLM) [7].

### 2.3 Cache-First Execution and Reproducibility

We use content-addressed caching of per-genome protein embedding tensors (`prot_emb_<key>.pt`), where keys incorporate
the model ID, maximum protein length (`max_prot_seq_len=1024`), pooling configuration, and a hash of input sequences.
This design makes cache hits deterministic across runs and environments, enables sharded embedding generation across
accelerators, and decouples downstream analyses and models from recomputation.

Large-scale embedding runs were executed on the EAFIT Apolo-3 HPC cluster using 2-way sharding across dual NVIDIA H100
NVL GPUs (one worker per GPU). Subsequent analytics and figure generation are CPU-feasible using the pooled embedding
bundle.

For analysis reproducibility, pooled embeddings and metadata are exported as a compressed NPZ bundle:
`foodguard/logs/genome_embeddings.npz`.

### 2.4 Dimensionality Reduction and Clustering

Embeddings are standardized (zero mean, unit variance per dimension). We compute PCA (50 components) and report
explained variance. For visualization we compute UMAP (`n_neighbors=30`, `min_dist=0.1`) [14] and t-SNE
(`perplexity=30`) [20] from PCA-reduced features.

For clustering we run HDBSCAN (`min_cluster_size=50`, `min_samples=10`) on the first 20 PCs [15]. Internal cluster
validity is quantified by silhouette score, Calinski–Harabasz index, and Davies–Bouldin index. External agreement with
the binary pathogenicity prior is reported using ARI and NMI. We assess cluster stability via bootstrap resampling and
cluster-wise Jaccard matching (50 iterations; 80% subsampling) following Hennig [16].

### 2.5 Homophily as a Geometry-Aware Confidence Signal

We compute kNN homophily (k=20 unless otherwise stated) as the fraction of a genome’s k nearest neighbors (Euclidean
distance in standardized embedding space) sharing its label, using both species labels (taxonomy consistency) and the
binary pathogenicity prior (coarse risk consistency).

We additionally compute **multi-scale homophily curves** (k ∈ {1, 5, 10, 20, 50, 100}) to characterize how label
consistency changes with neighborhood radius.

### 2.6 Outlier Detection

We operationalize outliers using two complementary criteria.

**Distance outliers** are genomes whose k-neighbor distance in 20D PCA space exceeds Q3 + 1.5×IQR, while **cluster
noise** are genomes labeled as noise by HDBSCAN (cluster = −1). We report “strict” outliers as genomes satisfying both
criteria (high-confidence anomalies).

### 2.7 Within-Genus Stress Tests

To partially mitigate label–taxonomy confounding, we report two focused stress tests in which pathogenic and
non-pathogenic priors occur within a related genus-level neighborhood. Concretely, we analyze *Listeria*
(*L. monocytogenes* vs *L. innocua*) and *Escherichia* (*E. coli* O157:H7 vs non-pathogenic *E. coli*).

For each subset we report kNN vote accuracy (k=20), balanced accuracy, mean homophily, and silhouette score. Together,
these methods operationalize the framework depicted in Figure 1 and enable the geometry-aware triage signals we report
below.

---

## 3. Results

We present results organized around the three operational signals highlighted in Figure 1: embedding geometry and
clustering (Sections 3.1–3.5), confidence via homophily (Section 3.6), and outlier detection for quality control
(Section 3.7). Within-genus stress tests (Section 3.8) validate discriminability under challenging conditions.

### 3.1 Low Dimensionality of Mean-Pooled Embeddings

PCA indicates that the embedding space is dominated by a small number of factors: PC1 explains 76.56% of variance, PC2
explains 10.95%, and only 3 PCs are needed to capture 90% of variance (Figure 2A). This strong low-dimensional
structure is consistent with broad taxonomic and proteome-level signals shaping the dominant geometry of the space.

Importantly, this result should be interpreted as a descriptive property of the current dataset and representation
(standardized, mean-pooled protein embeddings) rather than as a claim that the underlying biology is intrinsically
three-dimensional. PCA is a global linear model: it concentrates variance along directions that best explain the dataset
under a linear assumption, and additional biologically meaningful structure can reside in higher-order PCs and in
non-linear manifolds revealed by methods such as UMAP and t-SNE (Figure 3).

### 3.2 Taxonomic Structure in Low-Dimensional Projections

PCA already reveals pronounced taxonomic structure in the first two axes (Figure 2B–C), and non-linear projections via
UMAP and t-SNE sharpen the same species-level islands (Figure 3).

Qualitatively, Figure 2B suggests that the dominant PCA axis separates broad phylogenetic regimes: genomes from
Firmicutes in this dataset (e.g., *Bacillus* and *Listeria*) occupy the negative PC1 region, while Enterobacterales
(e.g., *Salmonella*, *Escherichia*, *Citrobacter*) concentrate at positive PC1 values. Within the Enterobacterales
region, the spread along PC2 and the partial overlap between closely related *Escherichia* taxa visually anticipate the
downstream pattern we quantify with homophily and within-genus stress tests: most neighborhoods are internally
consistent, but a small subset of genomes sits near decision boundaries where retrieval-based triage becomes less
reliable.

Figure 3 reinforces this interpretation. The species-colored views confirm that the "island" structure is robust across
both UMAP and t-SNE—the same local groupings emerge under different projections, indicating that the observed clusters
are not artifacts of PCA alone. Meanwhile, the pathogenicity-colored views (Figure 2C and Figure 3, top row) illustrate
why we treat the binary label as a prior rather than isolate-level ground truth: the red/blue separation largely tracks
taxonomy (because the label is species/pathotype-derived), and the most informative regions are the mixed neighborhoods
where related taxa carry different priors (e.g., *Escherichia* groups and the *Listeria* pair).

We emphasize that UMAP and especially t-SNE are visualization tools that prioritize local neighborhood structure and can
distort global geometry. Accordingly, we interpret Figure 3 in terms of neighborhood consistency and cross-method
stability rather than in terms of absolute inter-island distances or global layout.

### 3.3 Density-Based Clustering and Stability

Consistent with these projections, HDBSCAN identifies 34 clusters with 35.6% noise points—a typical outcome for
density-based clustering in biological embedding spaces with variable density. Internal validation indicates strong
cluster structure (excluding noise points): silhouette **0.555**, Calinski–Harabasz **10,849.94**, and Davies–Bouldin
**1.0719**. As a sanity check, silhouette drops well below zero under permutation of cluster labels (−0.229; 5,000-point
sample; excluding noise), confirming that the observed value reflects non-trivial structure rather than a clustering
artifact.

Bootstrap resampling with cluster-wise Jaccard matching (50 iterations; 80% subsampling) indicates that most clusters are
robust to resampling, with mean Jaccard stability **0.81** overall and **23/34 clusters** classified as stable or highly
stable (mean Jaccard ≥ 0.75).

### 3.4 Species Centroid Distances as Confusability Priors

At a coarser resolution, centroid-to-centroid distances between species provide an interpretable map of which taxa are
closest in this embedding space and therefore most likely to generate boundary cases under retrieval-based triage
(Figure 4). The matrix recapitulates the dominant taxonomic geometry: *Bacillus* and *Listeria* centroids are far from
the Enterobacterales taxa (typical distances ~40–46), consistent with the broad PC1 separation in Figure 2B. Within each
regime, distances compress and identify the most confusable neighborhoods. The closest pair is the *Listeria* duo
(*L. monocytogenes* vs *L. innocua*; 4.04). Within *Escherichia*, the non-pathogenic *E. coli* centroid lies near
*E. fergusonii* (5.34) and the O157:H7 pathotype remains relatively close to both (e.g., *E. coli* O157:H7 vs
non-pathogenic *E. coli*; 12.95). The two *Citrobacter* species are also closer to each other than to most other taxa
(10.46). Notably, some cross-genus Enterobacterales distances remain small compared to the Gram-positive/Gram-negative
gap (e.g., *Salmonella enterica* is 8.02 from *E. fergusonii* and 8.31 from non-pathogenic *E. coli*), implying that
species identity is the primary organizing signal while fine-grained separability is uneven across taxa.

These centroid distances should be read as a pragmatic "confusability map" rather than as a calibrated evolutionary
distance. Averaging collapses within-species multimodality and is sensitive to unequal sampling and within-species
variance, and Euclidean distances in standardized embedding space depend on the chosen scaling and metric. Accordingly,
we use Figure 4 to motivate which within-genus or within-order comparisons deserve conservative handling downstream, not
to claim a universal notion of biological distance.

To validate centroid distances as confusability priors, we quantified cross-species neighbor mixing (k=20) and found a
strong inverse association between centroid distance and symmetric mixing (Spearman ρ = −0.65; p = 1.5×10⁻⁵;
Figure 7). Moreover, bootstrap confidence intervals for key close pairs are narrow (e.g., *Listeria* centroid distance
4.04, 95% CI 3.99–4.12; O157:H7 vs non-pathogenic *E. coli* 12.95, 95% CI 12.71–13.21), supporting that these
confusability relationships are stable given current sampling.

### 3.5 Pathogenicity Separation and Phylogenetic Entanglement

Binary pathogenicity priors show substantial centroid separation (multivariate effect size 7.52; distance ratio
inter-/intra-class 1.106), but external agreement between clusters and pathogenicity is only moderate (ARI 0.2217; NMI
0.3429). At first glance, this may appear inconsistent with high cluster purity—yet the metrics answer different
questions. High purity indicates that most clusters are dominated by a single prior label, whereas ARI/NMI penalize
over-partitioning: splitting one label into many label-pure clusters (e.g., multiple *Salmonella* subclusters) yields
low global agreement with a two-class scheme even when each subcluster is internally homogeneous. In short, high purity
with low ARI/NMI reveals that taxonomy—and within-taxonomy substructure—is the primary organizing signal, and that
pathogenicity is entangled with phylogeny by design rather than cleanly separable as a binary attribute.

Despite this confounding, cluster membership is still actionable for triage: **33/34 clusters (97%) achieve ≥0.90
pathogenicity purity** (and similarly ≥0.90 dominant-species purity), supporting a cluster-based confidence layer.

We further stress-tested generalization by withholding each taxon and predicting its pathogenicity prior from the
remaining taxa using kNN. As expected under a species/pathotype-derived prior, zero-shot transfer across taxa is
unreliable (several taxa flip labels when held out), reinforcing that our binary labels are useful for retrieval
consistency and coarse routing within a known reference set, not for “unknown taxon” pathogenicity inference.

### 3.6 Homophily and Boundary Cases

Mean homophily is near-perfect for both labels (k=20): pathogenicity homophily **0.9929** and species homophily
**0.9923**.

Boundary cases remain informative: 102 genomes have pathogenicity homophily < 0.5, and 111 genomes have species
homophily < 0.5.

These low-homophily cases concentrate primarily in *Escherichia* groups (O157:H7 and non-pathogenic *E. coli*), rather
than in taxa that are globally well-separated (e.g., *Salmonella*).

Multi-scale homophily curves (Figure 5) show that neighborhood agreement remains high across scales for the dataset as a
whole (mean homophily 0.9958 at k=1 and 0.9862 at k=100), but the degradation is concentrated in confusable taxa. In
particular, *E. coli* O157:H7 shows a larger drop (0.973 at k=1 to 0.934 at k=100) and a substantial rise in mixed
neighborhoods as k grows: at k=100, 20.1% of O157:H7 genomes have homophily < 0.9. Conversely, taxa that are
geometrically isolated in the global embedding space (e.g., *Salmonella enterica* and *L. monocytogenes*) remain near-
perfect even at large k. Interpreted conservatively, Figure 5 operationalizes "hardness" as a scale-dependent property
of local geometry, and motivates using homophily-derived confidence to identify when retrieval-based triage should defer
to richer, context-aware modeling.

### 3.7 Outliers as QC and Novelty Candidates

Outlier analysis yields 510 distance outliers (2.4%) and 7,707 HDBSCAN noise points (35.6%); their intersection defines
**252 strict outliers (1.2%)**.

Strict outliers are enriched in *E. coli* (both groups), *E. fergusonii*, and *Salmonella*, with extreme k-neighbor
distances (up to 116.39 in PCA space). These cases are suitable for targeted QC review (assembly completeness,
contamination, annotation artifacts) or as candidates for novelty/drift monitoring in operational surveillance settings.

Using available assembly statistics as a lightweight QC proxy, strict outliers tend toward larger genomes and proteomes
(median genome size 5.18 Mb vs 4.78 Mb; median proteins 4,902 vs 4,403) and exhibit a heavy-tailed fragmentation profile
(75th percentile contig count 231 vs 54). This heterogeneity is consistent with a mixture of biologically larger genomes,
annotation variation, and genuinely problematic assemblies—reinforcing that outlier flags should trigger review rather
than be treated as definitive novelty calls.

### 3.8 Within-Genus Stress Tests and Confidence Calibration

Although the full dataset is dominated by taxonomy, within-genus subsets still show meaningful separation. We emphasize
that these experiments measure discrimination under known genus/pathotype annotations (e.g., O157:H7 vs non-pathogenic
*E. coli*), not discovery of novel virulent isolates in the absence of prior labels.

| Subset | n | kNN(20) accuracy (5-fold CV) | Balanced accuracy (5-fold CV) | Mean homophily | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Listeria* (*L. monocytogenes* vs *L. innocua*) | 4,949 | 0.9992 | 0.9955 | 0.9986 | 0.180 |
| *E. coli* (O157:H7 vs non-pathogenic) | 5,812 | 0.9842 | 0.9830 | 0.9752 | 0.392 |

For the *E. coli* stress test, 5-fold CV accuracy is 0.984 (95% CI 0.981–0.987) with balanced accuracy 0.983 (95% CI
0.979–0.987), confirming that performance is not an artifact of a single split.

This indicates that even the mean-pooled baseline can support rapid, retrieval-based triage among related taxa, while
also revealing where confusions persist (particularly within *Escherichia*).

Beyond point accuracy, Figure 6 provides an actionable calibration view for surveillance-oriented triage: in 5-fold CV on
the *E. coli* subset, deferring the lowest-consensus neighborhoods (kNN vote fraction threshold 0.9; 4.7% deferral)
captures 71.7% of errors and improves covered-set accuracy to 99.53%, while a stricter threshold (0.98; 8.9% deferral)
captures 87.0% of errors and yields 99.77% accuracy on the non-deferred set. This illustrates how neighborhood agreement
can be converted into an operator-facing "defer vs decide" policy with measurable trade-offs.

---

## 4. Discussion

Having established that whole-proteome ESM-2 embeddings recover taxonomic structure, yield high-purity clusters, and
provide actionable confidence signals (Figure 1), we now interpret these findings in the context of surveillance
requirements, position the approach relative to state-of-the-art methods, and outline the path toward deployment.

### 4.1 What Mean-Pooled Whole-Proteome Embeddings Encode

The strong low-dimensional structure and high species homophily indicate that whole-proteome, mean-pooled ESM-2 genome
embeddings capture dominant, taxonomically aligned signals—consistent with the view that PLM representations encode broad
evolutionary and functional patterns at the protein level [4,5]. For surveillance, this is operationally valuable:
taxonomy-aligned neighborhoods support rapid retrieval and coarse prioritization without alignment, while geometry-derived
diagnostics (homophily, purity, outliers) provide a principled way to expose uncertainty and flag boundary cases for
escalation. To be precise, our "context-free" claim applies *at the genome level*: ESM-2 is contextual within each
protein sequence, but mean pooling discards gene order, co-localization, and operon-scale context. These results also
align with evidence that medium-sized protein language models transfer well on realistic downstream tasks [17].

### 4.2 Positioning Relative to State of the Art

The landscape of computational genomics offers several alternative approaches to genome representation, each with
distinct trade-offs. Alignment-free sketching tools such as Mash have become foundational for large-scale genomic
indexing, providing very fast nucleotide-level similarity search [3]; containment-based variants like Mash Screen are
arguably closer production competitors for rapid detection in mixed samples [23]. Our approach is computationally
heavier at embedding time, but produces compact vectors that encode protein-level functional and structural signals
beyond raw nucleotide identity [5]. This trade-off becomes favorable in a cache-first setting where embeddings are
computed once and reused across downstream analytics and models.

The efficiency profile of our approach merits explicit consideration. At inference time, genome representation requires
per-protein embedding plus constant-time averaging, and once cached it enables fast CPU-feasible retrieval, scoring, and
re-analysis. Contextual genome language models that explicitly model gene order or gene–gene interactions must operate
over thousands of tokens per genome, with attention costs that can scale quadratically in sequence length. This does not
diminish the value of contextual models—rather, it clarifies why a cache-first mean-pooling baseline serves as a useful
operational substrate: it amortizes expensive embedding computation and provides geometry-aware diagnostics that can
determine when more expensive contextual modeling is warranted.

At the nucleotide level, foundation models such as DNABERT/DNABERT-2 [10,11], Nucleotide Transformer [9], HyenaDNA [12],
and GenSLMs [13] can in principle avoid gene calling and capture noncoding signals. However, these models face
long-context challenges on megabase-scale bacterial genomes [6] and often focus on specific organismal regimes (human
genomics [9,10,11,12], viruses [13]). Protein-level representations reduce sequence length and emphasize functional
units, offering a complementary route for microbial surveillance. Emerging long-range architectures partially address
these scaling limitations: GENA-LM provides long-sequence DNA foundation models [22], while state-space models with
biological inductive biases such as reverse-complement equivariance (e.g., Caduceus) aim for O(n) scaling [21]. These
approaches hold promise for future foodborne surveillance, but bacterial whole-genome benchmarks and operational
pipelines remain less standardized than in human genomics.

Among protein- and gene-level genome language models, several recent contributions define the current frontier. gLM
contextualizes ESM-2 protein embeddings at the operon/contig scale (~30 genes) to model co-regulation and function [7].
Protein Set Transformer aggregates proteomes as sets to build viral genome language models for viromics [8]. BacPT and
Bacformer extend this direction toward whole-proteome or ordered-genome context modeling in bacteria [18,19]. Our
mean-pooling baseline is deliberately simpler than these models, yet it serves a practical role: it delivers a scalable
cache and artifact pipeline for producing training and analysis data, exposes a geometry-aware diagnostic layer
(homophily, purity, outliers) that can operate before or alongside contextual models, and provides a clear baseline for
quantifying what additional modeling capacity is required to move beyond taxonomy toward within-species virulence
discrimination.

In terms of scale, our dataset (21,657 complete genomes) is substantial for a manuscript-level geometry study and
complements larger pretraining corpora used in foundational models (e.g., gLM's metagenomic contigs [7] and
bacteria-scale genome corpora in Bacformer [19]). Looking ahead, newer protein foundation models such as ESM3 [24]
suggest that protein-level representations will continue to improve, further strengthening the case for cache-first
reuse across surveillance pipelines.

### 4.3 Toward Operational Deployment: A Surveillance-Oriented Framework

Our long-term goal is to translate these embedding-space diagnostics into an end-to-end system for food safety triage:
a cache-first pipeline that embeds new assemblies once, then performs fast retrieval, scoring, and evidence-driven
escalation under operational constraints. The value of mean-pooled whole-proteome embeddings lies not only in their
compact representation, but in the principled, model-agnostic confidence signals their geometry provides.

In practice, homophily and cluster purity serve as deterministic confidence indicators. When a query genome falls in a
high-consistency neighborhood, retrieval-based triage can proceed as low-risk for coarse routing (e.g., selecting which
reference set to prioritize). When homophily degrades with neighborhood scale (Figure 5), the system can explicitly flag
these cases as boundary conditions and route them to deeper analysis—context-aware models, targeted evidence matching
against virulence/AMR gene databases, or expert review—rather than returning a brittle high-confidence label. The
within-genus risk–coverage curves (Figure 6) demonstrate how this geometry translates into an operator-facing "defer vs
decide" policy with quantifiable trade-offs.

Crucially, because our pathogenicity labels are priors, this escalation logic is fundamentally conservative: it is
designed to prevent overinterpretation when embedding neighborhoods become label-mixed, not to claim isolate-level
virulence resolution from a species-derived proxy.

Finally, strict outliers and distributional shifts in embedding space provide a complementary system-level signal for
quality control and novelty monitoring. In operational deployments, these genomes can be prioritized for
assembly/annotation checks, contamination screening, and drift tracking, and can trigger recalibration or retraining
when they accumulate over time.

### 4.4 Limitations

Several limitations bear on both scientific interpretation and any surveillance-oriented deployment.

First, our pathogenicity labels are **species/pathotype-derived priors**, not isolate-level phenotypes. Consequently,
the strong neighborhood agreement we observe largely reflects taxonomic coherence, and zero-shot transfer of the binary
label across held-out taxa is unreliable by construction. The within-genus stress tests should therefore be interpreted
as pathotype-level discriminability under known annotations, not as discovery of novel virulence factors in unlabeled
species.

Second, mean pooling discards protein order and neighborhood context—precisely the structure that encodes pathogenicity
islands, operons, and mobile element signatures. These are exactly the regimes where contextual genome language models
are expected to add value (e.g., gLM [7], Bacformer [19]).

Third, the representation inherits sensitivity to upstream annotation and proteome composition. Protein extraction
depends on gene calling/annotation, and unequal proteome sizes can bias a mean-pooled summary. Empirically, protein count
correlates with embedding “edge-case” signals in this dataset (e.g., Spearman ρ = 0.29 between protein count and
k-neighbor distance, and ρ ≈ −0.20 between protein count and label homophily), and strict outliers have larger median
protein counts than non-outliers (4,902 vs 4,403). These patterns are consistent with a mix of biological diversity and
assembly/annotation artifacts, and motivate explicit QC and missingness stress tests.

Fourth, our dataset is imbalanced (7,000 *Salmonella* vs 200 *C. freundii*; 35:1), which can bias centroid estimates,
neighbor mixing asymmetry, and density-based clustering toward abundant taxa. Finally, our cache configuration uses a
maximum protein sequence length (`max_prot_seq_len=1024`), so very long proteins may be truncated; the downstream impact
has not been quantified here.

We view these limitations as an honest boundary of what a mean-pooled baseline can claim, and as a roadmap for the next
iteration: robustness-to-incomplete-proteomes experiments (protein dropout), evidence integration (virulence/AMR gene
matching), and deeper outlier validation (e.g., completeness/contamination checks) before making stronger production
claims. More broadly, they motivate context-aware genome language models and evidence-driven pipelines as the next step.

### 4.5 Future Directions

Several follow-on analyses are feasible without additional embedding runs and would directly strengthen the path from
descriptive geometry to deployment readiness. One direction is to calibrate homophily and cluster purity into empirical
decision thresholds by mapping them to retrieval error rates (or downstream misclassification) under held-out splits. A
second is to stress-test cross-taxon generalization by training on a subset of taxa and evaluating on held-out taxa,
quantifying how much of the signal is genuinely transferable beyond taxonomy priors. Third, robustness-to-missingness
can be probed by simulating incomplete proteomes via protein downsampling (leveraging the per-protein cache for a
targeted subset) and measuring embedding stability and neighborhood drift. Finally, an interactive "hard-case" atlas of
low-homophily and strict-outlier genomes—paired with nearest neighbors and
metadata—would support human-in-the-loop quality control and create a concrete bridge between embedding diagnostics and
actionable surveillance workflows.

---

## 5. Conclusions

Whole-proteome ESM-2 embeddings provide a scalable, cache-friendly representation that strongly recovers microbial
taxonomy and yields the three actionable signals central to our surveillance framework (Figure 1): confidence scores
derived from neighborhood homophily, triage reliability via cluster purity, and early warnings through outlier detection.
At the same time, moderate agreement with a binary pathogenicity prior underscores the need for context-aware genome
language models to advance from taxonomy-aligned retrieval toward strain-level virulence discrimination and evidence
attribution [7,18,19].

For operational food safety surveillance, these results establish a concrete, deployment-aligned foundation: a reusable
embedding cache paired with geometry-derived confidence and novelty signals that can gate retrieval, prioritize evidence
extraction, and escalate boundary cases to richer models or expert review.

---

## Data and Code Availability

All code, analysis notebooks, and exported artifacts are available in the `foodguard/` directory of the Bacformer
repository: [https://github.com/jaygut/Bacformer/tree/main/foodguard](https://github.com/jaygut/Bacformer/tree/main/foodguard).

Key resources include:
- **Embedding bundle**: `foodguard/logs/genome_embeddings.npz`
- **Analysis notebook**: `notebooks/foodguard_cache_embedding_analysis.ipynb`
- **Validation script**: `scripts/foodguard_local_validations.py`
- **Figures and tables**: `foodguard/analysis/`

---

## References

1. World Health Organization (WHO). WHO estimates of the global burden of foodborne diseases (2015).
   https://www.who.int/data/gho/data/themes/who-estimates-of-the-global-burden-of-foodborne-diseases

2. Allard MW, Strain E, Melka D, et al. Practical value of food pathogen traceability through building a whole-genome
   sequencing network and database. *J Clin Microbiol* (2016). DOI: 10.1128/JCM.00081-16

3. Ondov BD, Treangen TJ, Melsted P, et al. Mash: fast genome and metagenome distance estimation using MinHash.
   *Genome Biol* (2016). DOI: 10.1186/s13059-016-0997-x

4. Rives A, Meier J, Sercu T, et al. Biological structure and function emerge from scaling unsupervised learning to 250
   million protein sequences. *PNAS* (2021). DOI: 10.1073/pnas.2016239118

5. Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model.
   *Science* (2023). DOI: 10.1126/science.ade2574

6. Benegas G, Ye C, Albors C, Canal Li J, Song YS. Genomic Language Models: Opportunities and Challenges. *arXiv*
   (2024). https://arxiv.org/abs/2407.11435

7. Hwang Y, Cornman AL, Kellogg EH, Ovchinnikov S, Girguis PR. Genomic language model predicts protein co-regulation and
   function. *Nat Commun* (2024). DOI: 10.1038/s41467-024-46947-9

8. Martin C, Gitter A, Anantharaman K. Protein Set Transformer: a protein-based genome language model to power
   high-diversity viromics. *Nat Commun* (2025). DOI: 10.1038/s41467-025-66049-4

9. Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. Nucleotide Transformer: building and evaluating robust
   foundation models for human genomics. *Nat Methods* (2024–2025). DOI: 10.1038/s41592-024-02523-z

10. Ji Y, Zhou Z, Liu H, Davuluri RV. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model
    for DNA-language in genome. *Bioinformatics* (2021). DOI: 10.1093/bioinformatics/btab083

11. Zhou Z, Ji Y, Li W, Dutta P, Davuluri RV, Liu H. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species
    Genomes. *arXiv* (2023). https://arxiv.org/abs/2306.15006

12. Nguyen E, Poli M, Faizi M, et al. HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution.
    *arXiv* / NeurIPS (2023). https://arxiv.org/abs/2306.15794

13. Zvyagin M, Brace A, Hippe K, et al. GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics.
    *bioRxiv* (2022). DOI: 10.1101/2022.10.10.511571

14. McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv*
    (2018). https://arxiv.org/abs/1802.03426

15. McInnes L, Healy J, Astels S. hdbscan: Hierarchical density based clustering. *J Open Source Software* (2017).
    DOI: 10.21105/joss.00205

16. Hennig C. Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* (2007).

17. Vieira LC, Handojo ML, Wilke CO. Medium-sized protein language models perform well at transfer learning on realistic
    datasets. *bioRxiv* (2025). DOI: 10.1101/2024.11.22.624936

18. Sethi P, Chevrette MG, Zhou J. Learning gene interactions and functional landscapes from entire bacterial proteomes.
    *bioRxiv* (2025). DOI: 10.1101/2025.03.19.644232

19. Wiatrak M, Viñas Torné R, Ntemourtsidou M, et al. A contextualised protein language model reveals the functional
    syntax of bacterial evolution. *bioRxiv* (2025). DOI: 10.1101/2025.07.20.665723

20. van der Maaten L, Hinton G. Visualizing Data using t-SNE. *J Mach Learn Res* (2008).
    https://www.jmlr.org/papers/v9/vandermaaten08a.html

21. Schiff Y, Kao C-H, Gokaslan A, Dao T, Gu A, Kuleshov V. Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence
    Modeling. *arXiv* (2024). https://arxiv.org/abs/2403.03234

22. Fishman VS, Kuratov Y, Shmelev A, et al. GENA-LM: a family of open-source foundational DNA language models for long
    sequences. *Nucleic Acids Res* (2025). DOI: 10.1093/nar/gkae1310

23. Ondov BD, Starrett GJ, Sappington A, et al. Mash Screen: high-throughput sequence containment estimation for genome
    discovery. *Genome Biol* (2019). DOI: 10.1186/s13059-019-1841-x

24. Hayes T, Rao R, Akin H, et al. Simulating 500 million years of evolution with a language model. *Science* (2025).
    DOI: 10.1126/science.ads0018

---

## Figures (Draft)

**Figure 1.** Conceptual overview of the embedding-based surveillance framework.

![Figure 1: Framework overview](../analysis/framework_infographic.png)

*Figure 1 legend.* Schematic of the embedding-based triage framework for foodborne bacterial surveillance. The pipeline addresses the surveillance bottleneck where 600 million annual foodborne illnesses demand rapid response that traditional alignment-based workflows cannot deliver. A foundation of 21,657 GenomeTrakr-derived genomes from nine critical taxa (*Salmonella*, *Listeria*, *E. coli*, and others) is processed through ESM-2, which converts each genome's proteins into a compact 480-dimensional vector "fingerprint" (Step 1). In the resulting embedding space, genomes from the same species form distinct, high-purity clusters that reveal taxonomic geometry (Step 2). From this geometry, three actionable signals are extracted (Step 3): confidence scores (homophily), triage reliability (cluster purity), and early warnings (outliers). The framework achieves high accuracy in stress tests discriminating dangerous pathogens from their benign relatives (98.35% for *E. coli* O157:H7 vs non-pathogenic *E. coli*; 99.88% for *L. monocytogenes* vs *L. innocua*), demonstrating operational utility for rapid, alignment-free pathogen triage.

**Figure 2.** PCA overview of whole-proteome genome embeddings (explained variance; PC1 vs PC2 colored by species and
pathogenicity prior).

[Interactive HTML (PCA by species)](../analysis/pca_by_species.html)

[Interactive HTML (PCA by pathogenicity)](../analysis/pca_by_pathogenicity.html)

![Figure 2: PCA overview](../analysis/manuscript_figure_pca_overview.png)

*Figure 2 legend.* Whole-proteome ESM-2 genome embeddings projected with PCA after standardizing embedding dimensions.
(A) Explained variance ratio for the first 10 principal components (bars) with cumulative variance (line); PC1 explains
76.56% of variance, PC2 explains 10.95%, and 3 PCs exceed 90% cumulative variance (94.3%). (B) PC1 vs PC2 for all
genomes (n=21,657), colored by species. (C) Same PC1 vs PC2 projection colored by the pathogenicity prior used in this
study (species/pathotype-derived). Each point represents one genome embedding obtained by mean-pooling per-protein ESM-2
representations (480 dimensions) computed from GenBank-annotated protein sequences.

**Figure 3.** UMAP and t-SNE projections by pathogenicity prior and species.

[Interactive HTML](../analysis/manuscript_figure_umap_tsne_panel.html)

![Figure 3: UMAP/t-SNE panel](../analysis/manuscript_figure_umap_tsne_panel.png)

*Figure 3 legend.* Non-linear 2D projections of whole-proteome genome embeddings, computed after standardization and PCA
pre-reduction (50 components). Top row: UMAP [14] (left) and t-SNE [20] (right) colored by the pathogenicity prior
(species/pathotype-derived). Bottom row: the same UMAP and t-SNE projections colored by species (nine taxa). Across both
methods, genomes form largely species-coherent islands, while the pathogenicity prior aligns with these islands due to
phylogenetic confounding; mixed regions highlight closely related taxa with different priors and motivate confidence-
aware retrieval and context-aware modeling. Axes are arbitrary, and global distances/layout are not interpreted as
metric-preserving.

**Figure 4.** Species centroid distance heatmap (Euclidean distance in standardized embedding space).

[Interactive HTML](../analysis/species_centroid_heatmap.html)

![Figure 4: Species centroid distances](../analysis/species_centroid_heatmap.png)

*Figure 4 legend.* Pairwise Euclidean distances between species centroids in the standardized whole-proteome embedding
space. Each species centroid is the mean of its z-scored genome embeddings (mean-pooled 480-dimensional ESM-2
representations), and each cell reports the L2 distance between two species centroids (diagonal = 0). Smaller distances
indicate taxa whose *average* embeddings occupy nearby regions of the space and are therefore more likely to yield
near-boundary neighbors under kNN retrieval; larger distances reflect broad taxonomic separation. Because centroids
compress within-species heterogeneity, the heatmap is interpreted as a coarse summary of global geometry rather than as
a distribution-aware or phylogenetically calibrated distance.

**Figure 5.** Multi-scale homophily of the pathogenicity prior (mean agreement and mixed-neighborhood fraction vs k).

![Figure 5: Homophily vs k](../analysis/homophily_vs_k.png)

*Figure 5 legend.* Multi-scale neighborhood agreement for the pathogenicity prior (species/pathotype-derived). For each
genome, pathogenicity homophily is defined as the fraction of its k nearest neighbors (Euclidean distance in the
standardized 480-dimensional embedding space) that share the same prior label. Top: mean homophily vs k for all genomes
and selected taxa; the y-axis is intentionally zoomed to reveal sub-ceiling differences. Bottom: fraction of genomes with
mixed neighborhoods, operationally defined here as homophily < 0.9 (a high threshold chosen to highlight early mixing),
as k increases. Because the prior is partially confounded with taxonomy, high homophily primarily reflects taxonomic
coherence of the embedding space; the informative signal is the subset of genomes whose local neighborhoods become
label-mixed as the neighborhood radius expands, highlighting boundary cases where confidence-aware triage is warranted.

**Figure 6.** Within-genus confidence calibration (risk–coverage tradeoff) for *E. coli* O157:H7 vs non-pathogenic
*E. coli*.

![Figure 6: E. coli risk-coverage](../analysis/ecoli_risk_coverage.png)

*Figure 6 legend.* Risk–coverage tradeoff for within-genus kNN classification of *E. coli* O157:H7 vs non-pathogenic
*E. coli* in the standardized 480-dimensional embedding space (k=20), evaluated with 5-fold stratified cross-validation.
Each point corresponds to a vote-fraction threshold on the predicted class; genomes below the threshold are deferred.
Coverage is the fraction not deferred, and risk is the error rate on the non-deferred set. This converts neighborhood
agreement into an operator-facing “defer vs decide” policy with measurable trade-offs.

**Figure 7.** Validating centroid distances as confusability priors.

![Figure 7: Centroid distance vs neighbor mixing](../analysis/centroid_distance_vs_neighbor_mix.png)

*Figure 7 legend.* Relationship between inter-species centroid distance (Euclidean in standardized embedding space) and
symmetric cross-species neighbor mixing (k=20), computed across all species pairs. Each point represents one taxon pair.
Lower centroid distances correspond to higher mixing, supporting centroid distance as a coarse confusability prior
(Spearman ρ = −0.65; p = 1.5×10⁻⁵). Mixing is symmetrized to reduce the effect of sampling imbalance on directional
neighbor rates.

---

*Corresponding authors: Jay Gutierrez (jg@graphoflife.com); Javier Correa Alvarez, PhD (jcorre38@eafit.edu.co)*
