# ESM-2 Protein Language Model Embeddings Capture Pathogenicity Signals in Bacterial Genomes: Implications for AI-Driven Food Safety Surveillance

**Jay Gutierrez** & **Javier Correa Alvarez**

---

## Abstract

Rapid pathogen identification is critical for food safety surveillance. We evaluated whether ESM-2 protein language model embeddings capture biologically meaningful signals that discriminate pathogenic from non-pathogenic bacterial genomes. Analyzing 21,657 genomes across nine foodborne-relevant species, we demonstrate that mean-pooled genome embeddings exhibit exceptional neighborhood homophily (0.993), robust species-level clustering (silhouette = 0.555), and significant pathogenicity separation (Cohen's d = 7.52). Density-based clustering identified 34 clusters, with 97% achieving >90% pathogenicity purity. These findings establish that pre-trained protein language models can serve as effective feature extractors for embedding-based pathogen classification, with potential applications in rapid, scalable food safety monitoring.

**Keywords:** protein language model, ESM-2, foodborne pathogens, genome embedding, machine learning, food safety surveillance

---

## 1. Introduction

Foodborne pathogens cause approximately 600 million illnesses and 420,000 deaths annually (WHO, 2015). Traditional surveillance using pulsed-field gel electrophoresis has largely been replaced by whole genome sequencing (WGS), enabling higher-resolution outbreak detection and source tracking. The GenomeTrakr network demonstrates that each 1,000 additional sequenced isolates associates with approximately six fewer illnesses per pathogen annually (Pires et al., 2021).

However, WGS pipelines remain computationally intensive, requiring alignment-based phylogenetic analysis. Protein language models (PLMs) offer an alternative paradigm. ESM-2, trained on billions of protein sequences, learns contextual representations capturing evolutionary, structural, and functional features without explicit alignment (Lin et al., 2023). Transfer learning from PLMs has achieved state-of-the-art performance in antimicrobial peptide classification (Saini et al., 2025) and bacterial effector prediction (Wang et al., 2025).

We hypothesized that genome-level embeddings derived from mean-pooling per-protein ESM-2 representations would: (H1) separate pathogenic from non-pathogenic genomes; (H2) preserve species-level phylogenetic structure; and (H3) exhibit neighborhood homophily predictive of classification reliability. This study evaluates these hypotheses to assess the potential of PLM embeddings for rapid, embedding-based pathogen risk assessment.

---

## 2. Materials and Methods

### 2.1 Dataset

We analyzed 21,657 bacterial genomes from nine species: *Salmonella enterica* (n=6,849), *Listeria monocytogenes* (n=4,502), *Escherichia coli* O157:H7 (n=1,342), non-pathogenic *E. coli* (n=4,437), *Bacillus subtilis* (n=2,358), *Citrobacter freundii* (n=195), *Citrobacter koseri* (n=871), *Escherichia fergusonii* (n=324), and *Listeria innocua* (n=79). Pathogenicity labels were assigned based on species-level epidemiological evidence: 13,000 pathogenic (60%) and 8,657 non-pathogenic (40%). Assemblies originated from the GenomeTrakr database.

### 2.2 Embedding Generation and Caching

Protein sequences extracted from GenBank annotations were embedded using ESM-2 (facebook/esm2_t12_35M_UR50D) with PyTorch 2.2.2+cu121, producing 480-dimensional per-protein vectors. Embeddings were computed in a sharded fashion (2-way) on Apolo-3 (node `a3-accel-0`, dual NVIDIA H100 NVL, driver 575/CUDA 12.9). Each genome’s per-protein embeddings were stored in cache files (`prot_emb_<key>.pt`) keyed by sequence content, model ID, and `max_prot_seq_len=1024` to guarantee reproducible hits across pipeline components. Genome-level representations were obtained via mean pooling across all proteins per genome, following genome-level representation practices (Flamholz et al., 2024). A pooled embedding bundle (`genome_embeddings.npz`) containing genome vectors and metadata (genome_id, species, pathogenicity, protein counts) was generated for downstream analytics and visualization.

### 2.3 Dimensionality Reduction and Clustering

Embeddings were standardized and reduced using PCA (50 components), UMAP (n_neighbors=15, min_dist=0.1), and t-SNE (perplexity=30). HDBSCAN clustering (min_cluster_size=50, min_samples=10) was applied to the first 20 principal components. Internal validation used Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index; external validation against pathogenicity labels used Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Bootstrap stability (n=50, 80% subsampling) was assessed via Jaccard similarity.

### 2.4 Homophily and Statistical Analysis

K-nearest neighbor (k=20) homophily was computed as the fraction of neighbors sharing pathogenicity or species labels. Effect sizes (Cohen's d) were computed for each principal component. Multivariate effect size was estimated as the Euclidean centroid distance divided by pooled standard deviation. Cluster enrichment used chi-square tests with Benjamini-Hochberg FDR correction.

### 2.5 Computational Environment

All large-scale embedding and caching runs were executed on the EAFIT Apolo-3 HPC cluster (Rocky Linux 9.5) with dual NVIDIA H100 NVL GPUs (driver 575.57.08, CUDA 12.9). Sharded jobs used 4 vCPUs and 32 GB RAM per GPU. Interactive analytics (PCA/UMAP/t-SNE dashboard) were run on CPU nodes (`bigmem`) with pooled embeddings precomputed from cache. SLURM was used for job orchestration; environments were managed via Conda (`FoodGuard310`, Python 3.10). External dependencies included PyTorch, transformers, umap-learn, scikit-learn, and HDBSCAN.

---

## 3. Results

### 3.1 Embedding Space Preserves Phylogenetic Structure (H2 Supported)

UMAP and t-SNE projections revealed clear species-level clustering (Figure 1). The nine species formed distinct, well-separated islands, confirming ESM-2's capacity to capture phylogenetic relationships through proteome-wide patterns. Mean species homophily was 0.992, with 99.2% of nearest neighbors belonging to the same species.

HDBSCAN identified 34 clusters with 35.6% noise points—typical for density-based clustering on biological data. Internal validation yielded strong metrics: Silhouette = 0.555, Calinski-Harabasz = 10,850, Davies-Bouldin = 1.07. Bootstrap analysis confirmed 76% of clusters achieved Jaccard stability >0.75 (highly stable).

### 3.2 Pathogenicity Signals Are Captured but Confounded by Phylogeny (H1 Partially Supported)

Pathogenicity-colored projections showed substantial but imperfect separation. The inter-class/intra-class distance ratio was 1.11, with multivariate Cohen's d = **7.52** (large effect), confirming well-separated class centroids.

External validation metrics were moderate: ARI = 0.22, NMI = 0.34. This reflects biological reality—pathogenicity is largely species-determined (*Salmonella* uniformly pathogenic; *B. subtilis* uniformly non-pathogenic). Critically, **33/34 clusters (97%) achieved >90% pathogenicity purity**, enabling confident classification based on cluster membership despite phylogenetic confounding.

### 3.3 Homophily Predicts Classification Reliability (H3 Supported)

Mean pathogenicity homophily was **0.993**, indicating 99.3% of nearest neighbors share the query's pathogenicity label. This supports H3: high-homophily genomes can be classified confidently via neighbor voting, while rare low-homophily cases (n=37, <0.5) represent boundary cases requiring scrutiny.

Low-homophily genomes concentrated among *Listeria* species, where pathogenic *L. monocytogenes* and non-pathogenic *L. innocua* share substantial proteome similarity, consistent with their close phylogenetic relationship.

### 3.4 Discriminative Embedding Dimensions

Principal components 4 and 5 showed the largest pathogenicity separation (|d| = 1.36, 1.12), representing "pathogenicity axes" where classes diverge most strongly.

### 3.5 Outlier Detection

Strict criteria (distance outlier AND noise) identified 252 genomes (1.2%) as high-confidence anomalies. These showed lower homophily (0.89 vs 0.99), consistent with boundary cases or quality issues warranting review.

---

## 4. Discussion

### 4.1 PLM Embeddings Enable Rapid Pathogen Classification

Our results demonstrate that ESM-2 genome embeddings capture signals relevant to pathogen detection without alignment. The exceptional homophily (0.993) and cluster purity (97%) suggest that embedding-based classification—using k-NN voting on pre-computed vectors—could achieve high accuracy with minimal computational overhead. This aligns with evidence that medium-sized PLMs perform well at transfer learning (Teufel et al., 2025).

The species centroid heatmap reveals expected phylogenetic structure: *E. coli* variants cluster tightly (distances 13–18), while *Listeria* species show low inter-species distance (4.04), explaining observed boundary cases.

### 4.2 Limitations

The moderate ARI (0.22) reflects phylogenetic confounding inherent to species-level pathogenicity labels. Future work should incorporate within-species virulence variation (e.g., *Salmonella* serovars) to assess strain-level discrimination.

The 35.6% noise rate suggests room for clustering optimization, though noise points exhibited genuinely lower homophily rather than representing algorithmic artifacts.

### 4.3 Toward Real-Time Food Safety Monitoring

These findings have implications for next-generation surveillance architectures:

**Embedding-based triage.** Pre-computed genome embeddings enable O(1) lookup for cached genomes and O(n) similarity search for novel isolates—orders of magnitude faster than alignment-based pipelines. This could enable rapid screening to prioritize samples for detailed phylogenetic analysis.

**Confidence calibration.** Homophily scores provide a principled basis for prediction confidence. Low-homophily cases (<0.8) could trigger additional evidence integration (virulence factor databases) or expert review.

**Anomaly surveillance.** Distance-based outliers represent candidates for enhanced scrutiny as potential novel variants or emerging threats—critical for early warning systems.

**Quality control.** Low-homophily genomes in phylogenetically close species pairs warrant label verification, improving training data quality for downstream models.

While alignment-based WGS remains the gold standard for outbreak investigation, embedding-based approaches could complement existing workflows by providing rapid initial risk stratification, potentially accelerating response times in food safety surveillance contexts.

---

## 5. Conclusions

ESM-2 protein language model embeddings capture species-level phylogenetic structure and pathogenicity signals in bacterial genomes. Exceptional homophily (0.993) and cluster purity (97%) support embedding-based classification for pathogen risk assessment. Homophily scores enable confidence calibration, while outlier detection identifies surveillance candidates. These findings advance the application of foundation models to food safety genomics and suggest potential for integration into rapid, scalable monitoring systems.

---

## Data Availability

Analysis code available at: https://github.com/macwiatrak/bacformer

---

## References

1. Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123-1130.

2. Pires SM, et al. (2021). An economic evaluation of the Whole Genome Sequencing source tracking program in the U.S. *PLOS ONE* 16:e0258262.

3. McInnes L, Healy J, Astels S (2017). HDBSCAN: Hierarchical density based clustering. *JOSS* 2:205.

4. Hennig C (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data Analysis* 52:258-271.

5. Rives A, et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118:e2016239118.

6. Armstrong G, et al. (2022). UMAP Reveals Composite Patterns and Resolves Visualization Artifacts in Microbiome Data. *mSystems* 7:e00691-21.

7. Saini P, et al. (2025). Transfer learning on protein language models improves antimicrobial peptide classification. *Scientific Reports* 15:21223.

8. Wang X, et al. (2025). Contrastive-learning of language embedding and biological features for effector prediction. *Nature Communications* 16:56526.

9. Teufel F, et al. (2025). Medium-sized protein language models perform well at transfer learning. *Scientific Reports* 15:5674.

10. Flamholz ZN, et al. (2024). Protein Set Transformer: A protein-based genome language model. *bioRxiv* 2024.07.26.605391.

---

**Figures:**
- Figure 1: UMAP and t-SNE projections by pathogenicity and species (2×2 panel)
- Figure 2: Pathogenicity homophily distribution and spatial mapping
- Figure 3: Intra/inter-class distance distributions with silhouette scores
- Figure 4: Species centroid distance heatmap
- Figure 5: Dataset overview (class balance, species distribution, protein counts)

---

*Corresponding author: Javier Correa Alvarez, PhD (jcorre38@eafit.edu.co)*
