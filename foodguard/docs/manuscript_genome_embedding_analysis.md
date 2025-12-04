# ESM-2 Protein Language Model Embeddings Capture Pathogenicity Signals in Bacterial Genomes: Implications for AI-Driven Food Safety Surveillance

**Jay Gutierrez** & **Javier Correa Alvarez**

---

## Abstract

Rapid pathogen identification is fundamental to food safety surveillance, yet current whole-genome sequencing pipelines remain computationally intensive. We investigated whether protein language model embeddings provide discriminative, alignment-free features for separating pathogenic from non-pathogenic bacterial genomes. Analyzing 21,657 genomes across nine foodborne-relevant species, we find that mean-pooled ESM-2 embeddings exhibit high neighborhood homophily (0.993), robust species-level clustering (silhouette = 0.555), and substantial pathogenicity separation (Cohen's d = 7.52). Density-based clustering yielded 34 groups with 97% exceeding 90% pathogenicity purity, though moderate agreement with pathogenicity labels (ARI = 0.22) reflects inherent phylogenetic confounding. These findings establish that pre-trained protein language models encode biologically coherent structure suitable for embedding-based pathogen triage, while highlighting the need for context-aware architectures to disentangle species identity from virulence. This work lays the foundation for integrating genomic language models into scalable food safety monitoring systems.

**Keywords:** protein language model, ESM-2, foodborne pathogens, genome embedding, machine learning, food safety surveillance

---

## 1. Introduction

Foodborne pathogens cause approximately 600 million illnesses and 420,000 deaths annually (WHO, 2015). Traditional surveillance using pulsed-field gel electrophoresis has been superseded by whole genome sequencing (WGS), enabling higher-resolution outbreak detection. The GenomeTrakr network demonstrates that expanded sequencing capacity reduces illness burden—each 1,000 additional isolates associates with approximately six fewer cases per pathogen annually (Pires et al., 2021).

However, WGS pipelines remain computationally demanding, requiring alignment-based phylogenetic reconstruction that scales poorly to real-time applications. Protein language models (PLMs) offer an alternative: ESM-2, trained on billions of protein sequences, learns contextual representations capturing evolutionary, structural, and functional features without explicit alignment (Lin et al., 2023). Transfer learning from PLMs has achieved strong performance in antimicrobial peptide classification (Saini et al., 2025) and bacterial effector prediction (Wang et al., 2025), suggesting broader applicability to microbial genomics.

In this work, we hypothesized that genome-level embeddings derived from mean-pooling per-protein ESM-2 representations would: (H1) separate pathogenic from non-pathogenic genomes; (H2) preserve species-level phylogenetic structure; and (H3) exhibit neighborhood homophily predictive of classification reliability. Evaluating these hypotheses informs whether PLM embeddings can support rapid, embedding-based pathogen risk stratification as a complement to traditional WGS workflows.

---

## 2. Materials and Methods

### 2.1 Dataset

We analyzed 21,657 bacterial genomes from nine species commonly implicated in foodborne illness or serving as non-pathogenic controls: *Salmonella enterica* (n=6,849), *Listeria monocytogenes* (n=4,502), *Escherichia coli* O157:H7 (n=1,342), non-pathogenic *E. coli* (n=4,437), *Bacillus subtilis* (n=2,358), *Citrobacter freundii* (n=195), *Citrobacter koseri* (n=871), *Escherichia fergusonii* (n=324), and *Listeria innocua* (n=79). Pathogenicity labels were assigned based on species-level epidemiological evidence: 13,000 pathogenic (60%) and 8,657 non-pathogenic (40%). Assemblies originated from the GenomeTrakr database.

### 2.2 Embedding Generation and Caching

Protein sequences extracted from GenBank annotations were embedded using ESM-2 (`facebook/esm2_t12_35M_UR50D`) with PyTorch 2.2.2+cu121, producing 480-dimensional per-protein vectors. Embeddings were computed via 2-way sharding on dual NVIDIA H100 NVL GPUs (CUDA 12.9). Per-protein embeddings were cached in content-addressed files (`prot_emb_<key>.pt`) keyed by sequence hash, model ID, and `max_prot_seq_len=1024` to ensure reproducibility. Genome-level representations were obtained via mean pooling across all proteins, following established genome-level practices (Flamholz et al., 2024). A consolidated bundle (`genome_embeddings.npz`) containing vectors and metadata (genome_id, species, pathogenicity, protein counts) was generated for analysis.

### 2.3 Dimensionality Reduction and Clustering

Embeddings were standardized and reduced using PCA (50 components), UMAP (n_neighbors=15, min_dist=0.1), and t-SNE (perplexity=30). HDBSCAN (min_cluster_size=50, min_samples=10) was applied to the first 20 principal components. Internal validation used Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index; external validation against pathogenicity labels used Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Bootstrap stability (n=50, 80% subsampling) was assessed via Jaccard similarity following Hennig (2007).

### 2.4 Homophily and Statistical Analysis

K-nearest neighbor (k=20) homophily was computed as the fraction of neighbors sharing pathogenicity or species labels. Effect sizes (Cohen's d) were computed for each principal component between pathogenic and non-pathogenic groups. Multivariate effect size was estimated as Euclidean centroid distance divided by pooled standard deviation. Cluster enrichment used chi-square tests with Benjamini-Hochberg FDR correction.

### 2.5 Computational Environment

Large-scale embedding runs executed on the EAFIT Apolo-3 HPC cluster (Rocky Linux 9.5) with dual NVIDIA H100 NVL GPUs. Sharded jobs used 4 vCPUs and 32 GB RAM per GPU. Interactive analytics ran on CPU nodes with precomputed pooled embeddings. SLURM orchestrated jobs; environments were managed via Conda (Python 3.10) with PyTorch, transformers, umap-learn, scikit-learn, and HDBSCAN.

---

## 3. Results

### 3.1 Embedding Space Preserves Phylogenetic Structure (H2 Supported)

UMAP and t-SNE projections revealed unambiguous species-level clustering (Figure 1). The nine species formed distinct, well-separated islands, confirming ESM-2's capacity to capture phylogenetic relationships through proteome-wide patterns. Mean species homophily was 0.992—99.2% of nearest neighbors belonged to the same species.

HDBSCAN identified 34 clusters with 35.6% noise points, typical for density-based methods on biological data where not all samples reside in dense regions. Internal validation yielded strong metrics: Silhouette = 0.555, Calinski-Harabasz = 10,850, Davies-Bouldin = 1.07. Bootstrap analysis confirmed 76% of clusters achieved Jaccard stability >0.75.

### 3.2 Pathogenicity Separation Is Substantial but Phylogenetically Confounded (H1 Partially Supported)

Pathogenicity-colored projections showed substantial but imperfect separation. The inter-class/intra-class distance ratio was 1.11, with multivariate Cohen's d = **7.52** (large effect), confirming well-separated class centroids in embedding space.

External validation metrics were moderate: ARI = 0.22, NMI = 0.34. This reflects biological reality: pathogenicity in our dataset is largely species-determined (*Salmonella* uniformly pathogenic; *B. subtilis* uniformly non-pathogenic), so clustering recovers taxonomy rather than the binary label. Critically, **33/34 clusters (97%) achieved >90% pathogenicity purity**, enabling confident risk stratification via cluster membership despite this confounding.

### 3.3 Homophily Predicts Classification Reliability (H3 Supported)

Mean pathogenicity homophily was **0.993**—99.3% of nearest neighbors share the query's pathogenicity label. High-homophily genomes can be classified confidently via neighbor voting; rare low-homophily cases (n=37 with homophily <0.5) represent boundary cases warranting scrutiny.

Low-homophily genomes concentrated among *Listeria* species, where pathogenic *L. monocytogenes* and non-pathogenic *L. innocua* share substantial proteome similarity, consistent with their close phylogenetic relationship.

### 3.4 Discriminative Embedding Dimensions

Principal components 4 and 5 showed largest pathogenicity separation (|d| = 1.36, 1.12), representing axes where classes diverge most strongly.

### 3.5 Outlier Detection

Strict criteria (distance outlier AND cluster noise) identified 252 genomes (1.2%) as high-confidence anomalies with lower homophily (0.89 vs 0.99), consistent with boundary cases or quality issues.

---

## 4. Discussion

### 4.1 PLM Embeddings Provide Efficient, Biologically Coherent Features

ESM-2 genome embeddings capture taxonomically meaningful structure without alignment. The exceptional homophily (0.993) and cluster purity (97%) demonstrate that embedding-based classification—e.g., k-NN on pre-computed vectors—can achieve practical utility with minimal computational overhead. This aligns with evidence that medium-sized PLMs transfer effectively to downstream tasks (Teufel et al., 2025).

The species centroid heatmap confirms expected phylogenetic structure: *E. coli* variants cluster tightly (distances 13–18), while *Listeria* species show lower inter-species distances (4.04), explaining observed boundary cases. The moderate external validation (ARI = 0.22) is not a failure of the embeddings but rather reflects the inherent correlation between species identity and pathogenicity in this dataset—a biological confounder rather than a technical limitation.

### 4.2 Limitations and the Need for Context-Aware Models

Several limitations warrant consideration. First, species-level pathogenicity labels preclude assessment of within-species virulence variation; future work should incorporate serovar-level annotations (e.g., distinct *Salmonella* serovars with varying pathogenicity) to evaluate strain-level discrimination.

Second, the 35.6% noise rate suggests density-based clustering may not optimally partition this embedding space, though noise points exhibited genuinely lower homophily rather than representing algorithmic artifacts.

Third, and most fundamentally, mean pooling discards genomic organization—the order and context of proteins along the chromosome. Virulence factors often cluster in pathogenicity islands or operons; capturing this positional information may enhance discrimination between pathogenic and commensal strains within the same species.

### 4.3 Toward Genomic Context: The Bacformer Architecture

To address the limitation of context-free pooling, we are leveraging the **Bacformer**, a prokaryotic foundation model that represents whole bacterial genomes as ordered sequences of protein embeddings. Unlike mean pooling, Bacformer applies transformer attention over proteins arranged by genomic coordinate, enabling the model to learn positional dependencies—such as operon structure, pathogenicity island organization, and syntenic conservation—that distinguish closely related strains.

Bacformer was trained on approximately 1.3 million bacterial genomes comprising over 3 billion proteins. It uses ESM-2 embeddings as input tokens and adds learnable positional encodings to capture genomic context. The architecture supports masked language modeling (predicting masked proteins from context) and downstream fine-tuning for tasks including pathogenicity classification, antimicrobial resistance prediction, and strain typing. By contextualizing protein embeddings within their genomic neighborhood, Bacformer aims to disentangle species-level signal from strain-specific virulence features—precisely the limitation observed in our current mean-pooling approach.

Preliminary results suggest that contextualized genome embeddings improve discrimination in phylogenetically challenging cases (e.g., *Listeria* species pairs), though full evaluation is ongoing. Integration of Bacformer embeddings into the surveillance framework described here represents a natural next step.

### 4.4 Implications for Real-Time Surveillance

These findings inform next-generation surveillance architectures:

**Embedding-based triage.** Pre-computed genome embeddings enable O(1) lookup for cached genomes and O(n) similarity search for novel isolates—orders of magnitude faster than alignment-based pipelines—supporting rapid screening to prioritize samples for detailed phylogenetic analysis.

**Confidence calibration.** Homophily scores provide a principled basis for prediction confidence. Low-homophily cases (<0.8) could trigger additional evidence integration (e.g., virulence factor databases) or expert review.

**Anomaly surveillance.** Distance-based outliers represent candidates for enhanced scrutiny as potential novel variants or emerging threats—critical for early warning systems.

While alignment-based WGS remains essential for outbreak investigation and regulatory action, embedding-based approaches can complement existing workflows by providing rapid initial risk stratification, potentially accelerating response in time-critical food safety contexts.

---

## 5. Conclusions

ESM-2 protein language model embeddings capture species-level phylogenetic structure and encode pathogenicity-relevant signals in bacterial genomes. High homophily (0.993) and cluster purity (97%) support embedding-based classification for pathogen risk triage, while moderate agreement with pathogenicity labels highlights the need for context-aware architectures to resolve within-species variation. The Bacformer model, currently under development, addresses this gap by incorporating genomic positional context. Together, these approaches advance the application of foundation models to food safety genomics and suggest potential for integration into rapid, scalable monitoring systems.

---

## Data Availability

Genome embeddings, analysis notebooks, and preprocessing pipelines are available at: https://github.com/jaygut/bacformer-foodguard

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
