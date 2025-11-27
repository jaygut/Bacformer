#!/usr/bin/env python3
"""
FoodGuardAI Genome Processor

A high-performance Python script that combines genome statistics analysis and 
unified manifest creation using efficient bioinformatics libraries.

Features:
- Fast GBFF parsing using BioPython
- Comprehensive genome statistics extraction
- Unified manifest generation with proper labeling
- Parallel processing for improved performance
- Detailed progress tracking and validation

Author: FoodGuardAI Team
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
from datetime import datetime

# Bioinformatics libraries
try:
    from Bio import SeqIO
    from Bio.SeqUtils import GC
    from Bio.SeqRecord import SeqRecord
except ImportError:
    print("ERROR: BioPython not installed. Install with: pip install biopython")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('foodguard_genome_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenomeProcessor:
    """High-performance genome analysis and manifest generation."""
    
    def __init__(self, data_dir: str, output_dir: str, n_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        
        # Define genome collections with metadata
        self.collections = {
            "salmonella_gbff_selected": {
                "pathogenicity": "pathogenic",
                "species": "Salmonella_enterica",
                "taxon_id": 590
            },
            "ecoli_o157h7_gbff_selected": {
                "pathogenicity": "pathogenic", 
                "species": "E_coli_O157H7",
                "taxon_id": 83334
            },
            "listeria_monocytogenes_gbff_selected": {
                "pathogenicity": "pathogenic",
                "species": "L_monocytogenes", 
                "taxon_id": 1639
            },
            "listeria_innocua_gbff_extracted": {
                "pathogenicity": "non_pathogenic",
                "species": "L_innocua",
                "taxon_id": 1640
            },
            "bacillus_subtilis_gbff_extracted": {
                "pathogenicity": "non_pathogenic",
                "species": "B_subtilis",
                "taxon_id": 1423
            },
            "citrobacter_freundii_gbff_extracted": {
                "pathogenicity": "non_pathogenic",
                "species": "C_freundii",
                "taxon_id": 546
            },
            "citrobacter_koseri_gbff_extracted": {
                "pathogenicity": "non_pathogenic",
                "species": "C_koseri",
                "taxon_id": 547
            },
            "escherichia_fergusonii_gbff_extracted": {
                "pathogenicity": "non_pathogenic",
                "species": "E_fergusonii",
                "taxon_id": 564
            }
        }
        
        # Check for filtered E. coli dataset
        filtered_ecoli = self.data_dir / "ecoli_nonpathogenic_gbff_filtered"
        if filtered_ecoli.exists():
            self.collections["ecoli_nonpathogenic_gbff_filtered"] = {
                "pathogenicity": "non_pathogenic",
                "species": "E_coli_nonpathogenic", 
                "taxon_id": 562
            }
            # Remove unfiltered version
            if "ecoli_all_strains_gbff_extracted" in self.collections:
                del self.collections["ecoli_all_strains_gbff_extracted"]
            logger.info("Using filtered E. coli dataset (O157:H7 strains removed)")
        else:
            self.collections["ecoli_all_strains_gbff_extracted"] = {
                "pathogenicity": "non_pathogenic",
                "species": "E_coli_nonpathogenic",
                "taxon_id": 562
            }

    def find_gbff_files(self) -> List[Tuple[str, Dict]]:
        """Find all GBFF files across collections."""
        gbff_files = []
        
        for collection_name, metadata in self.collections.items():
            collection_path = self.data_dir / collection_name
            
            if not collection_path.exists():
                logger.warning(f"Collection directory not found: {collection_path}")
                continue
                
            # Find GBFF files recursively
            for gbff_file in collection_path.rglob("*.gbff*"):
                if gbff_file.is_file():
                    file_metadata = metadata.copy()
                    file_metadata['collection'] = collection_name
                    file_metadata['gbff_path'] = str(gbff_file)
                    gbff_files.append((str(gbff_file), file_metadata))
        
        logger.info(f"Found {len(gbff_files)} GBFF files across {len(self.collections)} collections")
        return gbff_files

    @staticmethod
    def analyze_gbff_file(gbff_path: str, metadata: Dict) -> Optional[Dict]:
        """Analyze a single GBFF file and extract comprehensive statistics."""
        try:
            # Extract genome ID from path
            genome_id = Path(gbff_path).parent.name
            if not genome_id.startswith(('GCF_', 'GCA_')):
                # Try to extract from filename
                genome_id = Path(gbff_path).stem.split('_genomic')[0]
            
            # Initialize statistics
            stats = {
                'genome_id': genome_id,
                'gbff_path': gbff_path,
                'collection': metadata['collection'],
                'pathogenicity': metadata['pathogenicity'],
                'is_pathogenic': 1 if metadata['pathogenicity'] == 'pathogenic' else 0,
                'species': metadata['species'],
                'taxon_id': metadata['taxon_id'],
                'genome_size_bp': 0,
                'gc_content_percent': 0.0,
                'cds_count': 0,
                'trna_count': 0,
                'rrna_count': 0,
                'gene_count': 0,
                'contig_count': 0,
                'n50_length': 0,
                'assembly_level': 'unknown',
                'total_genes': 0,
                'protein_coding_genes': 0,
                'pseudogenes': 0
            }
            
            # Parse GBFF file with BioPython
            sequences = []
            contig_lengths = []
            
            with open(gbff_path, 'r') as handle:
                for record in SeqIO.parse(handle, "genbank"):
                    sequences.append(record)
                    contig_lengths.append(len(record.seq))
                    stats['genome_size_bp'] += len(record.seq)
                    
                    # Count features
                    for feature in record.features:
                        if feature.type == 'CDS':
                            stats['cds_count'] += 1
                            stats['protein_coding_genes'] += 1
                        elif feature.type == 'tRNA':
                            stats['trna_count'] += 1
                        elif feature.type == 'rRNA':
                            stats['rrna_count'] += 1
                        elif feature.type == 'gene':
                            stats['gene_count'] += 1
                        elif feature.type == 'pseudogene':
                            stats['pseudogenes'] += 1
                    
                    # Extract assembly level from annotations
                    if 'structured_comment' in record.annotations:
                        structured_comment = record.annotations['structured_comment']
                        if 'Assembly-Data' in structured_comment:
                            assembly_data = structured_comment['Assembly-Data']
                            if 'Assembly Level' in assembly_data:
                                stats['assembly_level'] = assembly_data['Assembly Level']
            
            # Calculate additional statistics
            stats['contig_count'] = len(sequences)
            stats['total_genes'] = stats['gene_count'] + stats['cds_count']
            
            # Calculate GC content from concatenated sequences
            if sequences:
                full_sequence = ''.join(str(record.seq) for record in sequences)
                if full_sequence:
                    stats['gc_content_percent'] = round(GC(full_sequence), 2)
            
            # Calculate N50
            if contig_lengths:
                contig_lengths.sort(reverse=True)
                total_length = sum(contig_lengths)
                cumulative_length = 0
                for length in contig_lengths:
                    cumulative_length += length
                    if cumulative_length >= total_length * 0.5:
                        stats['n50_length'] = length
                        break
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing {gbff_path}: {str(e)}")
            return None

    def process_genomes_parallel(self, gbff_files: List[Tuple[str, Dict]]) -> pd.DataFrame:
        """Process GBFF files in parallel and return results as DataFrame."""
        logger.info(f"Processing {len(gbff_files)} genomes using {self.n_workers} workers...")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.analyze_gbff_file, gbff_path, metadata): (gbff_path, metadata)
                for gbff_path, metadata in gbff_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(gbff_files), desc="Analyzing genomes") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        logger.info(f"Successfully processed {len(results)}/{len(gbff_files)} genomes")
        return pd.DataFrame(results)

    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics from the processed data."""
        stats = {
            'processing_summary': {
                'total_genomes': len(df),
                'pathogenic_genomes': len(df[df['is_pathogenic'] == 1]),
                'non_pathogenic_genomes': len(df[df['is_pathogenic'] == 0]),
                'species_count': df['species'].nunique(),
                'collections_processed': df['collection'].nunique()
            },
            'genome_size_stats': {
                'mean_size_bp': int(df['genome_size_bp'].mean()),
                'median_size_bp': int(df['genome_size_bp'].median()),
                'std_size_bp': int(df['genome_size_bp'].std()),
                'min_size_bp': int(df['genome_size_bp'].min()),
                'max_size_bp': int(df['genome_size_bp'].max())
            },
            'gc_content_stats': {
                'mean_gc_percent': round(df['gc_content_percent'].mean(), 2),
                'median_gc_percent': round(df['gc_content_percent'].median(), 2),
                'std_gc_percent': round(df['gc_content_percent'].std(), 2)
            },
            'gene_content_stats': {
                'mean_cds_count': int(df['cds_count'].mean()),
                'mean_total_genes': int(df['total_genes'].mean()),
                'mean_trna_count': int(df['trna_count'].mean()),
                'mean_rrna_count': int(df['rrna_count'].mean())
            },
            'assembly_quality': {
                'mean_contig_count': round(df['contig_count'].mean(), 1),
                'mean_n50_length': int(df['n50_length'].mean()),
                'assembly_levels': df['assembly_level'].value_counts().to_dict()
            }
        }
        
        # Per-species statistics
        stats['per_species_stats'] = {}
        for species in df['species'].unique():
            species_df = df[df['species'] == species]
            stats['per_species_stats'][species] = {
                'genome_count': len(species_df),
                'mean_size_bp': int(species_df['genome_size_bp'].mean()),
                'mean_gc_percent': round(species_df['gc_content_percent'].mean(), 2),
                'mean_cds_count': int(species_df['cds_count'].mean())
            }
        
        # Pathogenicity comparison
        pathogenic_df = df[df['is_pathogenic'] == 1]
        non_pathogenic_df = df[df['is_pathogenic'] == 0]
        
        stats['pathogenicity_comparison'] = {
            'pathogenic': {
                'count': len(pathogenic_df),
                'mean_size_bp': int(pathogenic_df['genome_size_bp'].mean()) if len(pathogenic_df) > 0 else 0,
                'mean_gc_percent': round(pathogenic_df['gc_content_percent'].mean(), 2) if len(pathogenic_df) > 0 else 0,
                'mean_cds_count': int(pathogenic_df['cds_count'].mean()) if len(pathogenic_df) > 0 else 0
            },
            'non_pathogenic': {
                'count': len(non_pathogenic_df),
                'mean_size_bp': int(non_pathogenic_df['genome_size_bp'].mean()) if len(non_pathogenic_df) > 0 else 0,
                'mean_gc_percent': round(non_pathogenic_df['gc_content_percent'].mean(), 2) if len(non_pathogenic_df) > 0 else 0,
                'mean_cds_count': int(non_pathogenic_df['cds_count'].mean()) if len(non_pathogenic_df) > 0 else 0
            }
        }
        
        return stats

    def save_results(self, df: pd.DataFrame, stats: Dict):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed genome statistics
        detailed_csv = self.output_dir / f"genome_detailed_statistics_{timestamp}.csv"
        df.to_csv(detailed_csv, index=False)
        logger.info(f"Detailed statistics saved to: {detailed_csv}")
        
        # Save unified manifest (subset of columns for training)
        manifest_columns = [
            'genome_id', 'gbff_path', 'is_pathogenic', 
            'species', 'taxon_id', 'pathogenicity'
        ]
        manifest_df = df[manifest_columns].copy()
        manifest_csv = self.output_dir / f"gbff_manifest_full_{timestamp}.tsv"
        manifest_df.to_csv(manifest_csv, sep='\t', index=False)
        logger.info(f"Unified manifest saved to: {manifest_csv}")
        
        # Save summary statistics as JSON
        stats_json = self.output_dir / f"genome_summary_statistics_{timestamp}.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Summary statistics saved to: {stats_json}")
        
        # Save collection-level summary as CSV
        collection_summary = []
        for collection, metadata in self.collections.items():
            collection_df = df[df['collection'] == collection]
            if len(collection_df) > 0:
                summary = {
                    'collection': collection,
                    'pathogenicity': metadata['pathogenicity'],
                    'species': metadata['species'],
                    'taxon_id': metadata['taxon_id'],
                    'genome_count': len(collection_df),
                    'mean_size_bp': int(collection_df['genome_size_bp'].mean()),
                    'mean_gc_percent': round(collection_df['gc_content_percent'].mean(), 2),
                    'mean_cds_count': int(collection_df['cds_count'].mean()),
                    'total_bp': int(collection_df['genome_size_bp'].sum())
                }
                collection_summary.append(summary)
        
        collection_df = pd.DataFrame(collection_summary)
        collection_csv = self.output_dir / f"collection_summary_{timestamp}.csv"
        collection_df.to_csv(collection_csv, index=False)
        logger.info(f"Collection summary saved to: {collection_csv}")
        
        return {
            'detailed_stats': detailed_csv,
            'manifest': manifest_csv,
            'summary_stats': stats_json,
            'collection_summary': collection_csv
        }

    def run(self) -> Dict:
        """Run the complete genome processing pipeline."""
        logger.info("Starting FoodGuardAI Genome Processing Pipeline")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Using {self.n_workers} parallel workers")
        
        # Find all GBFF files
        gbff_files = self.find_gbff_files()
        if not gbff_files:
            raise ValueError("No GBFF files found in the specified collections")
        
        # Process genomes in parallel
        df = self.process_genomes_parallel(gbff_files)
        if df.empty:
            raise ValueError("No genomes were successfully processed")
        
        # Generate statistics
        logger.info("Generating comprehensive statistics...")
        stats = self.generate_statistics(df)
        
        # Save results
        logger.info("Saving results...")
        output_files = self.save_results(df, stats)
        
        # Print summary
        self.print_summary(stats)
        
        return {
            'dataframe': df,
            'statistics': stats,
            'output_files': output_files
        }

    def print_summary(self, stats: Dict):
        """Print a comprehensive summary of the processing results."""
        print("\n" + "="*60)
        print("FOODGUARDAI GENOME PROCESSING COMPLETE!")
        print("="*60)
        
        summary = stats['processing_summary']
        print(f"📊 Dataset Overview:")
        print(f"   Total genomes processed: {summary['total_genomes']:,}")
        print(f"   Pathogenic genomes: {summary['pathogenic_genomes']:,}")
        print(f"   Non-pathogenic genomes: {summary['non_pathogenic_genomes']:,}")
        print(f"   Species diversity: {summary['species_count']} species")
        print(f"   Collections processed: {summary['collections_processed']}")
        
        if summary['pathogenic_genomes'] > 0 and summary['non_pathogenic_genomes'] > 0:
            ratio = summary['pathogenic_genomes'] / summary['non_pathogenic_genomes']
            print(f"   Pathogenic:Non-pathogenic ratio: {ratio:.2f}:1")
        
        genome_stats = stats['genome_size_stats']
        print(f"\n🧬 Genome Size Statistics:")
        print(f"   Mean size: {genome_stats['mean_size_bp']:,} bp")
        print(f"   Size range: {genome_stats['min_size_bp']:,} - {genome_stats['max_size_bp']:,} bp")
        
        gc_stats = stats['gc_content_stats']
        print(f"\n🔬 GC Content Statistics:")
        print(f"   Mean GC content: {gc_stats['mean_gc_percent']:.1f}%")
        
        gene_stats = stats['gene_content_stats']
        print(f"\n🧪 Gene Content Statistics:")
        print(f"   Mean CDS count: {gene_stats['mean_cds_count']:,}")
        print(f"   Mean total genes: {gene_stats['mean_total_genes']:,}")
        
        print(f"\n✅ Ready for:")
        print(f"   • ESM-2 cache population")
        print(f"   • Train/validation/test splits")
        print(f"   • Bacformer fine-tuning")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='FoodGuardAI Genome Processor')
    parser.add_argument('--data-dir', required=True, 
                       help='Path to genome data directory')
    parser.add_argument('--output-dir', required=True,
                       help='Path to output directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    try:
        processor = GenomeProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_workers=args.workers
        )
        
        results = processor.run()
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
