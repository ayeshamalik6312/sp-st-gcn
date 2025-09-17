# marker_genes.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

def find_marker_genes(sc_df, n_markers_per_type=5, method='fold_change'):
    """
    Identify marker genes for each cell type using fold change
    
    Args:
        sc_df: DataFrame with gene expression and 'cell_type' column
        n_markers_per_type: Number of top markers per cell type
        method: 'fold_change' or 'wilcoxon'
    
    Returns:
        marker_dict: {cell_type: [gene_indices]}
        marker_scores: DataFrame of scores for each gene/cell type
    """
    gene_names = list(sc_df.columns[:-1])  # exclude 'cell_type'
    cell_types = sc_df['cell_type'].unique()
    
    # Create a score matrix
    score_matrix = pd.DataFrame(index=gene_names, columns=cell_types)
    marker_dict = {}
    
    print("\n=== Finding Marker Genes ===")
    
    for cell_type in cell_types:
        # Split cells into this type vs others
        is_type = sc_df['cell_type'] == cell_type
        expr_in_type = sc_df.loc[is_type, gene_names]
        expr_other = sc_df.loc[~is_type, gene_names]
        
        if method == 'fold_change':
            # Calculate fold change (with pseudocount to avoid division by zero)
            mean_in_type = expr_in_type.mean() + 0.1
            mean_other = expr_other.mean() + 0.1
            fold_change = mean_in_type / mean_other
            scores = fold_change
            
        elif method == 'wilcoxon':
            # Wilcoxon rank-sum test for each gene
            scores = pd.Series(index=gene_names, dtype=float)
            for gene in gene_names:
                stat, pval = stats.mannwhitneyu(
                    expr_in_type[gene], 
                    expr_other[gene], 
                    alternative='greater'
                )
                # Use -log10(p-value) as score
                scores[gene] = -np.log10(pval + 1e-100)
        
        score_matrix[cell_type] = scores
        
        # Get top N markers
        top_markers = scores.nlargest(n_markers_per_type).index
        marker_indices = [gene_names.index(g) for g in top_markers]
        marker_dict[cell_type] = marker_indices
        
        # Print top 3 for visualization
        print(f"{cell_type[:20]:20s}: {list(top_markers[:3])}")
    
    return marker_dict, score_matrix

def create_marker_weight_matrix(marker_dict, n_genes, n_celltypes, 
                                cell_type_names, base_weight=1.0, marker_weight=3.0):
    """
    Create a weight matrix for genes based on marker status
    
    Returns:
        weight_matrix: [n_celltypes, n_genes] tensor of weights
    """
    import torch
    
    # Initialize with base weights
    weight_matrix = torch.ones(n_celltypes, n_genes) * base_weight
    
    # Increase weight for marker genes
    for i, ct in enumerate(cell_type_names):
        if ct in marker_dict:
            for gene_idx in marker_dict[ct]:
                weight_matrix[i, gene_idx] = marker_weight
    
    return weight_matrix