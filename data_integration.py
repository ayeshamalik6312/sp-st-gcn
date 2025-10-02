import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import multiprocessing
from tqdm import tqdm
from scipy import stats            
import random
from tqdm.auto import tqdm
from sklearn.decomposition import NMF


def find_marker_genes(sc_df, n_markers_per_type=5, find_marker_method='fold_change'):
    gene_names = [g for g in sc_df.columns if g != "cell_type"]
    cell_types = sc_df['cell_type'].unique()

    # Create a score matrix
    score_matrix = pd.DataFrame(index=gene_names, columns=cell_types)
    marker_dict = {}

    # progress bar over cell types
    for cell_type in tqdm(cell_types, desc="Finding Marker Genes", unit="type"):
        # Split cells into this type vs others
        is_type = sc_df['cell_type'] == cell_type
        expr_in_type = sc_df.loc[is_type, gene_names]
        expr_other   = sc_df.loc[~is_type, gene_names]

        if find_marker_method == 'fold_change':
            mean_in_type = expr_in_type.mean() + 0.1
            mean_other   = expr_other.mean() + 0.1
            fold_change  = mean_in_type / mean_other
            scores = fold_change

        elif find_marker_method == 'wilcoxon':
            scores = pd.Series(index=gene_names, dtype=float)
            # optional: inner progress bar over genes (nice when there are many genes)
            for gene in tqdm(gene_names, desc=f"{cell_type[:18]} genes", unit="gene", leave=False):
                stat, pval = stats.mannwhitneyu(
                    expr_in_type[gene],
                    expr_other[gene],
                    alternative='greater'
                )
                scores[gene] = -np.log10(pval + 1e-100)
        else:
            print("method name must either be 'wilcoxon' or 'fold_change'")
            continue

        score_matrix[cell_type] = scores

        # Get top N markers
        top_markers = scores.nlargest(n_markers_per_type).index
        marker_indices = [gene_names.index(g) for g in top_markers]
        marker_dict[cell_type] = marker_indices

        # tqdm.write(f"{cell_type[:20]:20s}: {list(top_markers[:5])}")

    return marker_dict, score_matrix

def ST_preprocess(ST_exp, 
                  normalize=True,
                  log=True,
                  highly_variable_genes=False, 
                  regress_out=False, 
                  scale=False,
                  scale_max_value=None,
                  scale_zero_center=True,
                  hvg_min_mean=0.0125,
                  hvg_max_mean=3,
                  hvg_min_disp=0.5,
                  highly_variable_gene_num=None
                 ):
    
    adata = ST_exp.copy()
    
    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        
    if log == True:
        sc.pp.log1p(adata)
        
    adata.layers['scale.data'] = adata.X.copy()
    
    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(adata, 
                                    min_mean=hvg_min_mean, 
                                    max_mean=hvg_max_mean, 
                                    min_disp=hvg_min_disp,
                                    n_top_genes=highly_variable_gene_num,
                                   )
        adata = adata[:, adata.var.highly_variable]
        
    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
    
    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)
    
    return adata



def generate_a_spot(sc_exp, 
                    min_cell_number_in_spot, 
                    max_cell_number_in_spot,
                    max_cell_types_in_spot,
                    generation_method,
                   ):
    
    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)
        
        while(True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]
        
        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])
            
        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')

        

def pseudo_spot_generation(sc_exp, 
                           idx_to_word_celltype,
                           spot_num, 
                           min_cell_number_in_spot, 
                           max_cell_number_in_spot,
                           max_cell_types_in_spot,
                           generation_method,
                           n_jobs = -1
                          ):
    
    cell_type_num = len(sc_exp.obs['cell_type'].unique())
    
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, 
             max_cell_types_in_spot, generation_method) for _ in range(spot_num)]

    # >>> NEW: serial path to avoid multiprocessing on Windows
    if n_jobs is None or n_jobs <= 1:
        generated_spots = [generate_a_spot(*a) for a in tqdm(args, desc='Generating pseudo-spots')]
    else:
        cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        with multiprocessing.Pool(processes=cores) as pool:
            generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))

    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:,:].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i]/pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)
        
    return pseudo_spots

