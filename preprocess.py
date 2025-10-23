import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import multiprocessing
from tqdm import tqdm
import random
from sklearn.decomposition import NMF
import pickle

from autoencoder import *



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



def find_marker_genes(sc_exp, 
                      preprocess = True,
                      highly_variable_genes = True,
                      regress_out = False,
                      scale = False,
                      PCA_components = 50, 
                      marker_gene_method = 'wilcoxon',
                      filter_wilcoxon_marker_genes = True, 
                      top_gene_per_type = 20, 
                      pvals_adj_threshold = 0.10,
                      log_fold_change_threshold = 1,
                      min_within_group_fraction_threshold = 0.7,
                      max_between_group_fraction_threshold = 0.3,
                     ):

    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(sc_exp.copy(), 
                                             normalize=True,
                                             log=True,
                                             highly_variable_genes=highly_variable_genes, 
                                             regress_out=regress_out, 
                                             scale=scale,
                                            )
    else:
        sc_adata_marker_gene = sc_exp.copy()

    sc.tl.pca(sc_adata_marker_gene, n_comps=PCA_components, svd_solver='arpack', random_state=None)
    
    layer = 'scale.data'
    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True, 
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)

    if marker_gene_method == 'wilcoxon':
        if filter_wilcoxon_marker_genes == True:
            gene_dict = {}
            gene_list = []
            for name in sc_adata_marker_gene.obs['cell_type'].unique():
                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name, key=marker_gene_method).sort_values('pvals_adj')
                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]
                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]
                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]
                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]
                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    else:
        print("marker_gene_method should be 'logreg' or 'wilcoxon'")
    
    return gene_list, gene_dict



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
    
    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method) for i in range(spot_num)]
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



def data_integration(real, 
                     pseudo, 
                     batch_removal_method="combat",
                     dimensionality_reduction_method='PCA', 
                     dim=50, 
                     scale=True,
                     autoencoder_epoches=2000,
                     autoencoder_LR=1e-3,
                     autoencoder_drop=0,
                     cpu_num=-1,
                     AE_device='GPU'
                    ):
    
    if batch_removal_method == 'mnn':
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True, var_subset=None)
        adata = mnn[0]
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table = table.iloc[pseudo.shape[0]:,:].append(table.iloc[:pseudo.shape[0],:])
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'scanorama':
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred = dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = table1.append(table2)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'combat':
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    else:
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist()+pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num', real.obs['cell_type_num'].values.tolist()+pseudo.obs['cell_type_num'].values.tolist())

    return table




def preprocess_SPGVAE(paths,
               find_marker_genes_paras,
               pseudo_spot_simulation_paras,
               data_normalization_paras,
               integration_for_feature_paras,
               use_marker_genes = True,
               external_genes = False,
               generate_new_pseudo_spots = True,
               n_jobs = -1,
               GCN_device = 'CPU'
              ):

    sc_path = paths['sc_path']
    ST_path = paths['ST_path']
    output_path = paths['output_path']
    
    sc_adata = sc.read_csv(sc_path+"/sc_data.tsv", delimiter='\t')
    sc_label = pd.read_table(sc_path+"/sc_label.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
    sc_label.columns = ['cell_type']
    sc_adata.obs['cell_type'] = sc_label['cell_type'].values

    cell_type_num = len(sc_adata.obs['cell_type'].unique())
    cell_types = sc_adata.obs['cell_type'].unique()

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
    sc_adata.obs['cell_type_idx'] = celltype_idx
    sc_adata.obs['cell_type'].value_counts()
    if use_marker_genes == True:
        if external_genes == True:
            with open(sc_path+"/marker_genes.tsv", 'r') as f:
                selected_genes = [line.rstrip('\n') for line in f]
        else:
            selected_genes, cell_type_marker_genes = find_marker_genes(sc_adata,
                                                                      preprocess = find_marker_genes_paras['preprocess'],
                                                                      highly_variable_genes = find_marker_genes_paras['highly_variable_genes'],
                                                                      PCA_components = find_marker_genes_paras['PCA_components'], 
                                                                      filter_wilcoxon_marker_genes = find_marker_genes_paras['filter_wilcoxon_marker_genes'], 
                                                                      marker_gene_method = find_marker_genes_paras['marker_gene_method'],
                                                                      pvals_adj_threshold = find_marker_genes_paras['pvals_adj_threshold'],
                                                                      log_fold_change_threshold = find_marker_genes_paras['log_fold_change_threshold'],
                                                                      min_within_group_fraction_threshold = find_marker_genes_paras['min_within_group_fraction_threshold'],
                                                                      max_between_group_fraction_threshold = find_marker_genes_paras['max_between_group_fraction_threshold'],
                                                                      top_gene_per_type = find_marker_genes_paras['top_gene_per_type'])
            with open(output_path+"/marker_genes.tsv", 'w') as f:
                for gene in selected_genes:
                    f.write(str(gene) + '\n')
            
    print("{} genes have been selected as marker genes.".format(len(selected_genes)))
    
    

    if generate_new_pseudo_spots == True:
        pseudo_adata = pseudo_spot_generation(sc_adata,
                                              idx_to_word_celltype,
                                              spot_num = pseudo_spot_simulation_paras['spot_num'],
                                              min_cell_number_in_spot = pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                                              max_cell_number_in_spot = pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                                              max_cell_types_in_spot = pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                                              generation_method = pseudo_spot_simulation_paras['generation_method'],
                                              n_jobs = n_jobs
                                              )
        data_file = open(output_path+'/pseudo_ST.pkl','wb')
        pickle.dump(pseudo_adata, data_file)
        data_file.close()
    else:
        data_file = open(output_path+'/pseudo_ST.pkl','rb')
        pseudo_adata = pickle.load(data_file)
        data_file.close()

    ST_adata = sc.read_csv(ST_path+"/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path+"/coordinates.csv", sep = ',', header = 0, index_col = 0, encoding = "utf-8")
    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']
    

    ST_genes = ST_adata.var.index.values
    pseudo_genes = pseudo_adata.var.index.values
    common_genes = set(ST_genes).intersection(set(pseudo_genes))
    ST_adata_filter = ST_adata[:,list(common_genes)]
    pseudo_adata_filter = pseudo_adata[:,list(common_genes)]
    
    
    ST_adata_filter_norm = ST_preprocess(ST_adata_filter, 
                                         normalize = data_normalization_paras['normalize'], 
                                         log = data_normalization_paras['log'], 
                                         scale = data_normalization_paras['scale'],
                                        )[:,selected_genes]
    
    try:
        try:
            ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
        except:
            ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']
    except:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', [0]*ST_adata_filter_norm.obs.shape[0])
    for i in cell_types:
        try:
            ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
        except:
            ST_adata_filter_norm.obs[i] = [0]*ST_adata_filter_norm.obs.shape[0]
    try:
        ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types]>0).sum(axis=1)
    except:
        ST_adata_filter_norm.obs['cell_type_num'] = [0]*ST_adata_filter_norm.obs.shape[0]


    pseudo_adata_norm = ST_preprocess(pseudo_adata_filter, 
                                      normalize = data_normalization_paras['normalize'], 
                                      log = data_normalization_paras['log'], 
                                      scale = data_normalization_paras['scale'],
                                     )[:,selected_genes]
    pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types]>0).sum(axis=1)
    
        
    ST_integration_batch_removed = data_integration(ST_adata_filter_norm, 
                                                    pseudo_adata_norm, 
                                                    batch_removal_method=integration_for_feature_paras['batch_removal_method'], 
                                                    dim=min(int(ST_adata_filter_norm.shape[1]*1/2), integration_for_feature_paras['dim']), 
                                                    dimensionality_reduction_method=integration_for_feature_paras['dimensionality_reduction_method'], 
                                                    scale=integration_for_feature_paras['scale'],
                                                    cpu_num=n_jobs,
                                                    AE_device=GCN_device
                                                   )


    # keep only the selected_genes in sc_adata 
    sc_adata = sc_adata[:, [g for g in selected_genes if g in sc_adata.var_names]].copy()
    sc_adata = ST_preprocess(sc_adata, 
                             normalize=True, 
                             log=True, 
                             scale=False)


    # feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)
    ST_integration_batch_removed = ST_integration_batch_removed.iloc[:, 3:]
    psuedo_dist = pseudo_adata.obs.iloc[:, :-1]

    print()
    print("========= Preprocessing Summary =========")
    print("Number of cell types: {}".format(cell_type_num))
    print("Number of common genes between ST data and pseudo spots: {}".format(len(common_genes)))
    print("Number of marker genes used: {} genes.".format(len(selected_genes)))
    print("ST data: ", ST_adata_filter_norm.shape)
    print("Pseudo spots: ", pseudo_adata_norm.shape)
    print("Total spots: ", ST_integration_batch_removed.shape)
    print("Pseudo dist: ", psuedo_dist.shape)

    # convert all these to pandas dataframe 
    sc_adata_df = sc.get.obs_df(sc_adata, ['cell_type', *sc_adata.var_names])
    ST_adata_filter_norm_df = pd.DataFrame(ST_adata_filter_norm.X, index=ST_adata_filter_norm.obs.index, columns=ST_adata_filter_norm.var.index)
    pseudo_adata_norm_df = pd.DataFrame(pseudo_adata_norm.X, index=pseudo_adata_norm.obs.index, columns=pseudo_adata_norm.var.index)
    ST_integration_batch_removed_df = pd.DataFrame(ST_integration_batch_removed.values, index=ST_integration_batch_removed.index, columns=ST_integration_batch_removed.columns)
    psuedo_dist_df = pd.DataFrame(psuedo_dist.values, index=psuedo_dist.index, columns=psuedo_dist.columns)

    return {
        "real_spots": ST_adata_filter_norm_df,
        "real_coordinates": ST_adata.obs[['coor_X', 'coor_Y']],
        "pseudo_spots": pseudo_adata_norm_df,
        "all_spots": ST_integration_batch_removed_df,
        "pseudo_dist": psuedo_dist_df,
        "cell_types": cell_types,
        "sc_ref": sc_adata_df,
    } 