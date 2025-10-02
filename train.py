#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import yaml
import torch
import pandas as pd
import scanpy as sc
import argparse, sys
import numpy as np, random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import *
from data_integration import *
from pathlib import Path
from models import SpatialVAE
from losses import vae_loss
from metrics import evaluate_all
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config (default: ./config.yaml)")
args, _ = parser.parse_known_args()

# Replace your existing "with open('config.yaml')" with:
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

st_data_dir = Path(config["st_data_dir"]).expanduser()
sc_data_dir = Path(config["sc_data_dir"]).expanduser()
results_path = Path(config["results_path"]).expanduser()
results_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Load Data ----------------
# ST data
Y_df = pd.read_csv(os.path.join(st_data_dir, "ST_data.tsv"), sep="\t", index_col=0)
coords_df = pd.read_csv(os.path.join(st_data_dir, "coordinates.csv"), index_col=0)

# keep intersection
intersection = sorted(set(Y_df.index) & set(coords_df.index))
Y_df = Y_df.loc[intersection].sort_index()
coords_df = coords_df.loc[intersection].sort_index()

Y = torch.tensor(Y_df.values, dtype=torch.float32).to(device)
coords = torch.tensor(coords_df.values, dtype=torch.float32).to(device)

# scRNA reference
sc_df = pd.read_csv(os.path.join(sc_data_dir, "sc_data.tsv"), sep="\t", index_col=0)
labels_df = pd.read_csv(os.path.join(sc_data_dir, "sc_label.tsv"), sep="\t", index_col=0)
sc_df = sc_df.loc[labels_df.index]
sc_df["cell_type"] = labels_df.squeeze()
sc_df_raw = sc_df.copy(deep=True)

# align genes
common_genes = Y_df.columns.intersection(sc_df.columns)
Y_df = Y_df[common_genes]
sc_df = sc_df[list(common_genes) + ["cell_type"]]

# -------------- Find marker genes -------------
marker_dict, marker_scores = find_marker_genes(
    sc_df, 
    n_markers_per_type=int(float(config.get("n_markers_per_type", 4))),
    find_marker_method=config['find_marker_method']
)

# Save marker genes info for analysis
marker_scores.to_csv(os.path.join(results_path, "marker_scores.csv"))
with open(os.path.join(results_path, "marker_genes.txt"), 'w') as f:
    for ct, indices in marker_dict.items():
        gene_names_list = [list(common_genes)[i] for i in indices[:5]]
        f.write(f"{ct}: {gene_names_list}\n")

# build marker name list and order it (for subsetting use)
gene_names = [g for g in sc_df.columns if g != "cell_type"]
selected_genes = sorted({ gene_names[i] for idxs in marker_dict.values() for i in idxs })
selected_genes_ordered = [g for g in Y_df.columns if g in selected_genes]

# subset to identified marker genes 
Y_df  = Y_df[selected_genes_ordered]
sc_df = sc_df[selected_genes_ordered + ["cell_type"]]
sc_raw_markers = sc_df_raw[selected_genes_ordered].copy()    # for pseudo spots
sc_raw_markers["cell_type"] = sc_df["cell_type"].values      # for pseudo spots

# ------------- Pseudo Spot Generation -------------

# build AnnData for pseudo generation
sc_adata = sc.AnnData(
    X=sc_raw_markers[selected_genes_ordered].values,  
    obs=pd.DataFrame({"cell_type": sc_raw_markers["cell_type"].astype(str).values},
                     index=sc_raw_markers.index),
    var=pd.DataFrame(index=selected_genes_ordered)
)

cell_types = sc_adata.obs["cell_type"].unique().tolist()
word_to_idx = {ct: i for i, ct in enumerate(cell_types)}
idx_to_word = {i: ct for ct, i in word_to_idx.items()}
sc_adata.obs["cell_type_idx"] = sc_adata.obs["cell_type"].map(word_to_idx).astype(int)

# generate pseudo spots 
from data_integration import pseudo_spot_generation
pseudo_cfg = config.get("pseudo", {
    "spot_num": 2000,
    "min_cell_num_in_spot": 2,
    "max_cell_num_in_spot": 8,
    "max_cell_types_in_spot": 3,
    "generation_method": "celltype",
    "n_jobs": 1
})
pseudo_adata = pseudo_spot_generation(
    sc_exp=sc_adata,
    idx_to_word_celltype=idx_to_word,
    spot_num=pseudo_cfg["spot_num"],
    min_cell_number_in_spot=pseudo_cfg["min_cell_num_in_spot"],
    max_cell_number_in_spot=pseudo_cfg["max_cell_num_in_spot"],
    max_cell_types_in_spot=pseudo_cfg["max_cell_types_in_spot"],
    generation_method=pseudo_cfg["generation_method"],
    n_jobs=pseudo_cfg["n_jobs"]
)

Y_pseudo_raw = pd.DataFrame(pseudo_adata.X,
                            index=pseudo_adata.obs_names,
                            columns=pseudo_adata.var_names)[selected_genes_ordered]

# --------------- Normalize ----------------

# scale real spatial transcriptomics data 
scaler_Y = StandardScaler()
Y_df[selected_genes_ordered] = scaler_Y.fit_transform(Y_df[selected_genes_ordered])

# scale pseudo spots
Y_pseudo = pd.DataFrame(
    scaler_Y.transform(Y_pseudo_raw), 
    index=Y_pseudo_raw.index,
    columns=selected_genes_ordered
)

# scale single cell ref data 
scaler_X = StandardScaler()
sc_df[selected_genes_ordered] = scaler_X.fit_transform(sc_df[selected_genes_ordered])

X_ref_df = sc_df.groupby("cell_type").mean()
X_ref_df[selected_genes_ordered] = scaler_X.fit_transform(X_ref_df[selected_genes_ordered])

assert Y_df.columns.equals(X_ref_df.columns), "ST and reference gene orders differ."

cell_type_names = X_ref_df.index.tolist()

# build pseudo ground truth
B_pseudo = (
    pseudo_adata.obs.reindex(columns=cell_type_names)  # ensures correct order
    .fillna(0.0)
    .astype(float)
)

# final checks
assert Y_df.columns.equals(X_ref_df.columns), "ST and reference gene orders differ."
assert list(Y_pseudo.columns) == list(Y_df.columns), "Pseudo/ST marker order mismatch."
assert list(B_pseudo.columns) == list(cell_type_names), "Pseudo fraction columns mismatch."

Y = torch.tensor(Y_df.values, dtype=torch.float32).to(device)
X = torch.tensor(X_ref_df.values, dtype=torch.float32).to(device)

print(f"Data loaded: ST spots={Y.shape[0]}, genes={Y.shape[1]}, cells={sc_df.shape[0]}, cell types={len(cell_type_names)}\n")

# ---------------- Adjacency ----------------
A = compute_adjacency(
    coords, 
    sigma=config["adjacency_sigma"], 
    threshold=config["adjacency_threshold"]
)
A = normalize_adjacency(A)
edge_index_all, edge_weight_all = get_edges(A)

# ---------------- Split ----------------
split_seed = config["split_seed"] if config["split_seed"] else int(time.time()) % 10000
train, val, test = custom_split(Y, A, coords,
                                val_size=config["val_size"],
                                test_size=config["split_size"],
                                seed=split_seed)
Y_train, A_train, coords_train = train["Y"], train["A"], train["coords"]
Y_val, A_val, coords_val = val["Y"], val["A"], val["coords"]
Y_test, A_test, coords_test = test["Y"], test["A"], test["coords"]

edge_index_train, edge_weight_train = get_edges(A_train)
edge_index_val, edge_weight_val = get_edges(A_val)
edge_index_test, edge_weight_test = get_edges(A_test)

# ---------------- Model ----------------
input_dim = Y.shape[1]
n_celltypes = X.shape[0]
model = SpatialVAE(input_dim, config["hidden_dim"], config["latent_dim"],
                   n_celltypes, tau=config["tau"], drop=config["dropout_rate"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["l2_lambda"]))

# ---------------- Training ----------------
loss_history, val_loss_history = [], []
best_val_loss, patience_counter = float("inf"), 0
patience = config["patience"]

for epoch in range(config["n_epochs"]):
    # Train
    model.train()
    optimizer.zero_grad()
    Y_hat, mu, logvar, B = model(Y_train, edge_index_train, edge_weight_train, X)
    loss, recon_loss, kl_loss, ot_loss, smooth_loss = vae_loss(
        Y_train, Y_hat, mu, logvar, A_train, B,
        alpha=config["loss_alpha"], beta=config["loss_beta"],
        gamma=config["loss_gamma"], delta=config["loss_delta"],
        latent_dim=config["latent_dim"]
    )
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        Y_hat_val, mu_val, logvar_val, B_val = model(Y_val, edge_index_val, edge_weight_val, X)
        val_loss, _, _, _, _ = vae_loss(
            Y_val, Y_hat_val, mu_val, logvar_val, A_val, B_val,
            alpha=config["loss_alpha"], beta=config["loss_beta"],
            gamma=config["loss_gamma"], delta=config["loss_delta"],
            latent_dim=config["latent_dim"]
        )
        val_loss_history.append(val_loss.item())

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}: Train={loss.item():.4f} | Val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# restore best model
model.load_state_dict(best_state)

# save loss curves
plt.plot(loss_history, label="Train")
plt.plot(val_loss_history, label="Val")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_path, "loss_curves.svg"), format="svg", dpi=300)
plt.close()

# ---------------- Test Evaluation ----------------
model.eval()
with torch.no_grad():
    Y_pred_test, mu_test, logvar_test, B_test = model(Y_test, edge_index_test, edge_weight_test, X)

# Save outputs
B_df = pd.DataFrame(B_test.cpu().numpy(), index=Y_df.iloc[test["idx"]].index, columns=cell_type_names)
B_df.to_csv(os.path.join(results_path, "predicted_proportions_test.csv"))
Yhat_df = pd.DataFrame(Y_pred_test.cpu().numpy(), index=Y_df.iloc[test["idx"]].index, columns=Y_df.columns)
Yhat_df.to_csv(os.path.join(results_path, "reconstructed_expression_test.csv"))

# evaluate metrics
G_df = pd.read_csv(os.path.join(st_data_dir, "ST_ground_truth.tsv"), sep="\t", index_col=0)
G_test = G_df.loc[B_df.index, cell_type_names].values
y_pred = B_df.values

results = evaluate_all(G_test, y_pred)
print("\nEvaluation Results (Test):")
for k, v in results.items():
    print(f"{k:12s}: {v:.4f}")

# append config parameters to results
for k, v in config.items():
    results[k] = v
results["split_seed"] = split_seed

# append if results already exist
metrics_file = os.path.join(results_path, "metrics.csv")
if os.path.exists(metrics_file):
    existing_results = pd.read_csv(metrics_file)
    results_df = pd.DataFrame([results])
    results_df = pd.concat([existing_results, results_df], ignore_index=True)
    results_df.to_csv(metrics_file, index=False)
else:
    pd.DataFrame([results]).to_csv(os.path.join(results_path, "metrics.csv"), index=False)

# ---------------- Visualization ----------------
plot_sets = config.get("plot_sets", [])

# Define colors for cell types (consistent)
colors = [plt.cm.tab10(i % 10) for i in range(len(cell_type_names))]

def get_subset_df(name, B_subset, idx_subset):
    return pd.DataFrame(
        B_subset.detach().cpu().numpy(),   # <-- detach before converting
        index=Y_df.iloc[idx_subset].index,
        columns=cell_type_names
    )


# ground truth dataframe aligned with indices
G_all_df = G_df[cell_type_names]

# plot requested sets
if "train" in plot_sets:
    gt = G_all_df.iloc[train["idx"]]
    pred = get_subset_df("train", model(Y_train, edge_index_train, edge_weight_train, X)[3], train["idx"])
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df, 
                      title="Ground Truth (Train)", 
                      save_path=os.path.join(results_path, "ground_truth_train.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df, 
                      title="SP-VAE-GCN (Train)", 
                      save_path=os.path.join(results_path, "predicted_train.svg"))

if "val" in plot_sets:
    gt = G_all_df.iloc[val["idx"]]
    pred = get_subset_df("val", model(Y_val, edge_index_val, edge_weight_val, X)[3], val["idx"])
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df, 
                      title="Ground Truth (Val)", 
                      save_path=os.path.join(results_path, "ground_truth_val.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df, 
                      title="SP-VAE-GCN (Val)", 
                      save_path=os.path.join(results_path, "predicted_val.svg"))

if "test" in plot_sets:
    gt = G_all_df.iloc[test["idx"]]
    pred = B_df  # already saved earlier
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df, 
                      title="Ground Truth (Test)", 
                      save_path=os.path.join(results_path, "ground_truth_test.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df, 
                      title="SP-VAE-GCN (Test)", 
                      save_path=os.path.join(results_path, "predicted_test.svg"))

if "all" in plot_sets:
    gt = G_all_df.iloc[Y_df.index]
    pred = pd.read_csv(os.path.join(results_path, "all_spots_proportions.csv"), index_col=0)
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df, 
                      title="Ground Truth (All)", 
                      save_path=os.path.join(results_path, "ground_truth_all.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df, 
                      title="SP-VAE-GCN (All)", 
                      save_path=os.path.join(results_path, "predicted_all.svg"))
