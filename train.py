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
cell_type_names = X_ref_df.index.tolist()

# build pseudo ground truth
B_pseudo = (
    pseudo_adata.obs.reindex(columns=cell_type_names)
    .fillna(0.0)
    .astype(float)
)

X = torch.tensor(X_ref_df.values, dtype=torch.float32).to(device)

# ---------------- Combine Real and Pseudo Data ----------------
Y_combined_df = pd.concat([Y_df, Y_pseudo], ignore_index=True)
Y_combined = torch.tensor(Y_combined_df.values, dtype=torch.float32).to(device)

G_real_df = pd.read_csv(os.path.join(st_data_dir, "ST_ground_truth.tsv"), sep="\t", index_col=0)
G_real_df = G_real_df.loc[Y_df.index, cell_type_names]

B_combined_df = pd.concat([G_real_df, B_pseudo], ignore_index=True)
B_combined_gt = torch.tensor(B_combined_df.values, dtype=torch.float32).to(device)


# ---------------- Build the Combined Graph ----------------
num_real_spots = Y_df.shape[0]
num_pseudo_spots = Y_pseudo.shape[0]
num_total_spots = num_real_spots + num_pseudo_spots

A_real = compute_adjacency(coords, sigma=config["adjacency_sigma"], threshold=config["adjacency_threshold"])
A_real_norm = normalize_adjacency(A_real)

A_combined = torch.zeros(num_total_spots, num_total_spots, device=device)

A_combined[:num_real_spots, :num_real_spots] = A_real_norm

# Add self-loops for the pseudo-spots (they are disconnected nodes)
pseudo_indices_loop = torch.arange(num_real_spots, num_total_spots)
A_combined[pseudo_indices_loop, pseudo_indices_loop] = 1.0

print(f"Combined adjacency matrix shape: {A_combined.shape}\n")


# ---------------- STEP 3: Define Splits and Masks ----------------
real_indices = np.arange(num_real_spots)
pseudo_indices = np.arange(num_real_spots, num_total_spots)

split_seed = config["split_seed"] if config["split_seed"] else 42
np.random.seed(split_seed)
np.random.shuffle(real_indices)

val_size = int(num_real_spots * config["val_size"])
test_size = int(num_real_spots * config["split_size"])

val_idx = real_indices[:val_size]
test_idx = real_indices[val_size : val_size + test_size]

val_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
val_mask[val_idx] = True

test_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
test_mask[test_idx] = True

train_pseudo_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
train_pseudo_mask[pseudo_indices] = True

edge_index_combined, edge_weight_combined = get_edges(A_combined)

# ---------------- Model ----------------
input_dim = Y_combined.shape[1]
n_celltypes = X.shape[0]
model = SpatialVAE(input_dim, config["hidden_dim"], config["latent_dim"],
                   n_celltypes, tau=config["tau"], drop=config["dropout_rate"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["l2_lambda"]))

# ---------------- Training ----------------
loss_history, val_loss_history = [], []
best_val_loss, patience_counter = float("inf"), 0
patience = config["patience"]
config_loss_params = {
    "alpha": config["loss_alpha"], "beta": config["loss_beta"],
    "gamma": config["loss_gamma"], "delta": config["loss_delta"],
    "latent_dim": config["latent_dim"]
}

for epoch in range(config["n_epochs"]):
    model.train()
    optimizer.zero_grad()
    
    Y_hat_all, mu_all, logvar_all, B_pred_all = model(Y_combined, edge_index_combined, edge_weight_combined, X)

    loss, _, _, _, _, _ = vae_loss(
        Y_combined[train_pseudo_mask],
        Y_hat_all[train_pseudo_mask],
        mu_all[train_pseudo_mask],
        logvar_all[train_pseudo_mask],
        A_combined[train_pseudo_mask, :][:, train_pseudo_mask], 
        B_pred_all[train_pseudo_mask],
        B_gt=B_combined_gt[train_pseudo_mask],
        **config_loss_params
    )
    
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # --- Validation Phase ---
    model.eval()
    with torch.no_grad():
        val_loss, _, _, _, _, _ = vae_loss(
            Y_combined[val_mask],
            Y_hat_all[val_mask],
            mu_all[val_mask],
            logvar_all[val_mask],
            A_combined[val_mask, :][:, val_mask], 
            B_pred_all[val_mask],
            B_gt=B_combined_gt[val_mask],
            **config_loss_params
        )
        val_loss_history.append(val_loss.item())

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}: Train (Pseudo)={loss.item():.4f} | Val (Real)={val_loss.item():.4f}")

    # --- Early Stopping Logic ---
    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}!")
            break

# restore best model
model.load_state_dict(best_state)

# save loss curves
plt.plot(loss_history, label="Train (Pseudo)")
plt.plot(val_loss_history, label="Val (Real)")
plt.legend()
plt.grid(True)
plt.title("Semi-Supervised Training Loss")
plt.savefig(os.path.join(results_path, "loss_curves.svg"), format="svg", dpi=300)
plt.close()

# ---------------- Test Evaluation ----------------
print("\n### Final Evaluation on the Test Set ###")
model.eval()
with torch.no_grad():
    _ , _, _, B_pred_all = model(Y_combined, edge_index_combined, edge_weight_combined, X)

B_test_pred = B_pred_all[test_mask]
G_test_true = B_combined_gt[test_mask]

# Convert to numpy for evaluation and saving
y_pred = B_test_pred.cpu().numpy()
y_true = G_test_true.cpu().numpy()

# Get the original spot names for the test set using the test_idx array
test_original_names = Y_df.index[test_idx]
B_df = pd.DataFrame(y_pred, index=test_original_names, columns=cell_type_names)
B_df.to_csv(os.path.join(results_path, "predicted_proportions_test.csv"))

# Evaluate metrics
results = evaluate_all(y_true, y_pred)
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
print("\n### Generating Visualizations ###")
plot_sets = config.get("plot_sets", [])
if not plot_sets:
    print("No plot sets specified in config. Skipping visualization.")

# Define colors for cell types (consistent)
colors = [plt.cm.tab10(i % 10) for i in range(len(cell_type_names))]

B_pred_real_df = pd.DataFrame(B_pred_all[:num_real_spots].cpu().detach().numpy(), index=Y_df.index, columns=cell_type_names)
G_all_df = G_real_df # This is your ground truth DataFrame for all real spots


if "train" in plot_sets:
    print("Plotting train set...")
    gt = G_all_df.iloc[train_real_idx]
    pred = B_pred_real_df.iloc[train_real_idx]
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df,
                      title="Ground Truth (Train Set)",
                      save_path=os.path.join(results_path, "ground_truth_train.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df,
                      title="SP-VAE-GCN (Train Set)",
                      save_path=os.path.join(results_path, "predicted_train.svg"))

if "val" in plot_sets:
    print("Plotting validation set...")
    gt = G_all_df.iloc[val_idx]
    pred = B_pred_real_df.iloc[val_idx]
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df,
                      title="Ground Truth (Val Set)",
                      save_path=os.path.join(results_path, "ground_truth_val.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df,
                      title="SP-VAE-GCN (Val Set)",
                      save_path=os.path.join(results_path, "predicted_val.svg"))

if "test" in plot_sets:
    print("Plotting test set...")
    gt = G_all_df.iloc[test_idx]
    pred = B_pred_real_df.iloc[test_idx] # This is the same as B_df we created earlier
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df,
                      title="Ground Truth (Test Set)",
                      save_path=os.path.join(results_path, "ground_truth_test.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df,
                      title="SP-VAE-GCN (Test Set)",
                      save_path=os.path.join(results_path, "predicted_test.svg"))

if "all" in plot_sets:
    print("Plotting all real spots...")
    gt = G_all_df
    pred = B_pred_real_df
    draw_pie_hex_grid(gt, cell_type_names, colors, coords_df,
                      title="Ground Truth (All Real Spots)",
                      save_path=os.path.join(results_path, "ground_truth_all.svg"))
    draw_pie_hex_grid(pred, cell_type_names, colors, coords_df,
                      title="SP-VAE-GCN (All Real Spots)",
                      save_path=os.path.join(results_path, "predicted_all.svg"))