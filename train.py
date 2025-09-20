#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse, sys, os, yaml

from utils import *
from marker_genes import *
from losses import *
from models import SpatialVAE
from sklearn.preprocessing import StandardScaler
from metrics import evaluate_all
from sklearn.metrics import mean_absolute_error


# ---------- helpers ----------
def scalar(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, np.generic):
        return float(x)
    if torch.is_tensor(x):
        x = x.detach()
        return x.mean().item()
    # fallback
    return float(x)

# ---------- args / config ----------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config (default: ./config.yaml)")
args, _ = parser.parse_known_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

st_data_dir = config["st_data_dir"]
sc_data_dir = config["sc_data_dir"]
results_path = config["results_path"]
os.makedirs(results_path, exist_ok=True)

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

# align genes
common_genes = Y_df.columns.intersection(sc_df.columns)
Y_df = Y_df[common_genes]
sc_df = sc_df[list(common_genes) + ["cell_type"]]

# -------------- Find marker genes -------------
marker_dict, marker_scores = find_marker_genes(
    sc_df, 
    n_markers_per_type=int(float(config.get("n_markers_per_type", 4))),
    method='fold_change'
)

# Save marker genes info for analysis
marker_scores.to_csv(os.path.join(results_path, "marker_scores.csv"))
with open(os.path.join(results_path, "marker_genes.txt"), 'w') as f:
    for ct, indices in marker_dict.items():
        gene_names_list = [list(common_genes)[i] for i in indices[:5]]
        f.write(f"{ct}: {gene_names_list}\n")
        
# --------------Normalize 
scaler_Y = StandardScaler()
Y_df[common_genes] = scaler_Y.fit_transform(Y_df[common_genes])
scaler_X = StandardScaler()
sc_df[common_genes] = scaler_X.fit_transform(sc_df[common_genes])

X_ref_df = sc_df.groupby("cell_type").mean()
X_ref_df[common_genes] = scaler_X.fit_transform(X_ref_df[common_genes])

Y = torch.tensor(Y_df.values, dtype=torch.float32).to(device)
X = torch.tensor(X_ref_df.values, dtype=torch.float32).to(device)
cell_type_names = X_ref_df.index.tolist()

print(f"Data loaded: ST spots={Y.shape[0]}, genes={Y.shape[1]}, cells={sc_df.shape[0]}, cell types={len(cell_type_names)}")

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
train, val, test = custom_split(
    Y, A, coords,
    val_size=config["val_size"],
    test_size=config["split_size"],
    seed=split_seed
)
Y_train, A_train, coords_train = train["Y"], train["A"], train["coords"]
Y_val, A_val, coords_val = val["Y"], val["A"], val["coords"]
Y_test, A_test, coords_test = test["Y"], test["A"], test["coords"]

# ---------------- Pseudo-spots  ----------------
if config.get("use_pseudo_spots", False):
    print("Pseudo spots on!")
    Y_pseudo, B_pseudo = create_pseudo_spots_torch(
        X_ref=X,  # your standardized reference [n_celltypes, genes]
        n_spots=int(config.get("pseudo_n_spots", 1000)),
        max_types=int(config.get("pseudo_max_types", 3)),
        noise_sigma=float(config.get("pseudo_noise_sigma", 0.1)),
        dirichlet_alpha=float(config.get("pseudo_dirichlet_alpha", 1.0)),
        seed=int(config.get("pseudo_seed", 0)) if config.get("pseudo_seed") is not None else None,
        device=device
    )

    # Identity adjacency so pseudo-spots are independent (disjoint subgraph)
    A_pseudo = torch.eye(Y_pseudo.size(0), device=device)
    A_pseudo = normalize_adjacency(A_pseudo)

    # Concatenate onto the training graph as a block-diagonal
    # (real train) ⊕ (pseudo spots)
    A_train = torch.block_diag(A_train, A_pseudo)
    Y_train = torch.cat([Y_train, Y_pseudo], dim=0)

    # Optional: if you ever want to track pseudo ground-truth B during training logs
    # you can stash B_pseudo for analysis; training itself doesn't need it.

# Rebuild training edges AFTER augmentation
edge_index_train, edge_weight_train = get_edges(A_train)
edge_index_val, edge_weight_val = get_edges(A_val)
edge_index_test, edge_weight_test = get_edges(A_test)

# ---------------- Model ----------------
input_dim = Y.shape[1]
n_celltypes = X.shape[0]
model = SpatialVAE(
    input_dim, config["hidden_dim"], config["latent_dim"],
    n_celltypes, tau=config["tau"], drop=config["dropout_rate"]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["lr"]),
    weight_decay=float(config["l2_lambda"])
)

# ---------- Marker Weight Matrix ----------
marker_weight_matrix = create_marker_weight_matrix(
    marker_dict, 
    input_dim, 
    n_celltypes, 
    cell_type_names,
    base_weight=1.0,
    marker_weight=float(config.get("marker_weight", 2.0))
).to(device)

# ---------------- Training ----------------
loss_history, val_loss_history = [], []
best_val_loss, patience_counter = float("inf"), 0
patience = config["patience"]
best_state = model.state_dict()  # safe default so load_state_dict always works

# Identify which rows are pseudo (computed once, outside the loop)
n_real = train["Y"].size(0)
is_pseudo = torch.zeros(Y_train.size(0), dtype=torch.bool, device=Y_train.device)
if config.get("use_pseudo_spots", False):
    is_pseudo[n_real:] = True

# How strongly to weight pseudo-spot reconstruction
lambda_pseudo = float(config.get("pseudo_recon_weight", 1.5))

for epoch in range(config["n_epochs"]):
    # Train 
    model.train()
    optimizer.zero_grad()
    Y_hat, mu, logvar, B = model(Y_train, edge_index_train, edge_weight_train, X)

    # ----- Reconstruction loss (real vs. pseudo, with extra weight on pseudo) -----
    if config.get("use_weighted_recon", False):
        recon_real = weighted_reconstruction_loss(
            Y_train[~is_pseudo], Y_hat[~is_pseudo], B[~is_pseudo], marker_weight_matrix
        )
        if is_pseudo.any():
            recon_pseudo = weighted_reconstruction_loss(
                Y_train[is_pseudo], Y_hat[is_pseudo], B[is_pseudo], marker_weight_matrix
            )
        else:
            recon_pseudo = torch.tensor(0.0, device=Y_train.device)
    else:
        recon_real = F.mse_loss(Y_hat[~is_pseudo], Y_train[~is_pseudo])
        recon_pseudo = F.mse_loss(Y_hat[is_pseudo], Y_train[is_pseudo]) if is_pseudo.any() \
                    else torch.tensor(0.0, device=Y_train.device)

    # NORMALIZED combo so train loss is comparable to val (which has no pseudo)
    recon_loss = (recon_real + lambda_pseudo * recon_pseudo) / (1.0 + lambda_pseudo)

    # KL divergence loss
    if config["loss_beta"] > 0:
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean() / config["latent_dim"]
    else:
        kl_loss = torch.tensor(0.0, device=Y_train.device)

    # Smoothness loss
    if config["loss_delta"] > 0:
        deg = A_train.sum(dim=1)
        L = torch.diag(deg) - A_train
        smooth_loss = torch.trace(B.t() @ L @ B) / B.size(0)
    else:
        smooth_loss = torch.tensor(0.0, device=Y_train.device)

    # Sparsity loss
    sparse_loss = sparsity_loss(
        B,
        sparsity_type=config.get("sparsity_type", "both"),
        sparsity_weight=config.get("sparsity_weight", 0.05)
    )

    # Pseudo B supervision (tiny) to improve correlations
    if config.get("use_pseudo_spots", False) and is_pseudo.any():
        pseudo_sup = F.mse_loss(B[is_pseudo], B_pseudo)
    else:
        pseudo_sup = torch.tensor(0.0, device=Y_train.device)

    lambda_sup = float(config.get("pseudo_supervise_weight", 0.1))

    # Total loss
    total_loss = (config["loss_alpha"] * recon_loss +
                config["loss_beta"]  * kl_loss +
                config["loss_delta"] * smooth_loss +
                sparse_loss +
                lambda_sup           * pseudo_sup)


    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    loss_history.append(scalar(total_loss))


    # ===== Validation =====
    model.eval()
    with torch.no_grad():
        Y_hat_val, mu_val, logvar_val, B_val = model(Y_val, edge_index_val, edge_weight_val, X)

        # Reconstruction loss
        if config.get("use_weighted_recon", False):
            val_recon = weighted_reconstruction_loss(
                Y_val, Y_hat_val, B_val, marker_weight_matrix
            )
        else:
            val_recon = F.mse_loss(Y_hat_val, Y_val)

        # KL divergence loss
        if config["loss_beta"] > 0:
            val_kl = -0.5 * torch.sum(
                1 + logvar_val - mu_val.pow(2) - logvar_val.exp(), dim=1
            ).mean() / config["latent_dim"]
        else:
            val_kl = torch.tensor(0.0, device=Y_val.device)

        # Smoothness loss
        if config["loss_delta"] > 0:
            deg_val = A_val.sum(dim=1)
            L_val = torch.diag(deg_val) - A_val
            val_smooth = torch.trace(B_val.t() @ L_val @ B_val) / B_val.size(0)
        else:
            val_smooth = torch.tensor(0.0, device=Y_val.device)

        # Sparsity loss
        val_sparse = sparsity_loss(
            B_val,
            sparsity_type=config.get("sparsity_type", "both"),
            sparsity_weight=config.get("sparsity_weight", 0.05)
        )

        # Total validation loss
        val_loss = (config["loss_alpha"] * val_recon +
                    config["loss_beta"] * val_kl +
                    config["loss_delta"] * val_smooth +
                    val_sparse)

        val_loss_history.append(scalar(val_loss))

    if (epoch + 1) % 25 == 0:
        with torch.no_grad():
            z_mean = scalar(mu_val.mean())
            z_std = scalar(torch.exp(0.5 * logvar_val).mean())
            eps = 1e-8
            B_entropy = scalar((-(B_val * torch.log(B_val + eps)).sum(dim=1)).mean())
            B_max = scalar(B_val.max(dim=1)[0].mean())

        recon_real_s   = scalar(recon_real)
        recon_pseudo_s = scalar(recon_pseudo)
        pseudo_sup_s = scalar(pseudo_sup)

        print(
            f"Epoch {epoch+1:03d} "
            f"| Train: total={scalar(total_loss):.4f}, recon={scalar(recon_loss):.4f}, kl={scalar(kl_loss):.4f}, smooth={scalar(smooth_loss):.4f}, sparse={scalar(sparse_loss):.4f} "
            f"\n Val: total={scalar(val_loss):.4f}, recon={scalar(val_recon):.4f}, kl={scalar(val_kl):.4f}, smooth={scalar(val_smooth):.4f}, sparse={scalar(val_sparse):.4f} "
            f"\n Latent μ={z_mean:.4f}, σ={z_std:.4f} | Pred entropy={B_entropy:.4f}, max_prop={B_max:.4f}"
            f"\n recon_real={recon_real_s:.4f}, recon_pseudo={recon_pseudo_s:.4f}, pseudo_sup={pseudo_sup_s:.6f,}"
            f"\n"
        )

    # Early stopping
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

# save loss curves (total loss)
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

# ----- Per-cell-type analysis -----
def per_celltype_analysis(G_true, B_pred, cell_type_names):
    print("\nPer-Cell-Type Performance:")
    for i, ct in enumerate(cell_type_names):
        mae_ct = mean_absolute_error(G_true[:, i], B_pred[:, i])
        if np.std(G_true[:, i]) == 0 or np.std(B_pred[:, i]) == 0:
            corr_ct = 0.0
        else:
            corr_ct = float(np.corrcoef(G_true[:, i], B_pred[:, i])[0, 1])
        print(f"{ct:15s} | MAE={mae_ct:.4f} | Corr={corr_ct:.4f}")

per_celltype_analysis(G_test, y_pred, cell_type_names)

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
        B_subset.detach().cpu().numpy(),
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