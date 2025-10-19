
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
G_real_df = pd.read_csv(os.path.join(st_data_dir, "ST_ground_truth.tsv"), sep="\t", index_col=0)


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
        gene_names_list = [list(common_genes)[i] for i in indices]
        f.write(f"{ct}: {gene_names_list}\n")

# build marker name list and order it (for subsetting use)
gene_names = [g for g in sc_df.columns if g != "cell_type"]
selected_genes = sorted({ gene_names[i] for idxs in marker_dict.values() for i in idxs })
selected_genes_ordered = [g for g in Y_df.columns if g in selected_genes]

# --- subset to identified marker genes ---
Y_df  = Y_df[selected_genes_ordered].copy()
sc_df = sc_df[selected_genes_ordered + ["cell_type"]].copy()
Y_log = cpm_log1p(Y_df[selected_genes_ordered])               # (spots x genes)

sc_cpm  = cpm(sc_df[selected_genes_ordered])                  # (cells x genes) linear CPM
sc_log  = np.log1p(sc_cpm)                                    # (cells x genes) logCPM
sc_cpm["cell_type"] = sc_df["cell_type"].values
sc_log["cell_type"] = sc_df["cell_type"].values

# ------------- Pseudo Spot Generation -------------
# build AnnData for pseudo generation
sc_adata = sc.AnnData(
    X=sc_cpm[selected_genes_ordered].values,  # linear CPM (not log)
    obs=pd.DataFrame({"cell_type": sc_cpm["cell_type"].astype(str).values},
                     index=sc_cpm.index),
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

# ---  Normalize ---
# --- Pseudo are sums of CPM; re-normalize each pseudo to same library, then log1p ---
pseudo_cpm = pd.DataFrame(pseudo_adata.X, index=pseudo_adata.obs_names, columns=pseudo_adata.var_names)
# re-normalize pseudo CPM rows to the same library size as ST/sc
pseudo_cpm = cpm(pseudo_cpm, target_sum=1e4)
Y_pseudo_log = np.log1p(pseudo_cpm)[selected_genes_ordered]   # logCPM on markers

Y_pseudo_raw = pd.DataFrame(pseudo_adata.X,
                            index=pseudo_adata.obs_names,
                            columns=pseudo_adata.var_names)[selected_genes_ordered]

X_ref_df = sc_log.groupby("cell_type")[selected_genes_ordered].mean()  # (C x G)
cell_type_names = X_ref_df.index.tolist()

scaler = StandardScaler()

Y_z = pd.DataFrame(
    scaler.fit_transform(Y_log[selected_genes_ordered]),
    index=Y_log.index, columns=selected_genes_ordered
)

Y_pseudo_z = pd.DataFrame(
    scaler.transform(Y_pseudo_log),
    index=Y_pseudo_log.index, columns=selected_genes_ordered
)

X_ref_z = pd.DataFrame(
    scaler.transform(X_ref_df[selected_genes_ordered]),
    index=X_ref_df.index, columns=selected_genes_ordered
)

# Pseudo ground truth fractions (already in pseudo_adata.obs)
B_pseudo = (
    pseudo_adata.obs.reindex(columns=cell_type_names)
    .fillna(0.0)
    .astype(float)
)

B_pseudo_gt = torch.tensor(
    B_pseudo.values, dtype=torch.float32, device=device
)

# Torch tensor for reference (cell-type x genes)
X = torch.tensor(X_ref_z.values, dtype=torch.float32).to(device)

# ---------------- Combine Real and Pseudo Data ----------------
Y_combined_df = pd.concat([Y_z, Y_pseudo_z], ignore_index=True)
Y_combined = torch.tensor(Y_combined_df.values, dtype=torch.float32).to(device)

G_real_df = G_real_df.loc[Y_z.index, cell_type_names]

B_combined_df = pd.concat([G_real_df, B_pseudo], ignore_index=True)
B_combined_gt = torch.tensor(B_combined_df.values, dtype=torch.float32).to(device)


# ---------------- Build the Combined Graph ----------------
num_real_spots   = Y_z.shape[0]
num_pseudo_spots = Y_pseudo_z.shape[0]
num_total_spots  = num_real_spots + num_pseudo_spots


A_real = compute_adjacency(coords, sigma=config["adjacency_sigma"], threshold=config["adjacency_threshold"])
A_real_norm = normalize_adjacency(A_real)

A_combined = torch.zeros(num_total_spots, num_total_spots, device=device)

A_combined[:num_real_spots, :num_real_spots] = A_real_norm

Nr = Y_z.shape[0]
Np = Y_pseudo_z.shape[0]

A_combined = build_hybrid_adjacency(
    Y_real_z=torch.tensor(Y_z.values, dtype=torch.float32, device=device),
    coords_real=coords,                              # (Nr x 2) tensor you already have
    Y_pseudo_z=torch.tensor(Y_pseudo_z.values, dtype=torch.float32, device=device),
    spatial_sigma=float(config["adjacency_sigma"]),
    spatial_threshold=float(config["adjacency_threshold"]),
    k_pseudo=int(config["k_pseudo"]),
    sigma_expr=float(config["sigma_expr"]),
    expr_metric=config.get("expr_distance", "cosine"),
    use_cross=bool(config["use_cross_edges"]),
    k_cross_real=int(config["k_cross_real"]),
    k_cross_pseudo=int(config["k_cross_pseudo"]),
    cross_mode=config.get("cross_mode", "mutual"),
    cross_scale=float(config["cross_scale"]),
    self_loop_eps=float(config.get("self_loop_eps", 1e-3)),
    norm=config.get("adjacency_norm", "sym"),
)

edge_index_combined, edge_weight_combined = get_edges(A_combined)


print(f"Combined adjacency matrix shape: {A_combined.shape}\n")

def _binary_components(A):
    # very small, rough check using BFS on CPU (ok for counts)
    import numpy as np
    import collections
    B = (A.detach().cpu().numpy() > 0).astype(np.uint8)
    N = B.shape[0]; seen = np.zeros(N, dtype=bool); comps = 0
    for s in range(N):
        if seen[s]: continue
        comps += 1
        q = collections.deque([s]); seen[s]=True
        while q:
            u = q.popleft()
            nbrs = np.nonzero(B[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v]=True; q.append(v)
    return comps

with torch.no_grad():
    N = A_combined.size(0); Nr = Y_z.shape[0]; Np = Y_pseudo_z.shape[0]

    deg = A_combined.sum(dim=1).detach().cpu().numpy()
    deg_real   = A_combined[:Nr, :].sum(dim=1).detach().cpu().numpy()
    deg_pseudo = A_combined[Nr:, :].sum(dim=1).detach().cpu().numpy()

    print(f"[Graph] degree all  min/mean/max = {deg.min():.2f}/{deg.mean():.2f}/{deg.max():.2f}")
    print(f"[Graph] degree real mean = {deg_real.mean():.2f} | pseudo mean = {deg_pseudo.mean():.2f}")

    comps = _binary_components(A_combined)
    print(f"[Graph] connected components (binary): {comps}")

    # Rough block means (pre-normalization would be better; but still indicative)
    # If you want exact, compute before normalization inside build_hybrid_adjacency and return stats.
    A = A_combined  # normalized
    mean_RR = (A[:Nr, :Nr][A[:Nr, :Nr] > 0].mean().item()) if (A[:Nr,:Nr] > 0).any() else 0.0
    mean_PP = (A[Nr:, Nr:][A[Nr:, Nr:] > 0].mean().item()) if (A[Nr:,Nr:] > 0).any() else 0.0
    mean_RP = (A[:Nr, Nr:][A[:Nr, Nr:] > 0].mean().item()) if (A[:Nr,Nr:] > 0).any() else 0.0
    print(f"[Graph] mean weight RR={mean_RR:.4f}  PP={mean_PP:.4f}  RP={mean_RP:.4f}")

# Pseudo neighbor purity (argmax over B_pseudo)
# (This is a quick biological sanity; optional.)
if "cell_type" in B_pseudo.columns:
    import numpy as np
    # build a hard label for each pseudo from B_pseudo row argmax
    pseudo_label = torch.tensor(B_pseudo.values.argmax(axis=1), device=device)  # (Np,)
    A_PP_bin = (A_combined[Nr:, Nr:] > 0).float()
    deg_pp = A_PP_bin.sum(dim=1, keepdim=True) + 1e-12
    # neighbor majority agreement rate
    # (for each pseudo, fraction of neighbors sharing its label)
    nbr_idx = (A_PP_bin > 0).nonzero(as_tuple=False)  # [E, 2]
    # build per-node agreement
    agree = torch.zeros(Np, device=device)
    counts = torch.zeros(Np, device=device)
    for (i,j) in nbr_idx:
        agree[i] += float(pseudo_label[i] == pseudo_label[j])
        counts[i] += 1.0
    purity = (agree / torch.clamp(counts, min=1.0)).mean().item()
    print(f"[Bio] pseudo neighbor purity (argmax cell type): {purity:.3f}")


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
train_real_idx = real_indices[val_size + test_size:]

train_real_mask   = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
train_real_mask[train_real_idx] = True

train_pseudo_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
train_pseudo_mask[pseudo_indices] = True

train_unsup_mask  = train_real_mask.clone()
train_unsup_mask[pseudo_indices] = True   # real(train) ∪ pseudo(all)

val_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
val_mask[val_idx] = True

test_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
test_mask[test_idx] = True

train_pseudo_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
train_pseudo_mask[pseudo_indices] = True

edge_index_combined, edge_weight_combined = get_edges(A_combined)


print("\n=== RunInfo ===")
print(f"Real spots:   {Y_z.shape[0]}")
print(f"Pseudo spots: {Y_pseudo_z.shape[0]}")
print(f"Total spots:  {Y_z.shape[0] + Y_pseudo_z.shape[0]}")
print(f"#Markers:     {len(selected_genes_ordered)}")
print(f"#Cell types:  {len(cell_type_names)}")
print(f"Y_z.shape       = {Y_z.shape}")
print(f"Y_pseudo_z.shape= {Y_pseudo_z.shape}")
print(f"X_ref_z.shape   = {X_ref_z.shape}")
print(f"train_real   : {train_real_mask.sum().item()}")
print(f"train_pseudo : {train_pseudo_mask.sum().item()}")
print(f"train_unsup  : {train_unsup_mask.sum().item()}")
print(f"val_real     : {val_mask.sum().item()}")
print(f"test_real    : {test_mask.sum().item()}")


assert list(Y_z.columns) == list(X_ref_z.columns) == selected_genes_ordered, "Marker columns misaligned!"
print("Column alignment (Y_z ~ X_ref_z ~ selected_genes_ordered): OK")

def _scaling_summary(df, label):
    mu = df.mean(axis=0).values
    sd = df.std(axis=0, ddof=0).values
    print(f"[{label}]  mean(|per-gene mean|)={np.mean(np.abs(mu)):.3f}   mean(per-gene std)={np.mean(sd):.3f}")

_scaling_summary(Y_z,        "Y_real(z)")
_scaling_summary(Y_pseudo_z, "Y_pseudo(z)")
_scaling_summary(X_ref_z,    "X_ref(z)")

# pseudo B sums
row_sums = B_pseudo.sum(axis=1).values
print(f"Pseudo B row-sum   min/mean/max = {row_sums.min():.3f}/{row_sums.mean():.3f}/{row_sums.max():.3f}")

# “units check”: do B*X_ref_z ≈ Y_z on a few real spots with known B?
sample_n  = min(5, Y_z.shape[0])
sample_ix = np.random.choice(np.arange(Y_z.shape[0]), size=sample_n, replace=False)
Y_real_sample = Y_z.iloc[sample_ix].values
B_real_sample = G_real_df.iloc[sample_ix][cell_type_names].values
Y_hat_sample  = B_real_sample @ X_ref_z.values
mse_list = ((Y_real_sample - Y_hat_sample)**2).mean(axis=1)
print("Sanity MSE on a few real spots using true B and X_ref (z-scored space):")
# print("  per-spot MSE:", ", ".join([f\"{m:.3f}\" for m in mse_list]))
print(f"  mean MSE    : {mse_list.mean():.3f}\n")

# ---------------- Model ----------------
input_dim = Y_combined.shape[1]
n_celltypes = X.shape[0]
model = SpatialVAE(input_dim, config["hidden_dim"], config["latent_dim"],
                   n_celltypes, tau=config["tau"], drop=config["dropout_rate"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["l2_lambda"]))

# ---------------- Training ----------------
loss_history, val_loss_history = [], []
best_val_loss, patience_counter = float("inf"), 0
patience = int(config["patience"])

# Base (config-driven) weights
config_loss_params = {
    "lambda_recon": float(config["loss_recon"]),
    "lambda_kl": float(config["loss_kl"]),
    "lambda_ot": float(config["loss_ot"]),
    "lambda_smooth": float(config["loss_smooth"]),
    "lambda_deconv": float(config["loss_deconv"]),
    "lambda_ent": float(config["loss_ent"]),
    "lambda_contrast": float(config["loss_contrast"]),
    "latent_dim": int(config["latent_dim"]),
}

# Split into two “views” of the loss:
# 1) UNSUP: used on (real_train ∪ pseudo_all) — no B supervision
unsup_params = dict(config_loss_params)
unsup_params["lambda_deconv"] = 0.0  # no supervised term in unsup pass

# 2) SUP (pseudo-only): only the deconv/entropy/contrast terms (to avoid double-counting)
sup_params = dict(config_loss_params)
sup_params["lambda_recon"]  = 0.0
sup_params["lambda_kl"]     = 0.0
sup_params["lambda_ot"]     = 0.0
sup_params["lambda_smooth"] = 0.0
# keep lambda_deconv / lambda_ent / lambda_contrast from config

print_every = int(config.get("print_every", 25))

# Shapes match?
assert Y_combined[train_pseudo_mask].shape[0] == B_pseudo_gt.shape[0], "Pseudo mask and GT size mismatch."

# Loss weights sanity:
print("Loss weights (unsup):", {k:v for k,v in unsup_params.items() if k.startswith("lambda_")})
print("Loss weights (sup):  ", {k:v for k,v in sup_params.items()  if k.startswith("lambda_")})

for epoch in range(int(config["n_epochs"])):
    model.train()
    optimizer.zero_grad()

    # Forward once (shared graph, shared X_ref)
    Y_hat_all, mu_all, logvar_all, B_pred_all = model(
        Y_combined, edge_index_combined, edge_weight_combined, X
    )

    # --- (1) Unsupervised loss on real_train ∪ all pseudo ---
    unsup_loss, recon_u, kl_u, ot_u, smooth_u, deconv_u, ent_u, contr_u = vae_loss(
        Y_combined[train_unsup_mask],
        Y_hat_all[train_unsup_mask],
        mu_all[train_unsup_mask],
        logvar_all[train_unsup_mask],
        A_combined[train_unsup_mask, :][:, train_unsup_mask],
        B_pred_all[train_unsup_mask],
        B_gt=None,                    
        **unsup_params
    )

    # --- (2) Supervised (deconv) loss on pseudo only ---
    sup_loss, recon_s, kl_s, ot_s, smooth_s, deconv_s, ent_s, contr_s = vae_loss(
        Y_combined[train_pseudo_mask],
        Y_hat_all[train_pseudo_mask],
        mu_all[train_pseudo_mask],
        logvar_all[train_pseudo_mask],
        A_combined[train_pseudo_mask, :][:, train_pseudo_mask],
        B_pred_all[train_pseudo_mask],
        B_gt=B_pseudo_gt,             
        **sup_params
    )

    # Total loss = unsup + sup
    loss = unsup_loss + sup_loss
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # --- Validation: unsupervised ONLY on real(val) ---
    model.eval()
    with torch.no_grad():
        val_loss, recon_v, kl_v, ot_v, smooth_v, deconv_v, ent_v, contr_v = vae_loss(
            Y_combined[val_mask],
            Y_hat_all[val_mask],
            mu_all[val_mask],
            logvar_all[val_mask],
            A_combined[val_mask, :][:, val_mask],
            B_pred_all[val_mask],
            B_gt=None,                
            **unsup_params
        )
        val_loss_history.append(val_loss.item())

    if (epoch + 1) % print_every == 0 or epoch == 0:
        # Optional: show some components to understand dynamics
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train total={loss.item():.4f}  (unsup={unsup_loss.item():.4f}, sup={sup_loss.item():.4f})  "
            f"| Val unsup={val_loss.item():.4f}  "
            f"[recon_u={recon_u.item():.3f}, kl_u={kl_u.item():.3f}, smooth_u={smooth_u.item():.3f}, "
            f"deconv_s={deconv_s.item():.3f}, ent_s={ent_s.item():.3f}, contr_s={contr_s.item():.3f}]"
        )

    # --- Early stopping on val (unsupervised) ---
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