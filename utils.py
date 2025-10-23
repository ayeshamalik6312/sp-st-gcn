import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def as_float(x):
    return x.item() if torch.is_tensor(x) else float(x)

# === PATCH 1.A: helpers ===
import torch.nn.functional as F

# def cosine_knn_adjacency(X_torch, k=10, thresh=0.0, self_loops=False):
#     """
#     Build a symmetric kNN adjacency using cosine similarity.
#     X_torch: [N, D] (torch.float32), assumed standardized already.
#     """
#     X = F.normalize(X_torch, p=2, dim=1)
#     S = X @ X.t()                                     # cosine similarity [-1,1]
#     S.fill_diagonal_(0.0)                              # ignore self before kNN

#     # kNN per row
#     N = S.size(0)
#     topk_vals, topk_idx = torch.topk(S, k=min(k, max(1, N-1)), dim=1, largest=True)
#     A = torch.zeros_like(S)
#     A.scatter_(1, topk_idx, topk_vals)

#     # threshold (clip negatives)
#     if thresh is not None and thresh > 0:
#         A = torch.where(A >= thresh, A, torch.zeros_like(A))
#     else:
#         A = torch.clamp_min(A, 0.0)

#     # symmetrize (max)
#     A = torch.maximum(A, A.t())

#     # optional self-loops
#     if self_loops:
#         A.fill_diagonal_(1.0)

#     return A

# def combine_and_normalize(A_list, w_list):
#     """Weighted sum of adjacencies followed by row-normalization."""
#     A = torch.zeros_like(A_list[0])
#     for Ai, wi in zip(A_list, w_list):
#         if Ai is not None and wi != 0:
#             A = A + wi * Ai
#     # row-normalize to avoid exploding degrees
#     deg = A.sum(dim=1, keepdim=True).clamp_min(1e-8)
#     A = A / deg
#     return A


# def compute_spatial_adjacency(coords_tensor, sigma=1.0, threshold=1e-4):
#     sigma = float(sigma)
#     threshold = float(threshold)
#     dists = torch.cdist(coords_tensor, coords_tensor, p=2) ** 2
#     A = torch.exp(-dists / (2 * sigma ** 2))
#     A[A < threshold] = 0.0
#     return A



# def normalize_adjacency(A):
#     A = A + torch.eye(A.size(0), device=A.device)
#     deg = A.sum(dim=1)
#     D_inv_sqrt = torch.diag(deg.pow(-0.5))
#     return D_inv_sqrt @ A @ D_inv_sqrt


# def cosine_knn_adjacency_raw(X_torch, k=10, thresh=0.0, mutual=True):
#     X = F.normalize(X_torch, p=2, dim=1)
#     S = X @ X.t()
#     S.fill_diagonal_(0.0)
#     N = S.size(0)
#     k_eff = min(k, max(1, N-1))
#     topk_vals, topk_idx = torch.topk(S, k=k_eff, dim=1)
#     A = torch.zeros_like(S)
#     A.scatter_(1, topk_idx, topk_vals)
#     A = torch.clamp_min(A, 0.0) if not (thresh and thresh > 0) else torch.where(A >= thresh, A, A.new_zeros(()))
#     if mutual:
#         A = torch.minimum(A, A.t())  # mutual-kNN
#     else:
#         A = torch.maximum(A, A.t())
#     return A  # no self-loops, no norm

# def spatial_knn_rbf(coords, k=15, sigma=None, mutual=True):
#     # compute kNN on Euclidean distance, weight edges by RBF
#     d2 = torch.cdist(coords, coords).pow(2)
#     N = d2.size(0)
#     k_eff = min(k, max(1, N-1))
#     vals, idx = torch.topk(-d2, k=k_eff, dim=1)  # negative for smallest distances
#     A = torch.zeros_like(d2)
#     if sigma is None:
#         # median of kNN distances per row -> robust sigma
#         knn_d = (-vals).sqrt()
#         sigma = knn_d.median().item() + 1e-8
#     w = torch.exp(-(-vals)/(2.0 * sigma**2))
#     A.scatter_(1, idx, w)
#     if mutual:
#         A = torch.minimum(A, A.t())
#     else:
#         A = torch.maximum(A, A.t())
#     return A  # raw, no self-loops, no norm

# def combine_then_gcn_norm(A_list, w_list):
#     A = sum(wi * Ai for Ai, wi in zip(A_list, w_list) if Ai is not None and wi != 0)
#     # add self-loops and symmetric normalize once
#     return normalize_adjacency(A)

# def get_edges(A):
#     return dense_to_sparse(A)

def custom_split(Y, A, coords, val_size=0.1, test_size=0.2, seed=42):
    n_spots = Y.shape[0]
    indices = np.arange(n_spots)
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=seed)

    def subset(idx):
        return {"Y": Y[idx], "A": A[idx][:, idx], "coords": coords[idx], "idx": idx}

    return subset(train_idx), subset(val_idx), subset(test_idx)


def draw_pie_hex_grid(df, cell_types, colors, coords_df, save_path, title="Spatial Distribution"):
    hex_radius = 1
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    for idx, row in df.iterrows():
        # Convert hex grid to 2D layout
        x, y = coords_df.loc[idx]
        x_coord = x * 3/2 * hex_radius
        y_coord = np.sqrt(3) * (y + 0.5 * (x % 2)) * hex_radius

        # Pie values
        values = row[cell_types].values
        total = values.sum()
        if total == 0:
            continue
        values = values / total  # Ensure sum=1
        start_angle = 0

        for frac, color in zip(values, colors):
            if frac == 0:
                continue
            theta1 = start_angle
            theta2 = start_angle + frac * 360
            wedge = plt.matplotlib.patches.Wedge((x_coord, y_coord), 0.8, theta1, theta2,
                                                 facecolor=color, edgecolor='k', linewidth=0.5)
            ax.add_patch(wedge)
            start_angle = theta2

    # Legend
    for ct, color in zip(cell_types, colors):
        ax.plot([], [], color=color, marker='o', linestyle='None', label=ct)
    ax.legend(title="Cell-type", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, format='svg', dpi=300)
    plt.close()