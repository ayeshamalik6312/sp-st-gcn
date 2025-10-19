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

def compute_adjacency(coords_tensor, sigma=1.0, threshold=1e-4):
    sigma = float(sigma)
    threshold = float(threshold)
    dists = torch.cdist(coords_tensor, coords_tensor, p=2) ** 2
    A = torch.exp(-dists / (2 * sigma ** 2))
    A[A < threshold] = 0.0
    return A


def normalize_adjacency(A):
    A = A + torch.eye(A.size(0), device=A.device)
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt

def custom_split(Y, A, coords, val_size=0.1, test_size=0.2, seed=42):
    n_spots = Y.shape[0]
    indices = np.arange(n_spots)
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=seed)

    def subset(idx):
        return {"Y": Y[idx], "A": A[idx][:, idx], "coords": coords[idx], "idx": idx}

    return subset(train_idx), subset(val_idx), subset(test_idx)

def get_edges(A):
    return dense_to_sparse(A)

import torch
import torch.nn.functional as F

def pairwise_distance(X, Y=None, metric="cosine"):
    """
    X: (N x G), Y: (M x G) or None (then Y=X)
    returns D: (N x M)
    """
    if Y is None: Y = X
    if metric == "cosine":
        Xn = F.normalize(X, p=2, dim=1)
        Yn = F.normalize(Y, p=2, dim=1)
        sim = Xn @ Yn.t()                      # cosine similarity
        D = 1.0 - sim                          # cosine distance
    elif metric == "euclidean":
        # (x - y)^2 = x^2 + y^2 - 2xy
        x2 = (X**2).sum(dim=1, keepdim=True)
        y2 = (Y**2).sum(dim=1, keepdim=True).t()
        D = torch.clamp(x2 + y2 - 2.0 * X @ Y.t(), min=0.0).sqrt()
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    return D

def knn_gaussian_weights(D, k, sigma, mutual=True):
    """
    D: (N x N) distance matrix (self-distances allowed)
    Returns W: (N x N) sparse-like dense with Gaussian weights on kNN edges.
    """
    N = D.size(0)
    # mask self to large distance so it won't be picked
    D = D + torch.eye(N, device=D.device) * 1e6

    # top-k smallest distances per row
    vals, idx = torch.topk(D, k=k, largest=False)  # vals: (N x k), idx: (N x k)

    # convert distances to weights
    w = torch.exp(- (vals**2) / (2.0 * (sigma**2) + 1e-12))  # (N x k)

    # build dense W (you can keep it dense for now)
    W = torch.zeros_like(D)
    row_idx = torch.arange(N, device=D.device).unsqueeze(1).expand_as(idx)
    W[row_idx, idx] = w

    if mutual:
        W = torch.minimum(W, W.t())  # keep weight only if mutual; symmetric & cleaner
    else:
        W = torch.maximum(W, W.t())  # symmetrize by union

    # zero out the large self distance we added earlier
    W.fill_diagonal_(0.0)
    return W

def knn_cross_block(Xr, Xp, k_r2p, k_p2r, sigma, mode="mutual", metric="cosine", scale=0.5):
    """
    Xr: (Nr x G) real features (z-scored)
    Xp: (Np x G) pseudo features (z-scored)
    Returns (A_RP, A_PR) as dense matrices (Nr x Np) and (Np x Nr)
    """
    Nr, Np = Xr.size(0), Xp.size(0)
    Drp = pairwise_distance(Xr, Xp, metric=metric)   # (Nr x Np)
    Dpr = Drp.t()                                    # (Np x Nr)

    # pick top-k per row (smallest distances)
    vals_rp, idx_rp = torch.topk(Drp, k=k_r2p, largest=False)  # for each real → pseudo
    wrp = torch.exp(- (vals_rp**2) / (2.0 * (sigma**2) + 1e-12))
    A_RP = torch.zeros_like(Drp)
    rows_r = torch.arange(Nr, device=Drp.device).unsqueeze(1).expand_as(idx_rp)
    A_RP[rows_r, idx_rp] = wrp

    vals_pr, idx_pr = torch.topk(Dpr, k=k_p2r, largest=False)  # for each pseudo → real
    wpr = torch.exp(- (vals_pr**2) / (2.0 * (sigma**2) + 1e-12))
    A_PR = torch.zeros_like(Dpr)
    rows_p = torch.arange(Np, device=Dpr.device).unsqueeze(1).expand_as(idx_pr)
    A_PR[rows_p, idx_pr] = wpr

    if mode == "mutual":
        # keep only mutual pairs; symmetrize numerically
        mutual_mask_rp = (A_RP > 0) & (A_PR.t() > 0)
        A_RP = torch.where(mutual_mask_rp, torch.maximum(A_RP, A_PR.t()), torch.zeros_like(A_RP))
        A_PR = A_RP.t()
    elif mode == "union":
        # keep union; take max weight for symmetry
        A_sym = torch.maximum(A_RP, A_PR.t())
        A_RP, A_PR = A_sym, A_sym.t()
    else:
        raise ValueError("mode must be 'mutual' or 'union'")

    # scale down cross edges so they support but don’t dominate
    A_RP = A_RP * scale
    A_PR = A_PR * scale
    return A_RP, A_PR

def build_hybrid_adjacency(
    Y_real_z, coords_real, Y_pseudo_z,
    spatial_sigma, spatial_threshold,
    k_pseudo, sigma_expr, expr_metric,
    use_cross, k_cross_real, k_cross_pseudo, cross_mode, cross_scale,
    self_loop_eps=1e-3, norm="sym"
):
    """
    Returns normalized A (Nr+Np x Nr+Np)
    """
    device = Y_real_z.device
    Nr = Y_real_z.size(0)
    Np = Y_pseudo_z.size(0)
    N  = Nr + Np

    # 1) real-real spatial block (keep your current builder)
    A_RR = compute_adjacency(coords_real, sigma=spatial_sigma, threshold=spatial_threshold)  # (Nr x Nr)

    # 2) pseudo-pseudo expr kNN block
    if k_pseudo > 0:
        D_pp = pairwise_distance(Y_pseudo_z, metric=expr_metric)
        A_PP = knn_gaussian_weights(D_pp, k=k_pseudo, sigma=sigma_expr, mutual=True)
    else:
        A_PP = torch.zeros((Np, Np), device=device)

    # 3) cross real↔pseudo (optional)
    if use_cross and (k_cross_real > 0 or k_cross_pseudo > 0):
        A_RP, A_PR = knn_cross_block(
            Y_real_z, Y_pseudo_z,
            k_r2p=k_cross_real, k_p2r=k_cross_pseudo,
            sigma=sigma_expr, mode=cross_mode, metric=expr_metric, scale=cross_scale
        )
    else:
        A_RP = torch.zeros((Nr, Np), device=device)
        A_PR = torch.zeros((Np, Nr), device=device)

    # 4) assemble big A
    A = torch.zeros((N, N), device=device)
    A[:Nr, :Nr] = A_RR
    A[Nr:, Nr:] = A_PP
    A[:Nr, Nr:] = A_RP
    A[Nr:, :Nr] = A_PR

    # 5) tiny self-loops for stability
    if self_loop_eps and self_loop_eps > 0:
        A = A + torch.eye(N, device=device) * self_loop_eps

    # 6) single global normalization
    if norm == "sym":
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.clamp(deg, min=1e-12).pow(-0.5))
        A = D_inv_sqrt @ A @ D_inv_sqrt
    elif norm == "row":
        deg = A.sum(dim=1, keepdim=True)
        A = A / torch.clamp(deg, min=1e-12)
    else:
        pass  # no normalization

    return A


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