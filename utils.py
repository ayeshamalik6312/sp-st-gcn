import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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