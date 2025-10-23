import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

# ---------------------------
# kNN graphs (RAW, no self-loops, no normalization)
# ---------------------------

def cosine_knn_adjacency(X_torch, k=10, thresh=0.0, mutual=True):
    """
    Build a symmetric kNN adjacency using cosine similarity (RAW weights).
    Returns [N,N] with weights in [0,1], no self-loops, not normalized.
    """
    X = F.normalize(X_torch, p=2, dim=1)
    S = X @ X.t()  # cosine similarity [-1, 1]
    N = S.size(0)
    if N <= 1:
        return torch.zeros_like(S)

    # ignore self before kNN
    S = S.clone()
    S.fill_diagonal_(float('-inf'))

    k_eff = min(max(1, k), max(1, N - 1))
    topk_vals, topk_idx = torch.topk(S, k=k_eff, dim=1)
    A = torch.zeros_like(S)
    A.scatter_(1, topk_idx, topk_vals)

    # clip negatives, then threshold if requested
    A = torch.clamp_min(A, 0.0)
    if thresh is not None and thresh > 0:
        A = torch.where(A >= thresh, A, torch.zeros_like(A))

    # symmetrize
    if mutual:
        A = torch.minimum(A, A.t())  # mutual-kNN
    else:
        A = torch.maximum(A, A.t())

    # reset diagonal to 0 (no self-loops at this stage)
    A.fill_diagonal_(0.0)
    return A


def cosine_bipartite_knn(X_real, X_pseudo, k_r2p=10, k_p2r=10, thresh=0.0, mutual=True):
    """
    Bipartite cosine kNN between REAL (N_r,D) and PSEUDO (N_p,D).
    Returns A_rp: [N_r, N_p] RAW weights in [0,1], not normalized.
    If mutual=True, keeps only edges that are among top-k in BOTH directions.
    """
    Xr = F.normalize(X_real, p=2, dim=1)
    Xp = F.normalize(X_pseudo, p=2, dim=1)

    # similarities: real rows, pseudo cols
    S_rp = Xr @ Xp.t()  # [N_r, N_p]

    N_r, N_p = S_rp.shape
    if N_r == 0 or N_p == 0:
        return torch.zeros((N_r, N_p), device=S_rp.device, dtype=S_rp.dtype)

    # real -> pseudo
    k_r2p_eff = min(max(1, k_r2p), N_p)
    vals_r2p, idx_r2p = torch.topk(S_rp, k=k_r2p_eff, dim=1)
    A_rp_rview = torch.zeros_like(S_rp)
    A_rp_rview.scatter_(1, idx_r2p, vals_r2p)

    # pseudo -> real (work on transpose for topk)
    S_pr = S_rp.t()  # [N_p, N_r]
    k_p2r_eff = min(max(1, k_p2r), N_r)
    vals_p2r, idx_p2r = torch.topk(S_pr, k=k_p2r_eff, dim=1)
    A_pr_pview = torch.zeros_like(S_pr)
    A_pr_pview.scatter_(1, idx_p2r, vals_p2r)
    A_rp_pview = A_pr_pview.t()  # back to [N_r, N_p]

    # clip negatives, then threshold
    A_rp_rview = torch.clamp_min(A_rp_rview, 0.0)
    A_rp_pview = torch.clamp_min(A_rp_pview, 0.0)
    if thresh is not None and thresh > 0:
        z = torch.zeros((), device=S_rp.device, dtype=S_rp.dtype)
        A_rp_rview = torch.where(A_rp_rview >= thresh, A_rp_rview, z)
        A_rp_pview = torch.where(A_rp_pview >= thresh, A_rp_pview, z)

    # combine the two directed views
    if mutual:
        A_rp = torch.minimum(A_rp_rview, A_rp_pview)  # keep only mutual matches
    else:
        A_rp = torch.maximum(A_rp_rview, A_rp_pview)  # union

    return A_rp  # [N_r, N_p], RAW weights


# ---------------------------
# Spatial kernels
# ---------------------------

def compute_spatial_adjacency(coords_tensor, sigma=1.0, threshold=1e-4):
    """
    Dense RBF spatial affinity (RAW), weights in (0,1], no self-loops, not normalized.
    """
    sigma = float(sigma)
    threshold = float(threshold)
    d2 = torch.cdist(coords_tensor, coords_tensor, p=2).pow(2)
    A = torch.exp(-d2 / (2.0 * sigma ** 2))
    A[A < threshold] = 0.0
    A.fill_diagonal_(0.0)
    return A


def spatial_knn_rbf(coords_tensor, k=15, sigma=None, mutual=True):
    """
    kNN based on Euclidean distance, weights by RBF with optional data-driven sigma.
    RAW, no self-loops, not normalized.
    """
    d2 = torch.cdist(coords_tensor, coords_tensor, p=2)
    N = d2.size(0)
    if N <= 1:
        return torch.zeros_like(d2)

    # exclude self by setting large distance
    big = torch.finfo(d2.dtype).max
    d2 = d2.clone()
    d2.fill_diagonal_(big)

    k_eff = min(max(1, k), max(1, N - 1))
    # smallest distances -> topk on negative
    neg_d2 = -d2
    vals, idx = torch.topk(neg_d2, k=k_eff, dim=1)
    # derive sigma if not provided
    if sigma is None:
        # median of kNN distances across rows as a robust scale
        knn_d = torch.sqrt(-vals)
        sigma = knn_d.median().item() + 1e-8

    w = torch.exp((vals) / (2.0 * sigma ** 2))  # since vals = -d2
    A = torch.zeros_like(d2)
    A.scatter_(1, idx, w)

    if mutual:
        A = torch.minimum(A, A.t())
    else:
        A = torch.maximum(A, A.t())

    A.fill_diagonal_(0.0)
    return A


# ---------------------------
# Normalization / assembly
# ---------------------------

def normalize_adjacency(A):
    """
    GCN-style: add self-loops and symmetric normalize: D^{-1/2} (A + I) D^{-1/2}.
    Accepts dense [N,N].
    """
    A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    deg = A.sum(dim=1).clamp_min(1e-12)
    D_inv_sqrt = deg.pow(-0.5)
    # efficient diag scaling
    A = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    return A


def build_block_adjacency(A_rr_raw, A_pp_raw, A_rp_raw, alpha_rr=(1.0, 1.0), alpha_cross=1.0):
    """
    Assemble full RAW block matrix from:
      - A_rr_raw: list/tuple of raw [N_r,N_r] adjacencies (e.g., [spatial, expr])
      - A_pp_raw: list/tuple of raw [N_p,N_p] adjacencies (e.g., [expr])
      - A_rp_raw: raw [N_r,N_p] bipartite adjacency
      - alpha_rr : (alpha_spatial, alpha_expr_real)
      - alpha_cross : scalar for real<->pseudo edges

    Returns dense [N_r+N_p, N_r+N_p] RAW adjacency (no self-loops, not normalized).
    """
    # sum weighted real-real
    A_rr_sum = None
    for Ai, wi in zip(A_rr_raw, alpha_rr):
        if Ai is None or wi == 0:
            continue
        A_rr_sum = Ai * wi if A_rr_sum is None else A_rr_sum + wi * Ai
    if A_rr_sum is None:
        A_rr_sum = torch.zeros_like(A_rr_raw[0])

    # sum weighted pseudo-pseudo (single term typical)
    A_pp_sum = None
    # allow alpha on pseudo list too (if caller passes a list of weights)
    alphas_pp = (1.0,) * len(A_pp_raw) if not isinstance(alpha_cross, (list, tuple)) else alpha_cross
    for Ai, wi in zip(A_pp_raw, (1.0,) * len(A_pp_raw)):
        if Ai is None or wi == 0:
            continue
        A_pp_sum = Ai * wi if A_pp_sum is None else A_pp_sum + wi * Ai
    if A_pp_sum is None:
        A_pp_sum = torch.zeros_like(A_pp_raw[0])

    # cross block
    A_rp = (alpha_cross * A_rp_raw) if alpha_cross != 0 else torch.zeros_like(A_rp_raw)

    N_r = A_rr_sum.size(0)
    N_p = A_pp_sum.size(0)
    device = A_rr_sum.device
    dtype = A_rr_sum.dtype

    A_full = torch.zeros((N_r + N_p, N_r + N_p), device=device, dtype=dtype)
    A_full[:N_r, :N_r] = A_rr_sum
    A_full[N_r:, N_r:] = A_pp_sum
    A_full[:N_r, N_r:] = A_rp
    A_full[N_r:, :N_r] = A_rp.t()
    return A_full


def row_normalize(A, eps=1e-8):
    deg = A.sum(dim=1, keepdim=True).clamp_min(eps)
    return A / deg

def get_edges(A):
    return dense_to_sparse(A)

def build_graphs_and_edges(
    Y_real,            # [N_r, D] float32
    Y_pseudo,          # [N_p, D] float32
    Y_real_loc,        # [N_r, 2] or [N_r, d] float32 (coords)
    device,
    gcfg,
):
    """
    Builds combined adjacency with:
      - real-real: spatial + expression (cosine)
      - pseudo-pseudo: expression (cosine)
      - real-pseudo: cosine bipartite
    Then performs ONE GCN-style normalization on the whole matrix.
    Returns: A_combined (normalized dense), edge_index, edge_weight
    """

    # ------- hyperparams with defaults -------
    alpha_spatial      = gcfg.get("alpha_spatial", 1.0)
    alpha_expr_real    = gcfg.get("alpha_expr_real", 1.0)
    alpha_expr_pseudo  = gcfg.get("alpha_expr_pseudo", 1.0)
    alpha_cross        = gcfg.get("alpha_cross", 1.0)

    k_real             = int(gcfg.get("k_real", 10))
    k_pseudo           = int(gcfg.get("k_pseudo", 10))
    k_r2p              = int(gcfg.get("k_cross_real_to_pseudo", 10))
    k_p2r              = int(gcfg.get("k_cross_pseudo_to_real", 10))

    expr_thresh        = float(gcfg.get("expr_thresh", 0.0))
    mutual_knn         = bool(gcfg.get("mutual_knn", True))

    # spatial
    use_spatial_knn    = bool(gcfg.get("use_spatial_knn", True))
    adjacency_sigma    = float(gcfg.get("adjacency_sigma", 1.0))
    adjacency_threshold= float(gcfg.get("adjacency_threshold", 1e-4))

    # ------- move to device just in case -------
    Y_real   = Y_real.to(device)
    Y_pseudo = Y_pseudo.to(device)
    Y_real_loc = Y_real_loc.to(device)

    num_real_spots   = Y_real.size(0)
    num_pseudo_spots = Y_pseudo.size(0)
    num_total_spots  = num_real_spots + num_pseudo_spots

    # ------- 1) Real-real graphs (RAW) -------
    if use_spatial_knn:
        A_real_spatial_raw = spatial_knn_rbf(
            Y_real_loc, k=k_real, sigma=adjacency_sigma if adjacency_sigma > 0 else None, mutual=mutual_knn
        )
    else:
        A_real_spatial_raw = compute_spatial_adjacency(
            Y_real_loc, sigma=adjacency_sigma, threshold=adjacency_threshold
        )

    A_real_expr_raw = cosine_knn_adjacency(
        Y_real, k=k_real, thresh=expr_thresh, mutual=mutual_knn
    )

    # ------- 2) Pseudo-pseudo (RAW) -------
    A_pp_expr_raw = cosine_knn_adjacency(
        Y_pseudo, k=k_pseudo, thresh=expr_thresh, mutual=mutual_knn
    )
    # scale pseudo expr by alpha now (kept separate for clarity)
    A_pp_raw_list = [alpha_expr_pseudo * A_pp_expr_raw]

    # ------- 3) Real-pseudo bipartite (RAW) -------
    A_rp_raw = cosine_bipartite_knn(
        Y_real, Y_pseudo, k_r2p=k_r2p, k_p2r=k_p2r, thresh=expr_thresh, mutual=mutual_knn
    )

    # ------- 4) Assemble full RAW block and normalize once -------
    A_full_raw = build_block_adjacency(
        A_rr_raw=[A_real_spatial_raw, A_real_expr_raw],
        A_pp_raw=A_pp_raw_list,
        A_rp_raw=A_rp_raw,
        alpha_rr=(alpha_spatial, alpha_expr_real),
        alpha_cross=alpha_cross,
    )

    # One GCN-style normalization for the whole graph
    A_combined = normalize_adjacency(A_full_raw)

    # ------- 5) Convert to edges for your GNN -------
    edge_index_combined, edge_weight_combined = get_edges(A_combined)

    # optional: logging
    print(f"Combined adjacency matrix shape: {A_combined.shape} (N_real={num_real_spots}, N_pseudo={num_pseudo_spots})")

    return A_combined, edge_index_combined, edge_weight_combined