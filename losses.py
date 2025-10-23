import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------
# Utilities
# ---------------------------

def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if hasattr(x, "values"):  # pandas
        return torch.as_tensor(x.values, device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=dtype)
    raise TypeError(f"Unsupported type {type(x)} for tensor conversion")

def _is_logits_like(M: torch.Tensor, eps=1e-8):
    """Heuristic: logits often have negatives and rows that don't sum to ~1."""
    if not M.dtype.is_floating_point:
        return False
    has_neg = (M < -eps).any().item()  # Use eps threshold instead of strict 0
    if has_neg:
        return True
    # Check if already normalized (more robust check)
    row_sum = M.sum(dim=1)
    is_normalized = torch.allclose(row_sum, torch.ones_like(row_sum), rtol=1e-2, atol=1e-2)
    return not is_normalized

def _as_prob_and_logprob(M: torch.Tensor, eps=1e-8):
    """
    Accepts either probabilities or logits. Returns (P, logP) with rows on the simplex.
    """
    if _is_logits_like(M):
        logP = F.log_softmax(M, dim=1)
        P = torch.exp(logP)  # More stable than logP.exp()
        return P, logP
    # treat as nonnegative weights/probabilities -> normalize
    # Handle negative values more robustly
    P = torch.clamp(M, min=0.0)
    row_sum = P.sum(dim=1, keepdim=True)
    P = P / (row_sum + eps)
    logP = torch.log(P + eps)
    return P, logP

def _safe_reduce(x, reduction: str):
    if reduction == "mean":
        return x.mean() if x.numel() > 0 else torch.zeros((), device=x.device, dtype=x.dtype)
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError("reduction must be 'mean'|'sum'|'none'")

def _mse_ignore_nan(pred, target, reduction="mean"):
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    
    err = (pred[mask] - target[mask]).pow(2)
    if reduction == "mean":
        return err.mean()
    if reduction == "sum":
        return err.sum()
    if reduction == "none":
        out = torch.full_like(pred, float("nan"))
        out[mask] = (pred - target)[mask].pow(2)
        return out
    raise ValueError("reduction must be 'mean'|'sum'|'none'")

# ---------------------------
# Loss components
# ---------------------------

def kl_loss(B_true, B_pred, mu, logvar, *, eps=1e-8, reduction="mean"):
    """
    If B_true or B_pred is None -> KL of latent N(mu, diag(exp(logvar))) to N(0, I).
    Else -> KL(B_true || B_pred) row-wise, accepting probabilities or logits.
    """
    # reference device
    ref = next((t for t in (mu, B_pred, B_true) if isinstance(t, torch.Tensor)), None)
    device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")

    if (B_true is None) or (B_pred is None):
        mu_t = _to_tensor(mu, device)
        logvar_t = _to_tensor(logvar, device)
        # clamp for numerical stability (more conservative bounds)
        logvar_t = torch.clamp(logvar_t, min=-10.0, max=10.0)
        # More stable computation: avoid exp when possible
        kl_node_dim = 0.5 * (mu_t.pow(2) + torch.exp(logvar_t) - logvar_t - 1.0)
        kl_per_node = kl_node_dim.sum(dim=1)
        return _safe_reduce(kl_per_node, reduction)

    # Mixture KL
    P = _to_tensor(B_true, device, dtype=torch.float32)
    Q = _to_tensor(B_pred, device, dtype=torch.float32)

    P, logP = _as_prob_and_logprob(P, eps=eps)
    Q, logQ = _as_prob_and_logprob(Q, eps=eps)

    # More stable: clip probabilities to avoid log(0)
    kl_per_spot = (P * (logP - logQ)).sum(dim=1)
    return _safe_reduce(kl_per_spot, reduction)

def jsd_loss(B_true, B_pred, eps=1e-8, reduction="mean", base=2):
    """
    Compute Jensen-Shannon Divergence between two distributions.
    
    JSD is a symmetric and bounded alternative to KL divergence:
    - JSD(P || Q) = JSD(Q || P)  [symmetric]
    - 0 <= JSD(P || Q) <= 1 (when base=2)  [bounded]
    - JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P + Q)
    
    Args:
        B_true: True distribution [N, K] (probabilities or logits)
        B_pred: Predicted distribution [N, K] (probabilities or logits)
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        base: Logarithm base (2 for bits, e for nats). Default is 2 for [0, 1] range.
        
    Returns:
        JSD value(s) depending on reduction
    """
    if not isinstance(B_true, torch.Tensor) or not isinstance(B_pred, torch.Tensor):
        raise TypeError("jsd_loss expects torch.Tensor inputs")
    
    device = B_true.device
    
    # Convert to probabilities and log probabilities
    P, logP = _as_prob_and_logprob(B_true, eps=eps)
    Q, logQ = _as_prob_and_logprob(B_pred, eps=eps)
    
    # Compute mixture distribution M = 0.5 * (P + Q)
    M = 0.5 * (P + Q)
    logM = torch.log(M + eps)
    
    # Compute KL(P || M) and KL(Q || M)
    kl_pm = (P * (logP - logM)).sum(dim=1)  # [N]
    kl_qm = (Q * (logQ - logM)).sum(dim=1)  # [N]
    
    # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    jsd_per_row = 0.5 * (kl_pm + kl_qm)
    
    # Convert to specified base if needed (default natural log)
    if base == 2:
        jsd_per_row = jsd_per_row / torch.log(torch.tensor(2.0, device=device))
    elif base != torch.e and base != "e":
        jsd_per_row = jsd_per_row / torch.log(torch.tensor(float(base), device=device))
    
    return _safe_reduce(jsd_per_row, reduction)

def row_entropy(B, eps=1e-8, reduction="mean"):
    """
    Entropy per row; accepts probabilities or logits.
    """
    if not isinstance(B, torch.Tensor):
        raise TypeError("row_entropy expects a torch.Tensor")
    P, logP = _as_prob_and_logprob(B, eps=eps)
    H = -(P * logP).sum(dim=1)
    return _safe_reduce(H, reduction)

@torch.no_grad()
def _median_safe(x):
    """Safely compute median, handling edge cases"""
    if x.numel() == 0:
        return None
    m = torch.median(x)
    return m.item() if torch.isfinite(m) and (m > 0) else None

def sinkhorn_loss(B_pred, B_true, epsilon=0.1, max_iter=200, tol=1e-3,
                  p=2, scale_cost=True, dtype=torch.float64):
    """
    Entropic OT (balanced) with log-domain Sinkhorn. Accepts probs or logits.
    Returns scalar <P, C>. Differentiable w.r.t. B_pred and B_true.
    """
    if not isinstance(B_pred, torch.Tensor) or not isinstance(B_true, torch.Tensor):
        raise TypeError("sinkhorn_loss expects torch.Tensor inputs")

    # Handle empty inputs
    if B_pred.numel() == 0 or B_true.numel() == 0:
        return torch.zeros((), device=B_pred.device, dtype=B_pred.dtype)

    # Convert to probabilities and then to double for stability
    xP, _ = _as_prob_and_logprob(B_pred)
    yP, _ = _as_prob_and_logprob(B_true)
    x = xP.to(dtype)
    y = yP.to(dtype)

    # Cost matrix (squared p-distance) - more efficient computation
    C = torch.cdist(x, y, p=p).pow(2)

    # Optional scaling to avoid exp underflow
    if scale_cost:
        med = _median_safe(C)
        if med is not None and med > 0:
            C = C / med

    n, m_ = C.shape
    device = C.device

    # Use double precision for numerical stability
    a = torch.full((n,), 1.0 / max(n, 1), device=device, dtype=dtype)
    b = torch.full((m_,), 1.0 / max(m_, 1), device=device, dtype=dtype)

    # Log kernel with eps protection
    eps_safe = max(epsilon, 1e-10)
    logK = -C / eps_safe

    log_u = torch.zeros(n, device=device, dtype=dtype)
    log_v = torch.zeros(m_, device=device, dtype=dtype)
    
    # Use log1p for better numerical stability
    log_a = torch.log(a)
    log_b = torch.log(b)

    # Iterate with better convergence check
    for it in range(max_iter):
        log_u_prev = log_u.clone()
        
        # Update u
        log_u = log_a - torch.logsumexp(logK + log_v[None, :], dim=1)
        
        # Update v
        log_v = log_b - torch.logsumexp(logK.T + log_u[None, :], dim=1)

        # Check convergence on both u and v
        u_diff = torch.max(torch.abs(log_u - log_u_prev))
        if u_diff < tol:
            break

    # Transport plan
    Pmat = torch.exp(logK + log_u[:, None] + log_v[None, :])

    # Compute loss and convert back to original dtype
    loss = (Pmat * C).sum().to(B_pred.dtype)
    return loss

# ---------------------------
# Composite VAE loss
# ---------------------------

def vae_loss(
    Y_all, Y_hat_all,
    mu, logvar,
    A_all,
    B_all, B_pseudo_gt,
    masks,
    lambda_recon=1.0,
    lambda_kl_lat=1e-4,
    lambda_kl_mix=1e-4,
    lambda_smooth=0.0,
    lambda_ent=0.1,
    lambda_align=0.0,
    lambda_l1=0.0,
    evaluate=False,
):
    """
    Composite objective with:
      - Reconstruction (MSE, NaN-safe)
      - KL (latent)
      - KL (mixture) on pseudo labels
      - Graph smoothness on B
      - Entropy regularizer on B
      - OT alignment between pseudo and real B
    Accepts B_* as probabilities or logits (auto-detected).
    """
    eps = 1e-10
    device = Y_all.device
    dtype = Y_all.dtype

    # Cache mask validity checks
    def _has_any(key):
        m = masks.get(key, None)
        return (m is not None) and torch.is_tensor(m) and (m.sum().item() > 0)
    
    # Pre-compute which masks are valid to avoid redundant checks
    has_train = _has_any('train')
    has_val = _has_any('val')
    has_pseudo = _has_any('pseudo')
    has_train_real = _has_any('train_real')

    # Helper to create zero tensor
    def _zero():
        return torch.zeros((), device=device, dtype=dtype)

    # 1) Reconstruction
    recon_loss = _zero()
    if lambda_recon != 0:
        if not evaluate and has_train:
            mask = masks['train']
            n_samples = mask.sum().item()
            if n_samples > 0:
                # recon_loss = F.gaussian_nll_loss(
                #     Y_hat_all[mask], 
                #     Y_all[mask], 
                #     # torch.ones_like(Y_all[mask]),
                #     logvar.exp().mean().expand_as(Y_all[mask]), 
                #     reduction='sum'
                # ) / n_samples
                recon_loss = _mse_ignore_nan(
                    Y_hat_all[mask], 
                    Y_all[mask], 
                    reduction='mean'
                )
        elif evaluate and has_val:
            mask = masks['val']
            n_samples = mask.sum().item()
            if n_samples > 0:
                # recon_loss = F.gaussian_nll_loss(
                #     Y_hat_all[mask], 
                #     Y_all[mask], 
                #     # torch.ones_like(Y_all[mask]),
                #     logvar.exp().mean().expand_as(Y_all[mask]), 
                #     reduction='sum'
                # ) / n_samples
                recon_loss = _mse_ignore_nan(
                    Y_hat_all[mask], 
                    Y_all[mask], 
                    reduction='mean'
                )
    # 2) KL (latent)
    kl_lat = _zero()
    if lambda_kl_lat != 0:
        if not evaluate and has_train:
            mask = masks['train']
            kl_lat = kl_loss(None, None, mu[mask], logvar[mask], eps=eps, reduction="mean")
        elif evaluate and has_val:
            mask = masks['val']
            kl_lat = kl_loss(None, None, mu[mask], logvar[mask], eps=eps, reduction="mean")

    # 3) KL (mixture) on pseudo-labeled subset
    kl_mix = _zero()
    if lambda_kl_mix != 0 and not evaluate and has_pseudo:
        mask = masks['pseudo']
        B_pred_pseudo = B_all[mask]
        # kl_mix = kl_loss(B_pseudo_gt, B_pred_pseudo, mu[mask], logvar[mask], eps=eps, reduction="mean")
        kl_mix = jsd_loss(B_pseudo_gt, B_pred_pseudo, eps=eps, reduction="mean", base=2)

    # 4) Graph smoothness - OPTIMIZED to avoid torch.diag
    smooth_loss = _zero()
    if lambda_smooth != 0:
        A = None
        B = None
        
        if not evaluate and has_train:
            mask = masks['train']
            A = A_all[mask][:, mask]
            B = B_all[mask]
        elif evaluate and has_val:
            mask = masks['val']
            A = A_all[mask][:, mask]
            B = B_all[mask]

        if A is not None and B is not None and A.numel() > 0 and B.numel() > 0:
            # Accept B as logits or probs
            P, _ = _as_prob_and_logprob(B, eps=eps)
            
            # Efficient Laplacian computation without constructing full diagonal matrix
            # trace(P^T L P) = trace(P^T D P) - trace(P^T A P)
            #                = sum_i deg_i * P[i]^T P[i] - trace(P^T A P)
            deg = A.sum(dim=1)
            
            # First term: diagonal contribution
            diag_term = (deg.unsqueeze(1) * P * P).sum()
            
            # Second term: off-diagonal (adjacency) contribution
            # trace(P^T A P) = sum_ij A[i,j] * P[i]^T P[j]
            AP = A @ P
            adj_term = (P * AP).sum()
            
            smooth_loss = (diag_term - adj_term) / max(P.size(0), 1)

    # 5) Entropy (maximize entropy => subtract in total)
    entropy_loss = _zero()
    if lambda_ent != 0:
        if not evaluate and has_train:
            entropy_loss = row_entropy(B_all[masks['train']], eps=eps, reduction="mean")
        elif evaluate and has_val:
            entropy_loss = row_entropy(B_all[masks['val']], eps=eps, reduction="mean")

    # 6) OT alignment (pseudo vs real)
    align_loss = _zero()
    if lambda_align != 0 and not evaluate and has_pseudo and has_train_real:
        B_real = B_all[masks['train_real']]
        B_pred_pseudo = B_all[masks['pseudo']]
        # Only compute if both have samples
        if B_real.size(0) > 0 and B_pred_pseudo.size(0) > 0:
            align_loss = sinkhorn_loss(B_pred_pseudo, B_real)

    # 7) l1 sparsity on B (optional)
    l1_loss = _zero()
    if lambda_l1 != 0 and not evaluate and has_train:
        mask = masks['train']
        l1_loss = B_all[mask].sum(dim=1).mean()  # Already sums to 1, so this pushes toward sparse

    total = (
        lambda_recon * recon_loss
        + lambda_kl_lat * kl_lat
        + lambda_kl_mix * kl_mix
        + lambda_smooth * smooth_loss
        - lambda_ent * entropy_loss
        + lambda_align * align_loss
        + lambda_l1 * l1_loss
    )

    return {
        'total': total,
        'recon': recon_loss,
        'kl_lat': kl_lat,
        'kl_mix': kl_mix,
        'smooth': smooth_loss,
        'entropy': entropy_loss,
        'align': align_loss,
        'l1': l1_loss,
    }