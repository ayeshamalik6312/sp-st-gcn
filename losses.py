import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)

def vae_loss(Y, Y_hat, mu, logvar, A, B,
             alpha=1, beta=1e-4, gamma=0, delta=0, latent_dim=32):
    if alpha != 0:
        recon_loss = F.mse_loss(Y_hat, Y, reduction="mean")
    else:
        recon_loss = 0

    if beta != 0:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() / latent_dim
    else:
        kl = 0

    if gamma != 0:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z_q = mu + eps * std
        z_p = torch.randn_like(z_q)
        ot_loss = sinkhorn(z_q, z_p)
    else:
        ot_loss = 0

    if delta != 0:
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        smooth_loss = torch.trace(B.t() @ L @ B) / B.size(0)
    else:
        smooth_loss = 0

    total = alpha * recon_loss + beta * kl + gamma * ot_loss + delta * smooth_loss
    return total, recon_loss, kl, ot_loss, smooth_loss


# Add to losses.py (keep existing imports and functions)

def weighted_reconstruction_loss(Y_true, Y_pred, B, marker_weight_matrix, 
                                 base_weight=1.0):
    """
    Weighted MSE that emphasizes marker genes based on predicted proportions
    
    Args:
        Y_true: [batch, genes] actual expression
        Y_pred: [batch, genes] predicted expression  
        B: [batch, celltypes] predicted proportions
        marker_weight_matrix: [celltypes, genes] marker weights
        base_weight: weight for non-marker genes
    
    The idea: if a spot has high proportion of cell type X,
    we care more about accurately predicting X's marker genes
    """
    batch_size, n_genes = Y_true.shape
    
    # Compute gene weights for each spot based on its composition
    # B: [batch, celltypes], marker_weight_matrix: [celltypes, genes]
    # Result: [batch, genes]
    dynamic_weights = torch.matmul(B, marker_weight_matrix)
    
    # Normalize weights to have mean=1 (prevents loss scale issues)
    dynamic_weights = dynamic_weights / dynamic_weights.mean()
    
    # Weighted MSE
    mse = (Y_true - Y_pred) ** 2
    weighted_mse = mse * dynamic_weights
    
    return weighted_mse.mean()

def sparsity_loss(B, sparsity_type='l1', sparsity_weight=0.1, target_entropy=None):
    """
    Encourage sparse cell type predictions
    
    Args:
        B: [batch, celltypes] predicted proportions (already sum to 1)
        sparsity_type: 'l1', 'entropy', or 'both'
        sparsity_weight: scaling factor
        target_entropy: if using entropy, target value (lower = sparser)
    
    Biological reasoning: Most spots contain 2-4 dominant cell types,
    not uniform mixture of all 23 types
    """
    loss = 0
    eps = 1e-8
    
    if sparsity_type in ['l1', 'both']:
        # L1 penalty on the proportions
        # This pushes small values toward 0
        l1_loss = torch.abs(B).mean()
        loss += l1_loss
    
    if sparsity_type in ['entropy', 'both']:
        # Entropy penalty - lower entropy = more peaked distribution
        entropy = -(B * torch.log(B + eps)).sum(dim=1).mean()
        
        if target_entropy is None:
            # Default target: ~3 cell types with equal weight
            # H = -3 * (1/3) * log(1/3) ≈ 1.1
            target_entropy = 1.1
        
        entropy_penalty = torch.abs(entropy - target_entropy)
        loss += entropy_penalty
    
    return sparsity_weight * loss