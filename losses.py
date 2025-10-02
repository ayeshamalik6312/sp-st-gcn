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
