import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)

def vae_loss(Y, Y_hat, mu, logvar, A, B, B_gt=None, 
             alpha=1, beta=1e-4, gamma=0, delta=0, eta=10.0, 
             latent_dim=32):
    
    if alpha != 0:
        recon_loss = F.mse_loss(Y_hat, Y, reduction="mean")
    else:
        recon_loss = torch.tensor(0.0, device=Y.device)

    if beta != 0:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() / latent_dim
    else:
        kl = torch.tensor(0.0, device=Y.device)

    if gamma != 0:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z_q = mu + eps * std
        z_p = torch.randn_like(z_q)
        ot_loss = sinkhorn(z_q, z_p)
    else:
        ot_loss = torch.tensor(0.0, device=Y.device)

    if delta != 0:
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        smooth_loss = torch.trace(B.t() @ L @ B) / B.size(0)
    else:
        smooth_loss = torch.tensor(0.0, device=Y.device)


    if eta != 0 and B_gt is not None:
        B_logs = torch.log(B + 1e-10)
        deconv_loss = F.kl_div(B_logs, B_gt, reduction='batchmean')
    else:
        deconv_loss = torch.tensor(0.0, device=Y.device)

    total = (alpha * recon_loss + 
             beta * kl + 
             gamma * ot_loss + 
             delta * smooth_loss + 
             eta * deconv_loss)

    return total, recon_loss, kl, ot_loss, smooth_loss, deconv_loss