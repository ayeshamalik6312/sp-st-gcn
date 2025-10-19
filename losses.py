import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)

def vae_loss(Y, Y_hat, mu, logvar, A, B, B_gt=None, 
             lambda_recon=1, lambda_kl=1e-4, lambda_ot=0, lambda_smooth=0, lambda_deconv=10.0, lambda_ent=0.1, lambda_contrast=0.1,
             latent_dim=32):
    
    if lambda_recon != 0:
        recon_loss = F.mse_loss(Y_hat, Y, reduction="mean")
    else:
        recon_loss = torch.tensor(0.0, device=Y.device)

    if lambda_kl != 0:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() / latent_dim
    else:
        kl = torch.tensor(0.0, device=Y.device)

    if lambda_ot != 0:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z_q = mu + eps * std
        z_p = torch.randn_like(z_q)
        ot_loss = sinkhorn(z_q, z_p)
    else:
        ot_loss = torch.tensor(0.0, device=Y.device)

    if lambda_smooth != 0:
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        smooth_loss = torch.trace(B.t() @ L @ B) / B.size(0)
    else:
        smooth_loss = torch.tensor(0.0, device=Y.device)


    if lambda_deconv != 0 and B_gt is not None:
        B_logs = torch.log(B + 1e-10)
        deconv_loss = F.kl_div(B_logs, B_gt, reduction='batchmean')
    else:
        deconv_loss = torch.tensor(0.0, device=Y.device)

    entropy = -torch.sum(B * torch.log(B + 1e-10), dim=1).mean()
    sorted_B, _ = torch.sort(B, dim=1, descending=True)
    contrast_loss = -torch.mean(sorted_B[:, 0] - sorted_B[:, 1])

    total = (lambda_recon * recon_loss + lambda_kl * kl + lambda_ot * ot_loss +
            lambda_smooth * smooth_loss + lambda_deconv * deconv_loss +
            lambda_ent * entropy + lambda_contrast * contrast_loss)


    # total = (lambda_recon * recon_loss + 
    #          lambda_kl * kl + 
    #          lambda_ot * ot_loss + 
    #          lambda_smooth * smooth_loss + 
    #          lambda_deconv * deconv_loss)

    return total, recon_loss, kl, ot_loss, smooth_loss, deconv_loss, entropy, contrast_loss