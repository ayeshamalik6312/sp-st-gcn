import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logv = GCNConv(hidden_dim, latent_dim)

    def forward(self, Y, edge_index, edge_weight=None):
        H = F.relu(self.conv1(Y, edge_index, edge_weight))
        mu = self.conv_mu(H, edge_index, edge_weight)
        logvar = self.conv_logv(H, edge_index, edge_weight)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, lat_dim, hidden_dim, n_celltypes, tau=1.0, drop=0.2):
        super().__init__()
        self.conv1 = GCNConv(lat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, n_celltypes)
        self.tau = tau
        self.dp = nn.Dropout(drop)

    def forward(self, Z, edge_index, edge_weight=None):
        H = self.dp(Z)
        H = F.relu(self.conv1(H, edge_index, edge_weight))
        B = self.conv2(H, edge_index, edge_weight)
        return F.softmax(B / self.tau, dim=1)

class SpatialVAE(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, n_celltypes, tau=1.0, drop=0.2):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid_dim, lat_dim)
        self.decoder = Decoder(lat_dim, hid_dim, n_celltypes, tau, drop)

    def forward(self, Y, edge_index, edge_weight, X_ref):
        mu, logvar = self.encoder(Y, edge_index, edge_weight)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        Z = mu + eps * std
        B = self.decoder(Z, edge_index, edge_weight)
        Y_hat = B @ X_ref
        return Y_hat, mu, logvar, B
