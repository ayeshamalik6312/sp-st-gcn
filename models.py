import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logv = GCNConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Y, edge_index, edge_weight=None):
        H = F.relu(self.conv1(Y, edge_index, edge_weight))
        H = self.dropout(H)
        mu = self.conv_mu(H, edge_index, edge_weight)
        logvar = self.conv_logv(H, edge_index, edge_weight)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, lat_dim, hidden_dim, n_celltypes, tau=1.0, drop=0.2, use_gcn=False):
        super().__init__()
        self.use_gcn = use_gcn
        self.tau = tau
        self.dp = nn.Dropout(drop)
        
        if use_gcn:
            self.conv1 = GCNConv(lat_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, n_celltypes)
        else:
            self.fc1 = nn.Linear(lat_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_celltypes)
            
    def forward(self, Z, edge_index=None, edge_weight=None):
        H = self.dp(Z)
        
        if self.use_gcn:
            H = F.relu(self.conv1(H, edge_index, edge_weight))
            B = self.conv2(H, edge_index, edge_weight)
        else:
            H = F.relu(self.fc1(H))
            B = self.fc2(H)
            
        return F.softmax(B / self.tau, dim=1)


class SpatialVAE(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, n_celltypes, tau=1.0, drop=0.2, use_gcn_decoder=False):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid_dim, lat_dim, dropout=drop)
        self.decoder = Decoder(lat_dim, hid_dim, n_celltypes, tau, drop, use_gcn=use_gcn_decoder)
        
    def reparameterize(self, mu, logvar):
        """Separate reparameterization for clarity and control"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, Y, edge_index, edge_weight, X_ref):
        mu, logvar = self.encoder(Y, edge_index, edge_weight)
        Z = self.reparameterize(mu, logvar)
        
        B = self.decoder(Z, edge_index, edge_weight)
        Y_hat = B @ X_ref
        
        return Y_hat, mu, logvar, B
    
    def encode(self, Y, edge_index, edge_weight=None):
        """Get latent representation without sampling"""
        mu, logvar = self.encoder(Y, edge_index, edge_weight)
        return mu
    
    def decode(self, Z, X_ref, edge_index=None, edge_weight=None):
        """Decode from latent space"""
        B = self.decoder(Z, edge_index, edge_weight)
        Y_hat = B @ X_ref
        return Y_hat, B