import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from opacus.grad_sample import GradSampleModule

class DPCVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128, hidden_dims=[512, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        dec_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def decode(self, z, y=None):
        return self.decoder(z)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, y=None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, gcr_weight=0.5):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # GCR: MMD loss to N(0,I)
        z = self.reparameterize(mu, log_var)
        mmd_loss = self.mmd_loss(z, torch.randn_like(z))
        
        total_loss = recon_loss + kld_loss + gcr_weight * mmd_loss
        return total_loss, recon_loss, kld_loss, mmd_loss
    
    def mmd_loss(self, x, y, kernel='rbf'):
        """Maximum Mean Discrepancy loss."""
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        
        XX, YY, XY = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(zz)
        if kernel == "rbf":
            bandwidth = [10, 15, 20, 50]
            for a in bandwidth:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)
        elif kernel == "poly":
            dxx = torch.clamp(dxx, 0, 1000)
            dyy = torch.clamp(dyy, 0, 1000)
            dxy = torch.clamp(dxy, 0, 1000)
            XX = (4 + dxx) ** 0.5
            YY = (4 + dyy) ** 0.5
            XY = (4 + dxy) ** 0.5
            
        return XX.mean() + YY.mean() - 2. * XY.mean()
    
    def get_dp_module(self, config):
        """Wrap encoder/decoder for DP-SGD."""
        dp_encoder = GradSampleModule(nn.Sequential(self.encoder, self.fc_mu, self.fc_var))
        dp_decoder = GradSampleModule(self.decoder)
        return dp_encoder, dp_decoder