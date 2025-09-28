import torch
import torch.nn as nn
from torch.distributions import Normal
import timm

class AdaptiveFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.trainable_head = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.trainable_head(features)

class DPCVAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=128, hidden_dims=[512, 256]):
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
        mmd_loss = self.mmd_loss(self.reparameterize(mu, log_var), torch.randn_like(mu))
        return recon_loss + kld_loss + gcr_weight * mmd_loss

    def mmd_loss(self, x, y, kernel='rbf'):
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
        return XX.mean() + YY.mean() - 2. * XY.mean()