import torch
import numpy as np
from scipy.ndimage import variance
from scipy import ndimage

def compute_data_quality_score(embeddings, labels, server_stats):
    # Sharpness
    sharpness = np.mean([variance(ndimage.laplace(e.cpu().numpy())) for e in embeddings])
    # SNR
    defect_mask = labels == 1
    if defect_mask.any():
        mu_defect = embeddings[defect_mask].mean().item()
        mu_bg = embeddings[~defect_mask].mean().item()
        snr = 10 * np.log10((mu_defect + 1e-8) / (mu_bg + 1e-8))
    else:
        snr = 0.0
    # Normalize
    sharp_norm = (sharpness - server_stats['sharp_mean']) / (server_stats['sharp_std'] + 1e-8)
    snr_norm = (snr - server_stats['snr_mean']) / (server_stats['snr_std'] + 1e-8)
    label_consistency = 0.95
    return (sharp_norm + snr_norm + label_consistency) / 3.0

def compute_decoder_reliability_score(model, val_loader, device):
    model.eval()
    recon_losses = []
    synthetic_data = []
    synthetic_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            recon_x, mu, log_var = model(x)
            recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='none').mean(dim=1)
            recon_losses.extend(recon_loss.cpu().numpy())
            z = torch.randn(x.size(0), model.latent_dim).to(device)
            synth_x = model.decode(z, y)
            synthetic_data.append(synth_x.cpu())
            synthetic_labels.append(y.cpu())
    avg_recon_loss = np.mean(recon_losses)
    synth_x = torch.cat(synthetic_data, dim=0).numpy()
    synth_y = torch.cat(synthetic_labels, dim=0).numpy()
    real_x, real_y = next(iter(val_loader))
    real_x, real_y = real_x.numpy(), real_y.numpy()
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(synth_x, synth_y)
    pred_y = clf.predict(real_x)
    synth_f1 = f1_score(real_y, pred_y, average='macro')
    return (1.0 / (1.0 + avg_recon_loss)) * synth_f1

def compute_global_consistency_score(encoder, val_loader, device, latent_dim=128):
    encoder.eval()
    zs = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            mu, log_var = encoder(x)
            z = torch.randn_like(mu) * torch.exp(0.5 * log_var) + mu
            zs.append(z.cpu())
    z_all = torch.cat(zs, dim=0)
    mu = z_all.mean(dim=0)
    sigma = z_all.std(dim=0)
    kl = 0.5 * torch.sum(mu**2 + sigma**2 - 2*torch.log(sigma) - 1)
    return torch.exp(-kl).item()