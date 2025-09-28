import torch
import numpy as np
from config import Config

class Server:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.global_decoder = self._init_global_decoder()
        self.server_stats = {
            'sharp_mean': 0.0, 'sharp_std': 1.0,
            'snr_mean': 0.0, 'snr_std': 1.0,
            'trust_mean': 0.5, 'trust_std': 0.1
        }

    def _init_global_decoder(self):
        from models import DPCVAE
        dummy_cvae = DPCVAE(latent_dim=self.config.LATENT_DIM, hidden_dims=self.config.CVAE_HIDDEN)
        return dummy_cvae.decoder.state_dict()

    def aggregate(self, client_updates, trust_scores):
        # Clip trust scores
        lower = self.server_stats['trust_mean'] - 3 * self.server_stats['trust_std']
        upper = self.server_stats['trust_mean'] + 3 * self.server_stats['trust_std']
        clipped = np.clip(trust_scores, lower, upper)
        weights = clipped / np.sum(clipped)
        # Weighted average
        global_update = {}
        for key in client_updates[0].keys():
            global_update[key] = torch.zeros_like(client_updates[0][key])
            for i, local_update in enumerate(client_updates):
                global_update[key] += weights[i] * local_update[key]
        # Update stats
        new_mean = np.mean(clipped)
        new_std = np.std(clipped)
        self.server_stats['trust_mean'] = new_mean
        self.server_stats['trust_std'] = new_std
        return global_update