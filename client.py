import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from models import AdaptiveFeatureExtractor, DPCVAE
from utils import compute_data_quality_score, compute_decoder_reliability_score, compute_global_consistency_score
from config import Config

class Client:
    def __init__(self, client_id, train_loader, val_loader, config):
        self.id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        self.foundation_model = AdaptiveFeatureExtractor().to(self.device)
        self.cvae = None
        self.global_decoder = None

    def extract_embeddings(self):
        self.foundation_model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for images, labs in self.train_loader:
                images = images.to(self.device)
                embeds = self.foundation_model(images)
                embeddings.append(embeds.cpu())
                labels.append(labs.cpu())
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

    def train_cvae(self, embeddings, labels):
        dataset = torch.utils.data.TensorDataset(embeddings, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.cvae = DPCVAE(input_dim=embeddings.shape[1], latent_dim=self.config.LATENT_DIM, hidden_dims=self.config.CVAE_HIDDEN).to(self.device)
        self.cvae = ModuleValidator.fix(self.cvae)
        optimizer = torch.optim.SGD(self.cvae.parameters(), lr=1e-3)
        privacy_engine = PrivacyEngine()
        self.cvae, optimizer, loader = privacy_engine.make_private(
            module=self.cvae,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=self.config.DP_CVAE['noise_multiplier'],
            max_grad_norm=self.config.DP_CVAE['max_grad_norm']
        )
        self.cvae.train()
        for _ in range(self.config.LOCAL_EPOCHS):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                recon_x, mu, log_var = self.cvae(x, y)
                loss = self.cvae.loss_function(recon_x, x, mu, log_var, gcr_weight=self.config.GCR_WEIGHT)
                loss.backward()
                optimizer.step()
        return self.cvae.decoder.state_dict()

    def compute_trust_score(self, embeddings, labels, server_stats):
        q_m = compute_data_quality_score(embeddings, labels, server_stats)
        r_m = compute_decoder_reliability_score(self.cvae, self.val_loader, self.device)
        c_m = compute_global_consistency_score(self.cvae.encoder, self.val_loader, self.device, self.config.LATENT_DIM)
        t_m = (self.config.TRUST_WEIGHTS[0] * q_m +
               self.config.TRUST_WEIGHTS[1] * r_m +
               self.config.TRUST_WEIGHTS[2] * c_m)
        return t_m

    def update_global_decoder(self, global_decoder_state):
        self.global_decoder = global_decoder_state