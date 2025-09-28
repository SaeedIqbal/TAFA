import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from .trust_score import compute_data_quality_score, compute_decoder_reliability_score, compute_global_consistency_score
from .aggregation import tafa_aggregation
from models.foundation_model import AdaptiveFeatureExtractor
from models.dp_cvae import DPCVAE

class TAFATrainer:
    def __init__(self, config, datasets):
        self.config = config
        self.datasets = datasets  # Dict: {dataset_name: [client0_dict, client1_dict, ...]}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.server_stats = {
            'sharp_mean': 0.0, 'sharp_std': 1.0,
            'snr_mean': 0.0, 'snr_std': 1.0,
            'trust_mean': 0.5, 'trust_std': 0.1
        }
        # Assume we're using the first dataset for training
        self.clients_data = self.datasets[config['datasets'][0]] # 1, 2, ...

    def train(self):
        # Initialize models
        foundation_model = AdaptiveFeatureExtractor().to(self.device)
        global_decoder = self._init_global_decoder()

        for round_idx in range(self.config['num_rounds']):
            client_updates = []
            trust_scores = []

            for client_id in range(self.config['num_clients']):
                client_data = self.clients_data[client_id]
                train_loader = client_data['train']
                val_loader = client_data['val']

                # Step 1: AFE with DP
                afe_model = foundation_model.get_dp_module(self.config['dp_afe'])
                embeddings, labels = self._extract_embeddings(afe_model, train_loader)

                # Step 2: DP-CVAE with GCR
                cvae = DPCVAE(
                    input_dim=embeddings.shape[1],
                    latent_dim=self.config['latent_dim'],
                    hidden_dims=self.config['cvae_hidden']
                ).to(self.device)
                local_decoder, local_encoder = self._train_cvae_dp(cvae, embeddings, labels, val_loader, client_id)

                # Step 2.5: Trust Score
                q_m = compute_data_quality_score(embeddings, labels, self.server_stats)
                r_m = compute_decoder_reliability_score(cvae, val_loader, self.device)
                c_m = compute_global_consistency_score(cvae.encoder, val_loader, self.device, self.config['latent_dim'])
                t_m = (self.config['trust_weights'][0] * q_m +
                       self.config['trust_weights'][1] * r_m +
                       self.config['trust_weights'][2] * c_m)

                client_updates.append(local_decoder.state_dict())
                trust_scores.append(t_m)

            # Step 3: TAFA Aggregation
            global_decoder_state, new_stats = tafa_aggregation(client_updates, trust_scores, self.server_stats)
            global_decoder.load_state_dict(global_decoder_state)
            self.server_stats.update(new_stats)

            # Broadcast global decoder to all clients (in practice, send over network)
            # Here we just keep it in memory

        return global_decoder

    def _extract_embeddings(self, model, data_loader):
        """Extract embeddings using the AFE model."""
        model.eval()
        embeddings_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                embeds = model(images)
                embeddings_list.append(embeds.cpu())
                labels_list.append(labels.cpu())
        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return embeddings, labels

    def _train_cvae_dp(self, model, embeddings, labels, val_loader, client_id):
        """Train DP-CVAE with GCR using Opacus."""
        # Prepare dataset
        dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        # Make model compatible with Opacus
        model = ModuleValidator.fix(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        
        # Setup DP
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=self.config['dp_cvae']['noise_multiplier'],
            max_grad_norm=self.config['dp_cvae']['max_grad_norm'],
            poisson_sampling=False  # For simplicity; set True for strict DP
        )

        model.train()
        gcr_weight = self.config['gcr_weight']
        
        for epoch in range(self.config['local_epochs']):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                recon_x, mu, log_var = model(x, y)
                loss, recon_loss, kld_loss, mmd_loss = model.loss_function(
                    recon_x, x, mu, log_var, gcr_weight=gcr_weight
                )
                loss.backward()
                optimizer.step()

        return model.decoder, model.encoder

    def _init_global_decoder(self):
        """Initialize a global decoder with random weights."""
        decoder = nn.Sequential(
            nn.Linear(self.config['latent_dim'], self.config['cvae_hidden'][-1]),
            nn.ReLU(),
            nn.Linear(self.config['cvae_hidden'][-1], self.config['cvae_hidden'][-2]),
            nn.ReLU(),
            nn.Linear(self.config['cvae_hidden'][-2], self.config['cvae_hidden'][0]),
            nn.ReLU(),
            nn.Linear(self.config['cvae_hidden'][0], 768)  # DINOv2 embedding dim
        ).to(self.device)
        return decoder

    def _update_client_decoder(self, client_id, global_decoder):
        """In a real FL system, this would send the model to the client.
        Here, we do nothing since clients are simulated in-memory."""
        pass