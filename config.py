import torch

class Config:
    # Paths
    DATA_ROOT = "/home/phd/datasets/"
    DATASETS = ["mvtec", "mtd", "ksdd2"]

    
    # Federation
    NUM_CLIENTS = 8 # 13
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    
    # Model
    LATENT_DIM = 128
    CVAE_HIDDEN = [512, 256, 128]
    GCR_WEIGHT = 0.5
    TRUST_WEIGHTS = [1/3, 1/3, 1/3]  # alpha, beta, gamma
    
    # DP
    DP_AFE = {'epsilon': 5.0, 'delta': 1e-3, 'max_grad_norm': 2.0}
    DP_CVAE = {'epsilon': 1.0, 'delta': 1e-5, 'max_grad_norm': 1.0, 'noise_multiplier': 1.2}
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')