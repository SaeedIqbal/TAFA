import torch
from config import Config
from client import Client
from server import Server
from datasets import load_datasets  # Assume this loads your data

def main():
    config = Config()
    datasets = load_datasets(config.DATA_ROOT, config.NUM_CLIENTS)
    clients = [Client(i, ds['train'], ds['val'], config) for i, ds in enumerate(datasets)]
    server = Server(config)

    for round in range(config.NUM_ROUNDS):
        client_updates = []
        trust_scores = []

        for client in clients:
            embeddings, labels = client.extract_embeddings()
            decoder_update = client.train_cvae(embeddings, labels)
            trust = client.compute_trust_score(embeddings, labels, server.server_stats)
            client_updates.append(decoder_update)
            trust_scores.append(trust)

        global_decoder = server.aggregate(client_updates, trust_scores)
        for client in clients:
            client.update_global_decoder(global_decoder)

    print("Training completed.")

if __name__ == "__main__":
    main()