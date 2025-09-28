import yaml
from core.dp_orchestration import TAFATrainer
from datasets.industrial_datasets import create_federated_datasets

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load datasets
    datasets = create_federated_datasets(config)
    
    # Train
    trainer = TAFATrainer(config, datasets)
    global_decoder = trainer.train()
    
    # Save
    torch.save(global_decoder, 'models/tafa_vision_decoder.pth')

if __name__ == "__main__":
    main()