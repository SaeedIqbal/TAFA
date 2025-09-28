import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class IndustrialDataset(Dataset):
    def __init__(self, root, dataset_name, client_id, transform=None):
        self.root = os.path.join(root, dataset_name)
        self.client_id = client_id
        self.transform = transform
        # Load image paths and labels (customize per dataset)
        self.samples = self._load_samples()
        
    def _load_samples(self):
        # Implement dataset-specific loading (MVTec, DAGM, KSDD2)
        # Return list of (image_path, label)
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def create_federated_datasets(config):
    """Create 8 heterogeneous clients per dataset."""
    datasets = {}
    for ds_name in config['datasets']:
        clients = []
        for cid in range(config['num_clients']):
            # Apply client-specific augmentations (Q-threat, D-threat)
            transform = get_client_transform(cid, ds_name)
            dataset = IndustrialDataset(
                root=config['data_root'],
                dataset_name=ds_name,
                client_id=cid,
                transform=transform
            )
            # Split into train/val/test
            train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2])
            clients.append({
                'train': DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True),
                'val': DataLoader(val_ds, batch_size=config['batch_size']),
                'test': DataLoader(test_ds, batch_size=config['batch_size'])
            })
        datasets[ds_name] = clients
    return datasets

def get_client_transform(client_id, dataset_name):
    """Inject heterogeneity: noise, blur, domain shift."""
    transforms = []
    if client_id in [0,1,2]:  # Blue tint + high contrast
        transforms.extend([T.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.2, hue=0.1)])
    elif client_id in [3,4,5]:  # Blur + low contrast
        transforms.extend([T.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)), T.ColorJitter(contrast=0.5)])
    elif client_id in [6,7]:   # Lens distortion
        transforms.append(T.RandomPerspective(distortion_scale=0.2, p=1.0))
    
    # Add Q-threat: noise for some clients
    if client_id in [1,4,6]:
        transforms.append(AddGaussianNoise(mean=0., std=0.1))
    
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean