import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataset(dataset_name, root='./data'):
    dataset_name = dataset_name.lower()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    elif dataset_name in ['fashion-mnist', 'fashionmnist']:
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    return train_dataset, test_dataset

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)
    
    # Handle different torchvision versions/dataset types
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = np.array(dataset.train_labels)
    else:
        # Fallback: iterate (slow)
        labels = np.array([y for _, y in dataset])
        
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance check to avoid empty partitions in extreme non-IID
        proportions = np.array([p if p > 0 else 1e-5 for p in proportions])
        proportions = proportions / proportions.sum()
        
        # Calculate split indices
        split_indices = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_batch = np.split(idx_k, split_indices)
        
        for i in range(num_clients):
            client_indices[i] += idx_batch[i].tolist()

    return client_indices

def get_dataloader(dataset, indices, batch_size, shuffle=True):
    subset = Subset(dataset, indices)
    # num_workers=0 avoids multiprocessing overhead in simulation
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)