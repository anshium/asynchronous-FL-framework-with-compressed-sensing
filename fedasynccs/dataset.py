import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataset(dataset_name, root='./data'):
    """
    Downloads and returns the Training and Test datasets.
    """
    dataset_name = dataset_name.lower()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST/FashionMNIST stats
    ])

    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    elif dataset_name == 'fashion-mnist' or dataset_name == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    return train_dataset, test_dataset

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partitions the dataset among clients using a Dirichlet distribution to simulate 
    Non-IID data heterogeneity.

    Args:
        dataset: The PyTorch dataset to split.
        num_clients: Number of clients.
        alpha: The concentration parameter for Dirichlet distribution. 
               Lower alpha = Higher Heterogeneity (more Non-IID).
        seed: Random seed for reproducibility.
    
    Returns:
        client_indices: A dictionary {client_id: [list_of_indices]}
    """
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
        
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    # We iterate over each class and split it among clients according to a sampled probability
    for k in range(num_classes):
        
        # Get all indices for class k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample the proportions for this class among clients using Dirichlet
        # proportions shape: [num_clients]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance check: Ensure no client gets empty partition if possible 
        # (though strictly following Dirichlet might allow it, we usually re-normalize)
        proportions = np.array([p * (len(idx_k) < num_clients * 20 + 1) or p for p in proportions])
        proportions = proportions / proportions.sum()
        
        # Calculate split points based on proportions
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split the indices for class k
        idx_batch = np.split(idx_k, proportions)
        
        # Assign to clients
        for i in range(num_clients):
            client_indices[i] += idx_batch[i].tolist()

    return client_indices

def get_dataloader(dataset, indices, batch_size, shuffle=True):
    """
    Helper to create a DataLoader for a specific subset of indices (a client).
    """
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
