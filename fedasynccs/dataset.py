import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset

class FederatedDataset:
    def __init__(self, data_dir, num_clients=3, batch_size=200, mode="mnist"):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.mode = mode
        
        self.train_data = None
        self.val_data = None
        self._prepare_data()

    def _prepare_data(self):
        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.mode == "random":
            self.train_data = TensorDataset(torch.randn(2000, 10), torch.randint(0, 2, (2000,)))
            self.val_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        elif self.mode == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.train_data = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
            self.val_data = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
        elif self.mode == "fashion":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_data = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=transform)
            self.val_data = datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=transform)

    def get_client_loader(self, client_id):
        # Simple iid split
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError("Invalid client_id")
            
        total_len = len(self.train_data)
        samples_per_client = total_len // self.num_clients
        indices = list(range(total_len))
        
        start = client_id * samples_per_client
        end = (client_id + 1) * samples_per_client
        
        subset = Subset(self.train_data, indices[start:end])
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)