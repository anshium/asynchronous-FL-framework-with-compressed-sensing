import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")

class Model(nn.Module):
    def __init__(self, batch_size=32, device=None):
        super(Model, self).__init__()
        self.device = DEVICE if device is None else device
        
        # Architecture for 28x28 inputs (MNIST/FashionMNIST)
        # Conv1: 1 -> 8 filters. Output: 24x24 -> Pool -> 12x12
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)   
        # Conv2: 8 -> 16 filters. Output: 8x8 -> Pool -> 4x4
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)  
        
        # Flatten: 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(256, 64)   
        self.fc2 = nn.Linear(64, 10)    

        self.to(self.device)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return raw logits for use with CrossEntropyLoss
        return x

def get_model(dataset_name):
    if dataset_name.lower() in ['mnist', 'fashion-mnist', 'fashionmnist']:
        return Model(batch_size=32, device=DEVICE)
    else:
        raise ValueError(f"Model for dataset {dataset_name} not implemented.")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")