import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A robust CNN for Fashion-MNIST aiming for ~95% accuracy.
    Architecture:
    - Conv block 1: 32 filters + BN + ReLU + MaxPool
    - Conv block 2: 64 filters + BN + ReLU + MaxPool
    - Conv block 3: 128 filters + BN + ReLU (No pooling, preserves spatial features)
    - Classifier: Flatten -> Dense(512) -> Dropout -> Dense(10)
    """
    def __init__(self, num_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Block 1: Input (1, 28, 28) -> Output (32, 14, 14)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2: Input (32, 14, 14) -> Output (64, 7, 7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Input (64, 7, 7) -> Output (128, 7, 7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers
        # Flatten size: 128 filters * 7 * 7 image size = 6272
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # Classifier
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        return x

def get_model(dataset_name):
    """
    Factory function to return the appropriate model based on dataset.
    """
    if dataset_name.lower() in ['mnist', 'fashion-mnist', 'fashionmnist']:
        return SimpleCNN(num_channels=1, num_classes=10)
    else:
        raise ValueError(f"Model for dataset {dataset_name} not implemented.")

def get_device():
    """
    Helper to get the most capable device available (CUDA > MPS > CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_step(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test_step(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({acc:.2f}%)\n')
    return acc

if __name__ == "__main__":

    import dataset

    print("=== Running High-Accuracy Model Check ===")
    
    BATCH_SIZE = 64
    EPOCHS = 10 
    LR = 0.001
    DATASET_NAME = 'mnist' 

    device = get_device()
    print(f"Using Device: {device}")

    print(f"Loading {DATASET_NAME} dataset...")
    train_ds, test_ds = dataset.get_dataset(DATASET_NAME)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)
    
    model = get_model(DATASET_NAME).to(device)
    print(f"Model initialized: {model.__class__.__name__}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    
    for epoch in range(1, EPOCHS + 1):
        train_step(model, train_loader, optimizer, epoch, device)
        acc = test_step(model, test_loader, device)
        scheduler.step()
        
    print("=== Model Check Complete ===")
