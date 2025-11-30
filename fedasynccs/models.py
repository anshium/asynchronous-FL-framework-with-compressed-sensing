import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import io
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Optimized for CS-FL: ~22k params
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 64) # 16 * 4 * 4 = 256
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Ensure input is [Batch, 1, 28, 28] (MNIST/Fashion)
        if x.dim() == 2: # Handle flat input if any
             x = x.view(-1, 1, 28, 28)
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 256) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelHandler:
    def __init__(self, device, dataset_handler=None):
        self.device = device
        self.model = Net().to(self.device)
        self.dataset_handler = dataset_handler
        self.criterion = F.nll_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def train(self, client_id):
        if not self.dataset_handler:
            raise ValueError("Dataset handler not set for training")
        
        self.model.train()
        loader = self.dataset_handler.get_client_loader(client_id)
        total_loss = 0.0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Client {client_id} Train Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        if not self.dataset_handler:
            return 0, 0
            
        self.model.eval()
        loader = self.dataset_handler.get_val_loader()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        accuracy = 100 * correct / total
        print(f"Eval Loss: {total_loss/len(loader):.4f}, Acc: {accuracy:.2f}%")
        return total_loss / len(loader), accuracy

    def get_weights(self):
        return self.model.state_dict()
    
    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))