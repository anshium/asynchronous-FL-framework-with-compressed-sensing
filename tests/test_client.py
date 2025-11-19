import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import os
import sys

# Import your local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedasynccs.compression import CSCompressor
import fedasynccs.dataset as dataset
import fedasynccs.models as models

# --- Configuration ---
CONFIG = {
    'dataset': 'mnist',
    'batch_size': 64,
    'lr': 0.05,           # Higher LR often helps CS methods overcome noise
    'epochs': 10,          # Keep it short for testing
    'compression_ratio': 0.1,
    'subset_size': 1000,  # Use a subset to speed up the slow CS reconstruction
    'device': models.get_device()
}

def test_model(model, test_loader, device):
    """Evaluates the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def run_simulation(mode, train_loader, test_loader, original_model):
    """
    Runs a training simulation for a specific mode: 'baseline', 'cs-fl', '1bit-cs-fl'.
    """
    print(f"\n--- Starting Simulation: {mode.upper()} ---")
    
    # Clone the model so everyone starts from the exact same weights
    model = copy.deepcopy(original_model).to(CONFIG['device'])
    
    # Count parameters
    flat_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    total_params = flat_params.numel()
    
    # Initialize Compressor (only used for CS modes)
    compressor = None
    residual = None
    
    if mode != 'baseline':
        print(f"Initializing Compressor (Ratio: {CONFIG['compression_ratio']})...")
        compressor = CSCompressor(total_params, CONFIG['compression_ratio'], device=CONFIG['device'])
        # Residual must be a Tensor on the correct device
        residual = torch.zeros(total_params, device=CONFIG['device'])
    
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    losses = []
    
    start_time = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
            
            # 1. Standard Forward/Backward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss += loss.item()

            # 2. Apply Updates based on Mode
            if mode == 'baseline':
                # Standard SGD
                optimizer.step()
                
            else:
                # --- Simulate Client-Server CS Loop ---
                
                # A. Get Gradients as a flat vector
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                    else:
                        grads.append(torch.zeros(p.numel(), device=CONFIG['device']))
                flat_grad = torch.cat(grads)
                
                # B. Client: Compress
                # We compress the gradient vector directly
                seed = int(time.time() * 100000) % 100000
                use_1bit = (mode == '1bit-cs-fl')
                
                payload, new_residual = compressor.compress(
                    update_vector=flat_grad,
                    residual_vector=residual,
                    seed=seed,
                    use_1bit=use_1bit
                )
                
                residual = new_residual # Update client memory
                
                # C. Server: Reconstruct
                # (Simulating the server receiving the payload)
                reconstructed_grad = compressor.reconstruct(
                    payload=payload,
                    seed=seed,
                    use_1bit=use_1bit,
                    iterations=50 # Lower iterations for speed in test
                )
                
                # D. Update Model Weights
                # Apply the noisy reconstructed update: w = w - lr * reconstructed_grad
                offset = 0
                with torch.no_grad():
                    for p in model.parameters():
                        numel = p.numel()
                        if p.grad is not None:
                            layer_update = reconstructed_grad[offset:offset+numel].view(p.shape)
                            p.data.sub_(CONFIG['lr'] * layer_update)
                        offset += numel

        # End of Epoch Metrics
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Test Accuracy
        acc = test_model(model, test_loader, CONFIG['device'])
        accuracies.append(acc)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    duration = time.time() - start_time
    print(f"Finished {mode.upper()} in {duration:.2f}s")
    return losses, accuracies

def plot_results(results):
    """Plots comparison graphs."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for mode, data in results.items():
        plt.plot(range(1, CONFIG['epochs'] + 1), data['acc'], label=mode, marker='o')
    plt.title(f"Test Accuracy (Ratio: {CONFIG['compression_ratio']})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    for mode, data in results.items():
        plt.plot(range(1, CONFIG['epochs'] + 1), data['loss'], label=mode, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Plot generated.")

if __name__ == "__main__":
    print(f"=== Simulation Config ===")
    print(f"Dataset: {CONFIG['dataset']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Compression Ratio: {CONFIG['compression_ratio']}")
    print(f"Note: Using subset of {CONFIG['subset_size']} samples for speed.")
    
    # 1. Prepare Data
    train_ds, test_ds = dataset.get_dataset(CONFIG['dataset'])
    
    # Subset for speed
    indices = list(range(CONFIG['subset_size']))
    train_loader = dataset.get_dataloader(train_ds, indices, CONFIG['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)
    
    # 2. Prepare Base Model
    base_model = models.get_model(CONFIG['dataset'])
    
    # 3. Run Experiments
    results = {}
    
    # Experiment A: Baseline
    loss, acc = run_simulation('baseline', train_loader, test_loader, base_model)
    results['baseline'] = {'loss': loss, 'acc': acc}
    
    # Experiment B: CS-FL (Analog)
    loss, acc = run_simulation('cs-fl', train_loader, test_loader, base_model)
    results['cs-fl'] = {'loss': loss, 'acc': acc}
    
    # Experiment C: 1-Bit CS-FL
    loss, acc = run_simulation('1bit-cs-fl', train_loader, test_loader, base_model)
    import torch
    
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import copy
# import time
# import matplotlib.pyplot as plt
# import os
# import sys

# # Import your local modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from fedasynccs.compression import CSCompressor
# import fedasynccs.dataset as dataset
# import fedasynccs.models as models

# # --- Configuration ---
# CONFIG = {
#     'dataset': 'mnist',
#     'batch_size': 64,
#     'lr': 0.05,           # Higher LR often helps CS methods overcome noise
#     'epochs': 3,          # Keep it short for testing
#     'compression_ratio': 0.1,
#     'subset_size': 1000,  # Use a subset to speed up the slow CS reconstruction
#     'device': models.get_device()
# }

# def test_model(model, test_loader, device):
#     """Evaluates the model on the test set."""
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#     return 100 * correct / total

# def run_simulation(mode, train_loader, test_loader, original_model):
#     """
#     Runs a training simulation for a specific mode: 'baseline', 'cs-fl', '1bit-cs-fl'.
#     """
#     print(f"\n--- Starting Simulation: {mode.upper()} ---")
    
#     # Clone the model so everyone starts from the exact same weights
#     model = copy.deepcopy(original_model).to(CONFIG['device'])
    
#     # Count parameters
#     flat_params = torch.cat([p.data.view(-1) for p in model.parameters()])
#     total_params = flat_params.numel()
    
#     # Initialize Compressor (only used for CS modes)
#     compressor = None
#     residual = None
    
#     if mode != 'baseline':
#         print(f"Initializing Compressor (Ratio: {CONFIG['compression_ratio']})...")
#         compressor = CSCompressor(total_params, CONFIG['compression_ratio'], device=CONFIG['device'])
#         # Residual must be a Tensor on the correct device
#         residual = torch.zeros(total_params, device=CONFIG['device'])
    
#     optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'])
#     criterion = nn.CrossEntropyLoss()
    
#     accuracies = []
#     losses = []
    
#     start_time = time.time()

#     for epoch in range(1, CONFIG['epochs'] + 1):
#         model.train()
#         epoch_loss = 0
        
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
            
#             # 1. Standard Forward/Backward
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             epoch_loss += loss.item()

#             # 2. Apply Updates based on Mode
#             if mode == 'baseline':
#                 # Standard SGD
#                 optimizer.step()
                
#             else:
#                 # --- Simulate Client-Server CS Loop ---
                
#                 # A. Get Gradients as a flat vector
#                 grads = []
#                 for p in model.parameters():
#                     if p.grad is not None:
#                         grads.append(p.grad.view(-1))
#                     else:
#                         grads.append(torch.zeros(p.numel(), device=CONFIG['device']))
#                 flat_grad = torch.cat(grads)
                
#                 # B. Client: Compress
#                 # We compress the gradient vector directly
#                 seed = int(time.time() * 100000) % 100000
#                 use_1bit = (mode == '1bit-cs-fl')
                
#                 payload, new_residual = compressor.compress(
#                     update_vector=flat_grad,
#                     residual_vector=residual,
#                     seed=seed,
#                     use_1bit=use_1bit
#                 )
                
#                 residual = new_residual # Update client memory
                
#                 # C. Server: Reconstruct
#                 # (Simulating the server receiving the payload)
#                 reconstructed_grad = compressor.reconstruct(
#                     payload=payload,
#                     seed=seed,
#                     use_1bit=use_1bit,
#                     iterations=20 # Lower iterations for speed in test
#                 )
                
#                 # D. Update Model Weights
#                 # Apply the noisy reconstructed update: w = w - lr * reconstructed_grad
#                 offset = 0
#                 with torch.no_grad():
#                     for p in model.parameters():
#                         numel = p.numel()
#                         if p.grad is not None:
#                             layer_update = reconstructed_grad[offset:offset+numel].view(p.shape)
#                             p.data.sub_(CONFIG['lr'] * layer_update)
#                         offset += numel

#         # End of Epoch Metrics
#         avg_loss = epoch_loss / len(train_loader)
#         losses.append(avg_loss)
        
#         # Test Accuracy
#         acc = test_model(model, test_loader, CONFIG['device'])
#         accuracies.append(acc)
        
#         print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

#     duration = time.time() - start_time
#     print(f"Finished {mode.upper()} in {duration:.2f}s")
#     return losses, accuracies

# def plot_results(results):
#     """Plots comparison graphs."""
#     plt.figure(figsize=(12, 5))
    
#     # Accuracy Plot
#     plt.subplot(1, 2, 1)
#     for mode, data in results.items():
#         plt.plot(range(1, CONFIG['epochs'] + 1), data['acc'], label=mode, marker='o')
#     plt.title(f"Test Accuracy (Ratio: {CONFIG['compression_ratio']})")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
#     plt.grid(True)
    
#     # Loss Plot
#     plt.subplot(1, 2, 2)
#     for mode, data in results.items():
#         plt.plot(range(1, CONFIG['epochs'] + 1), data['loss'], label=mode, marker='o')
#     plt.title("Training Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
#     print("Plot generated.")

# if __name__ == "__main__":
#     print(f"=== Simulation Config ===")
#     print(f"Dataset: {CONFIG['dataset']}")
#     print(f"Device: {CONFIG['device']}")
#     print(f"Compression Ratio: {CONFIG['compression_ratio']}")
#     print(f"Note: Using subset of {CONFIG['subset_size']} samples for speed.")
    
#     # 1. Prepare Data
#     train_ds, test_ds = dataset.get_dataset(CONFIG['dataset'])
    
#     # Subset for speed
#     indices = list(range(CONFIG['subset_size']))
#     train_loader = dataset.get_dataloader(train_ds, indices, CONFIG['batch_size'])
#     test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)
    
#     # 2. Prepare Base Model
#     base_model = models.get_model(CONFIG['dataset'])
    
#     # 3. Run Experiments
#     results = {}
    
#     # Experiment A: Baseline
#     loss, acc = run_simulation('baseline', train_loader, test_loader, base_model)
#     results['baseline'] = {'loss': loss, 'acc': acc}
    
#     # Experiment B: CS-FL (Analog)
#     loss, acc = run_simulation('cs-fl', train_loader, test_loader, base_model)
#     results['cs-fl'] = {'loss': loss, 'acc': acc}
    
#     # Experiment C: 1-Bit CS-FL
#     loss, acc = run_simulation('1bit-cs-fl', train_loader, test_loader, base_model)
    