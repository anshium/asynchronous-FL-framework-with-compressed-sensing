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
    'num_clients': 5,
    'rounds': 10,          # Global communication rounds
    'local_epochs': 10,    # Local training epochs per round
    'batch_size': 64,
    'lr': 0.05,
    'compression_ratio': 0.1,
    'subset_size': 5000,  # Increased for multi-client
    'device': models.get_device()
}

SERVER_LR_PHASE1 = 0.2  # Server-side learning rate for Phase 1 (1-bit CS)
SERVER_LR_PHASE2 = 0.002  # Server-side learning rate for Phase 2 (SignSGD)

def test_model(model, test_loader, device):
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

def run_simulation(mode, client_loaders, test_loader, original_model):
    """
    Runs a Federated Learning simulation with multiple clients.
    """
    print(f"\n--- Starting Simulation: {mode.upper()} ---")
    
    # Initialize Global Model
    global_model = copy.deepcopy(original_model).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    
    # Get parameter info
    flat_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    total_params = flat_params.numel()
    
    # Initialize Compressor & Client States
    compressor = None
    client_residuals = [torch.zeros(total_params, device=CONFIG['device']) for _ in range(CONFIG['num_clients'])]
    client_error_feedback = [torch.zeros(total_params, device=CONFIG['device']) for _ in range(CONFIG['num_clients'])]
    
    if mode != 'baseline':
        # Use 0.005 (0.5%) sparsity as per reference for 1-bit CS to ensure convergence
        sparsity = 0.005 if mode == '1bit-cs-fl' else None
        compressor = CSCompressor(total_params, CONFIG['compression_ratio'], sparsity_ratio=sparsity, device=CONFIG['device'])

    # Metrics
    accuracies = []
    bits_uploaded = []
    total_bits_counter = 0
    
    start_time = time.time()

    for round_idx in range(1, CONFIG['rounds'] + 1):
        print(f"Round {round_idx}/{CONFIG['rounds']}")
        
        if mode == '1bit-cs-fl':
            # TWO-PHASE APPROACH for 1-bit CS-FL
            # ========== PHASE 1: 1-bit CS ==========
            print("  Phase 1: 1-bit CS")
            local_updates_phase1 = []
        
        # --- Client Loop ---
        for client_idx in range(CONFIG['num_clients']):
            # 1. Client Download (Copy global model)
            client_model = copy.deepcopy(global_model)
            client_model.train()
            optimizer = optim.SGD(client_model.parameters(), lr=CONFIG['lr'])
            
            # 2. Local Training
            loader = client_loaders[client_idx]
            for epoch in range(CONFIG['local_epochs']):
                for data, target in loader:
                    data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 3. Calculate Update (Update = New - Old)
            # We want to send the *change* in weights
            with torch.no_grad():
                new_weights = torch.cat([p.data.view(-1) for p in client_model.parameters()])
                old_weights = torch.cat([p.data.view(-1) for p in global_model.parameters()])
                update_vector = new_weights - old_weights
            
            # 4. Compress & Upload
            if mode == 'baseline':
                local_updates.append(update_vector)
                step_bits = total_params * 32
            else:
                # CS Compression
                use_1bit = (mode == '1bit-cs-fl')
                
                if use_1bit:
                    # For 1-bit, all clients must use SAME seed for aggregation
                    seed = round_idx 
                else:
                    # For normal CS, can be different
                    seed = round_idx * 1000 + client_idx

                # Reference code does not seem to use error feedback for 1-bit CS
                res_vec = client_residuals[client_idx] if not use_1bit else None

                payload, new_residual = compressor.compress(
                    update_vector=update_vector,
                    residual_vector=res_vec,
                    seed=seed,
                    use_1bit=use_1bit
                )
                client_residuals[client_idx] = new_residual
                
                # Track bits
                step_bits = compressor.M * (1 if use_1bit else 32)
                    
                if use_1bit:
                    # Send payload (signs) directly
                    local_updates.append(payload)
                else:
                    # Server Reconstructs individually (Standard CS-FL)
                    reconstructed_update = compressor.reconstruct(
                        payload=payload,
                        seed=seed,
                        use_1bit=use_1bit,
                        iterations=20
                    )
                    local_updates.append(reconstructed_update)
            
            total_bits_counter += step_bits

        # --- Server Aggregation (FedAvg) ---
        if mode == '1bit-cs-fl':
             # Aggregate Signs (Majority Vote)
            unpacked_updates = []
            for payload in local_updates:
                if isinstance(payload, bytes):
                    unpacked_updates.append(compressor._unpack_bits(payload, compressor.M))
                else:
                    unpacked_updates.append(payload)
            
            stacked_payloads = torch.stack(unpacked_updates)
            # Average the signs (float mean), then take sign again
            # avg_signs = torch.sign(stacked_payloads.float().mean(dim=0))
            
            # Match reference aggregation: sum, then sign, then map >=0 to 1
            sum_signs = stacked_payloads.float().sum(dim=0)
            avg_signs = torch.sign(sum_signs)
            # Convert to {-1, 1} explicitly (reference does: 2 * z_t_avg - 1)
            # But avg_signs is already {-1, 0, 1}, so just handle 0s
            avg_signs[avg_signs == 0] = 1
            
            # Reconstruct ONCE at Server
            # Use the SAME seed as clients
            avg_update = compressor.reconstruct(
                payload=avg_signs, 
                seed=round_idx, 
                use_1bit=True,
                iterations=10  # Reference uses max_iter=10
            )
            
            # Apply Server LR
            avg_update = avg_update * SERVER_LR
        else:
            # Average the updates from all clients (Baseline or Standard CS-FL)
            avg_update = torch.stack(local_updates).mean(dim=0)
        
        # Apply averaged update to global model
        start_idx = 0
        with torch.no_grad():
            for p in global_model.parameters():
                numel = p.numel()
                p.data.add_(avg_update[start_idx:start_idx+numel].view(p.shape))
                start_idx += numel
        
        # Evaluate Global Model
        acc = test_model(global_model, test_loader, CONFIG['device'])
        accuracies.append(acc)
        bits_uploaded.append(total_bits_counter / 8 / 1024 / 1024) # MB
        
        print(f"  Acc: {acc:.2f}% | Total Upload: {bits_uploaded[-1]:.2f} MB")

    print(f"Finished {mode.upper()} in {time.time() - start_time:.2f}s")
    return bits_uploaded, accuracies

def plot_results(results):
    plt.figure(figsize=(10, 6))
    
    for mode, data in results.items():
        # X-Axis is MB uploaded, Y-Axis is Accuracy
        plt.plot(data['bits'], data['acc'], label=mode, marker='o', linewidth=2)
        
    plt.title(f"Communication Efficiency (Compression Ratio: {CONFIG['compression_ratio']})")
    plt.xlabel("Data Uploaded (MB)")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    print(f"=== Simulation Config ===")
    print(f"Dataset: {CONFIG['dataset']}")
    print(f"Clients: {CONFIG['num_clients']}")
    print(f"Rounds: {CONFIG['rounds']}")
    
    # 1. Prepare Data
    train_ds, test_ds = dataset.get_dataset(CONFIG['dataset'])
    
    # Split data for clients
    total_indices = list(range(CONFIG['subset_size']))
    split_size = len(total_indices) // CONFIG['num_clients']
    client_loaders = []
    for i in range(CONFIG['num_clients']):
        indices = total_indices[i*split_size : (i+1)*split_size]
        loader = dataset.get_dataloader(train_ds, indices, CONFIG['batch_size'])
        client_loaders.append(loader)
        
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    # 2. Prepare Base Model
    base_model = models.get_model(CONFIG['dataset'])
    
    results = {}
    
    # 3. Run Experiments
    # bits, acc = run_simulation('baseline', client_loaders, test_loader, base_model)
    # results['baseline'] = {'bits': bits, 'acc': acc}
    
    # bits, acc = run_simulation('cs-fl', client_loaders, test_loader, base_model)
    # results['cs-fl'] = {'bits': bits, 'acc': acc}
    
    bits, acc = run_simulation('1bit-cs-fl', client_loaders, test_loader, base_model)
    results['1bit-cs-fl'] = {'bits': bits, 'acc': acc}