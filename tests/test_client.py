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

def run_baseline(client_loaders, test_loader, original_model):
    """
    Baseline Federated Averaging (no compression)
    """
    print(f"\n--- Starting Simulation: BASELINE ---")
    
    # Initialize Global Model
    global_model = copy.deepcopy(original_model).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    
    # Get parameter info
    flat_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    total_params = flat_params.numel()
    
    # Metrics
    accuracies = []
    bits_uploaded = []
    total_bits_counter = 0
    
    start_time = time.time()

    for round_idx in range(1, CONFIG['rounds'] + 1):
        print(f"Round {round_idx}/{CONFIG['rounds']}")
        local_updates = []
        
        for client_idx in range(CONFIG['num_clients']):
            # 1. Download global model
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
            
            # 3. Compute update
            with torch.no_grad():
                new_weights = torch.cat([p.data.view(-1) for p in client_model.parameters()])
                old_weights = torch.cat([p.data.view(-1) for p in global_model.parameters()])
                update_vector = new_weights - old_weights
            
            local_updates.append(update_vector)
            total_bits_counter += total_params * 32  # 32 bits per float
        
        # Average updates
        avg_update = torch.stack(local_updates).mean(dim=0)
        
        # Apply to global model
        start_idx = 0
        with torch.no_grad():
            for p in global_model.parameters():
                numel = p.numel()
                p.data.add_(avg_update[start_idx:start_idx+numel].view(p.shape))
                start_idx += numel
        
        # Evaluate
        acc = test_model(global_model, test_loader, CONFIG['device'])
        accuracies.append(acc)
        bits_uploaded.append(total_bits_counter / 8 / 1024 / 1024)
        print(f"  Acc: {acc:.2f}% | Total Upload: {bits_uploaded[-1]:.2f} MB")

    print(f"Finished BASELINE in {time.time() - start_time:.2f}s")
    return bits_uploaded, accuracies

def run_cs_fl(client_loaders, test_loader, original_model):
    """
    Standard CS-FL (analog compressed sensing)
    """
    print(f"\n--- Starting Simulation: CS-FL ---")
    
    # Initialize Global Model
    global_model = copy.deepcopy(original_model).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    
    # Get parameter info
    flat_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    total_params = flat_params.numel()
    
    # Initialize Compressor
    compressor = CSCompressor(total_params, CONFIG['compression_ratio'], device=CONFIG['device'])
    
    # Client state tracking
    client_residuals = [torch.zeros(total_params, device=CONFIG['device']) for _ in range(CONFIG['num_clients'])]
    
    # Metrics
    accuracies = []
    bits_uploaded = []
    total_bits_counter = 0
    
    start_time = time.time()

    for round_idx in range(1, CONFIG['rounds'] + 1):
        print(f"Round {round_idx}/{CONFIG['rounds']}")
        local_updates = []
        
        for client_idx in range(CONFIG['num_clients']):
            # 1. Download global model
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
            
            # 3. Compute update
            with torch.no_grad():
                new_weights = torch.cat([p.data.view(-1) for p in client_model.parameters()])
                old_weights = torch.cat([p.data.view(-1) for p in global_model.parameters()])
                update_vector = new_weights - old_weights
            
            # 4. Compress (analog CS)
            seed = round_idx * 1000 + client_idx
            payload, new_residual = compressor.compress(
                update_vector=update_vector,
                residual_vector=client_residuals[client_idx],
                seed=seed,
                use_1bit=False
            )
            client_residuals[client_idx] = new_residual
            
            # 5. Reconstruct
            reconstructed_update = compressor.reconstruct(
                payload=payload,
                seed=seed,
                use_1bit=False,
                iterations=20
            )
            local_updates.append(reconstructed_update)
            
            total_bits_counter += compressor.M * 32  # 32 bits per float
        
        # Average updates
        avg_update = torch.stack(local_updates).mean(dim=0)
        
        # Apply to global model
        start_idx = 0
        with torch.no_grad():
            for p in global_model.parameters():
                numel = p.numel()
                p.data.add_(avg_update[start_idx:start_idx+numel].view(p.shape))
                start_idx += numel
        
        # Evaluate
        acc = test_model(global_model, test_loader, CONFIG['device'])
        accuracies.append(acc)
        bits_uploaded.append(total_bits_counter / 8 / 1024 / 1024)
        print(f"  Acc: {acc:.2f}% | Total Upload: {bits_uploaded[-1]:.2f} MB")

    print(f"Finished CS-FL in {time.time() - start_time:.2f}s")
    return bits_uploaded, accuracies

def run_1bit_cs_fl(client_loaders, test_loader, original_model):
    """
    Two-Phase 1-bit CS-FL as per reference notebook
    """
    print(f"\n--- Starting Simulation: 1BIT-CS-FL (Two-Phase) ---")
    
    # Initialize Global Model
    global_model = copy.deepcopy(original_model).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    
    # Get parameter info
    flat_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    total_params = flat_params.numel()
    
    # Initialize Compressor
    compressor = CSCompressor(total_params, CONFIG['compression_ratio'], sparsity_ratio=0.005, device=CONFIG['device'])
    
    # Client state tracking (for error feedback in Phase 2)
    client_error_feedback = [torch.zeros(total_params, device=CONFIG['device']) for _ in range(CONFIG['num_clients'])]
    
    # Metrics
    accuracies = []
    bits_uploaded = []
    total_bits_counter = 0
    
    start_time = time.time()

    for round_idx in range(1, CONFIG['rounds'] + 1):
        print(f"Round {round_idx}/{CONFIG['rounds']}")
        
        # ========== PHASE 1: 1-bit CS ==========
        print("  Phase 1: 1-bit CS")
        z_payloads = []
        
        for client_idx in range(CONFIG['num_clients']):
            # 1. Download global model
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
            
            # 3. Compute h_t = w_new - w_old
            with torch.no_grad():
                new_weights = torch.cat([p.data.view(-1) for p in client_model.parameters()])
                old_weights = torch.cat([p.data.view(-1) for p in global_model.parameters()])
                h_t = new_weights - old_weights
            
            # 4. Sparsify: s_t = top-k(h_t)
            seed = round_idx  # Same seed for all clients
            payload, _ = compressor.compress(
                update_vector=h_t,
                residual_vector=None,  # No residual in Phase 1
                seed=seed,
                use_1bit=True
            )
            
            # 5. Store error feedback: e_t = h_t - s_t
            # We need to reconstruct s_t to compute error
            z = compressor._unpack_bits(payload, compressor.M) if isinstance(payload, bytes) else payload
            y = torch.matmul(compressor.generate_measurement_matrix(seed), h_t.flatten())
            # Sparsify h_t directly
            k_val = compressor.k
            abs_h = torch.abs(h_t)
            if k_val > 0 and k_val < total_params:
                threshold = torch.kthvalue(abs_h.flatten(), total_params - k_val + 1).values
                s_t = torch.where(abs_h >= threshold, h_t, torch.zeros_like(h_t))
            else:
                s_t = h_t
            
            e_t = h_t - s_t
            client_error_feedback[client_idx] = e_t.detach()
            
            z_payloads.append(z)
            
            # Track bits
            total_bits_counter += compressor.M * 1  # 1 bit per measurement
        
        # 6. Server aggregates signs
        stacked_z = torch.stack(z_payloads)
        z_avg = torch.sign(stacked_z.float().sum(dim=0))
        z_avg[z_avg == 0] = 1  # Handle ties
        
        # 7. Server reconstructs
        s_reconstructed = compressor.reconstruct(
            payload=z_avg,
            seed=round_idx,
            use_1bit=True,
            iterations=10
        )
        
        # 8. Update global model with Phase 1 update
        start_idx = 0
        with torch.no_grad():
            for p in global_model.parameters():
                numel = p.numel()
                p.data.add_(SERVER_LR_PHASE1 * s_reconstructed[start_idx:start_idx+numel].view(p.shape))
                start_idx += numel
        
        # ========== PHASE 2: SignSGD with Error Feedback ==========
        print("  Phase 2: SignSGD")
        r_signs = []
        
        for client_idx in range(CONFIG['num_clients']):
            # 1. Download updated global model
            client_model = copy.deepcopy(global_model)
            client_model.train()
            optimizer = optim.SGD(client_model.parameters(), lr=CONFIG['lr'])
            
            # 2. Local Training (another epoch)
            loader = client_loaders[client_idx]
            for epoch in range(CONFIG['local_epochs']):
                for data, target in loader:
                    data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 3. Compute h_t = w_new - w_old
            with torch.no_grad():
                new_weights = torch.cat([p.data.view(-1) for p in client_model.parameters()])
                old_weights = torch.cat([p.data.view(-1) for p in global_model.parameters()])
                h_t = new_weights - old_weights
            
            # 4. Add error feedback: r_t = sign(h_t + e_t)
            r_t = torch.sign(h_t + client_error_feedback[client_idx])
            r_t[r_t == 0] = 1
            r_signs.append(r_t)
            
            # Track bits (each sign is 1 bit)
            total_bits_counter += total_params * 1
        
        # 5. Server aggregates signs
        stacked_r = torch.stack(r_signs)
        r_avg = torch.sign(stacked_r.float().sum(dim=0))
        r_avg[r_avg == 0] = 1
        
        # 6. Update global model with Phase 2 update
        start_idx = 0
        with torch.no_grad():
            for p in global_model.parameters():
                numel = p.numel()
                p.data.add_(SERVER_LR_PHASE2 * r_avg[start_idx:start_idx+numel].view(p.shape))
                start_idx += numel
        
        # Evaluate Global Model
        acc = test_model(global_model, test_loader, CONFIG['device'])
        accuracies.append(acc)
        bits_uploaded.append(total_bits_counter / 8 / 1024 / 1024) # MB
        
        print(f"  Acc: {acc:.2f}% | Total Upload: {bits_uploaded[-1]:.2f} MB")

    print(f"Finished 1BIT-CS-FL in {time.time() - start_time:.2f}s")
    return bits_uploaded, accuracies

def plot_results(results):
    """Plot comparison of all three methods"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Accuracy vs Communication
    plt.subplot(1, 2, 1)
    for mode, data in results.items():
        plt.plot(data['bits'], data['acc'], label=mode, marker='o', linewidth=2)
    plt.title(f"Communication Efficiency (Compression Ratio: {CONFIG['compression_ratio']})")
    plt.xlabel("Data Uploaded (MB)")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Accuracy vs Rounds
    plt.subplot(1, 2, 2)
    for mode, data in results.items():
        rounds = list(range(1, len(data['acc']) + 1))
        plt.plot(rounds, data['acc'], label=mode, marker='o', linewidth=2)
    plt.title("Accuracy vs Communication Rounds")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparison_3methods.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to results/plots/comparison_3methods.png")
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
    
    # 3. Run all three experiments
    print("\n" + "="*60)
    bits, acc = run_baseline(client_loaders, test_loader, base_model)
    results['Baseline (FedAvg)'] = {'bits': bits, 'acc': acc}
    
    print("\n" + "="*60)
    bits, acc = run_cs_fl(client_loaders, test_loader, base_model)
    results['CS-FL (Analog)'] = {'bits': bits, 'acc': acc}
    
    print("\n" + "="*60)
    bits, acc = run_1bit_cs_fl(client_loaders, test_loader, base_model)
    results['1-bit CS-FL (Two-Phase)'] = {'bits': bits, 'acc': acc}
    
    # 4. Print summary
    print("\n" + "="*60)
    print("=== FINAL RESULTS SUMMARY ===")
    print("="*60)
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Final Accuracy: {data['acc'][-1]:.2f}%")
        print(f"  Total Communication: {data['bits'][-1]:.2f} MB")
        print(f"  Compression Ratio: {data['bits'][-1] / results['Baseline (FedAvg)']['bits'][-1]:.1%}")
    
    # 5. Plot results
    plot_results(results)
