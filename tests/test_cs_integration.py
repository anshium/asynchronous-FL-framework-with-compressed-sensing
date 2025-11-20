import torch
import numpy as np
import time
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fedasynccs.models as models
import fedasynccs.dataset as dataset
from fedasynccs.compression import CSCompressor

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_federated_cs_test():
    # ========================================
    # CONFIGURATION
    # ========================================
    dataset_name = 'mnist'
    num_clients = 5
    total_clients = 5
    num_epochs = 5
    batch_size = 64
    
    # CS Parameters
    compression_ratio = 0.1
    use_1bit = True
    biht_iterations = 20
    
    # Learning rates
    lr_phase1 = 0.001  # For local training
    lr_phase2 = 0.0001  # For compressed update
    
    # Client selection
    client_lr = 0.01
    
    print("=" * 60)
    print("FEDERATED CS-FL TEST")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Clients per round: {num_clients}/{total_clients}")
    print(f"Compression: {compression_ratio} | 1-bit: {use_1bit}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    # ========================================
    # SETUP
    # ========================================
    device = models.get_device()
    print(f"Device: {device}\n")
    
    # Load dataset
    train_ds, test_ds = dataset.get_dataset(dataset_name)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    # Create client data splits (IID for simplicity)
    client_indices = [list(range(i, len(train_ds), total_clients)) for i in range(total_clients)]
    train_loaders = [dataset.get_dataloader(train_ds, indices, batch_size) for indices in client_indices]
    
    # Initialize server model
    server_model = models.get_model(dataset_name).to(device)
    total_params = sum(p.numel() for p in server_model.parameters())
    print(f"Model parameters: {total_params}")
    
    # Initialize client models (copies of server)
    client_models = [models.get_model(dataset_name).to(device) for _ in range(total_clients)]
    
    # Client optimizers
    optimizers = [torch.optim.SGD(model.parameters(), lr=client_lr) for model in client_models]
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize CS Compressor
    compressor = CSCompressor(original_dim=total_params, compression_ratio=compression_ratio)
    
    # Client state tracking
    w_ts = [server_model.state_dict() for _ in range(total_clients)]  # Last global model each client saw
    e_ts = [None for _ in range(total_clients)]  # Error feedback residuals
    training_freqs = [0 for _ in range(total_clients)]  # Track training frequency
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # --- CLIENT SELECTION (Fair Round-Robin) ---
        selected_clients = []
        for _ in range(num_clients):
            min_idx = min(range(total_clients), key=lambda i: training_freqs[i])
            selected_clients.append(min_idx)
            training_freqs[min_idx] += 1
        
        print(f"Selected clients: {selected_clients}")
        
        # ========================================
        # PHASE 1: COMPRESSED SENSING AGGREGATION
        # ========================================
        print("\n--- PHASE 1: CS Aggregation ---")
        
        compressed_updates = []
        
        for client_idx in selected_clients:
            print(f"Training client {client_idx}...", end=" ")
            
            # Local training
            client_models[client_idx].train()
            for images, labels in train_loaders[client_idx]:
                images, labels = images.to(device), labels.to(device)
                optimizers[client_idx].zero_grad()
                loss = criterion(client_models[client_idx](images), labels)
                loss.backward()
                optimizers[client_idx].step()
            
            # Compute update: h_t = w_new - w_old
            h_t = {}
            for key in client_models[client_idx].state_dict():
                h_t[key] = client_models[client_idx].state_dict()[key] - w_ts[client_idx][key]
            
            # Flatten update
            h_t_flat = torch.cat([v.flatten() for v in h_t.values()]).cpu().numpy()
            
            # Initialize error feedback if first time
            if e_ts[client_idx] is None:
                e_ts[client_idx] = np.zeros_like(h_t_flat)
            
            # Compress using CS
            seed = epoch * 1000 + client_idx  # Deterministic seed
            payload, new_residual = compressor.compress(
                update_vector=h_t_flat,
                residual_vector=e_ts[client_idx],
                seed=seed,
                use_1bit=use_1bit
            )
            
            # Update residual
            e_ts[client_idx] = new_residual
            
            # Send to server (simulate)
            compressed_updates.append((payload, seed))
            print("✓")
        
        # --- SERVER AGGREGATION ---
        print("Server aggregating...", end=" ")
        
        reconstructed_updates = []
        for payload, seed in compressed_updates:
            s_t_reconstructed = compressor.reconstruct(
                payload=payload,
                seed=seed,
                use_1bit=use_1bit,
                iterations=biht_iterations
            )
            reconstructed_updates.append(s_t_reconstructed)
        
        # Average reconstructed updates
        avg_update = np.mean(reconstructed_updates, axis=0)
        
        # Convert to state dict format
        avg_update_tensor = torch.from_numpy(avg_update).to(device)
        offset = 0
        avg_update_dict = {}
        for key, param in server_model.state_dict().items():
            numel = param.numel()
            avg_update_dict[key] = avg_update_tensor[offset:offset+numel].view(param.shape)
            offset += numel
        
        # Update all client models with aggregated update
        for client_idx in range(total_clients):
            new_state = {}
            for key in w_ts[client_idx]:
                new_state[key] = w_ts[client_idx][key] + lr_phase1 * avg_update_dict[key]
            client_models[client_idx].load_state_dict(new_state)
            w_ts[client_idx] = client_models[client_idx].state_dict()
        
        print("✓")
        
        # ========================================
        # PHASE 2: 1-BIT SIGN AGGREGATION (Optional)
        # ========================================
        if not use_1bit:  # If we already used 1-bit in Phase 1, skip Phase 2
            print("\n--- PHASE 2: Sign Aggregation ---")
            
            sign_updates = []
            
            for client_idx in selected_clients:
                print(f"Client {client_idx} local step...", end=" ")
                
                # Another round of local training
                client_models[client_idx].train()
                for images, labels in train_loaders[client_idx]:
                    images, labels = images.to(device), labels.to(device)
                    optimizers[client_idx].zero_grad()
                    loss = criterion(client_models[client_idx](images), labels)
                    loss.backward()
                    optimizers[client_idx].step()
                
                # Compute update with error feedback
                h_t = {}
                for key in client_models[client_idx].state_dict():
                    h_t[key] = client_models[client_idx].state_dict()[key] - w_ts[client_idx][key]
                
                h_t_flat = torch.cat([v.flatten() for v in h_t.values()]).cpu().numpy()
                
                # Add error feedback and take sign
                r_t = np.sign(h_t_flat + e_ts[client_idx])
                sign_updates.append(r_t)
                print("✓")
            
            # Aggregate signs
            print("Aggregating signs...", end=" ")
            avg_sign = np.sign(np.sum(sign_updates, axis=0))
            
            # Convert to state dict
            avg_sign_tensor = torch.from_numpy(avg_sign).to(device)
            offset = 0
            avg_sign_dict = {}
            for key, param in server_model.state_dict().items():
                numel = param.numel()
                avg_sign_dict[key] = avg_sign_tensor[offset:offset+numel].view(param.shape)
                offset += numel
            
            # Update clients
            for client_idx in range(total_clients):
                new_state = {}
                for key in w_ts[client_idx]:
                    new_state[key] = w_ts[client_idx][key] + lr_phase2 * avg_sign_dict[key]
                client_models[client_idx].load_state_dict(new_state)
                w_ts[client_idx] = client_models[client_idx].state_dict()
            
            print("✓")
        
        # Update server model
        server_model.load_state_dict(client_models[0].state_dict())
        
        # ========================================
        # EVALUATION
        # ========================================
        print("\n--- Evaluation ---")
        server_acc = evaluate_model(server_model, test_loader, device)
        print(f"Server Model Accuracy: {server_acc:.2f}%")
        
        # Evaluate selected clients
        for client_idx in selected_clients[:3]:  # Show first 3 for brevity
            client_acc = evaluate_model(client_models[client_idx], test_loader, device)
            print(f"Client {client_idx} Accuracy: {client_acc:.2f}%")
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
    
    # ========================================
    # FINAL RESULTS
    # ========================================
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final Accuracy: {evaluate_model(server_model, test_loader, device):.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_federated_cs_test()