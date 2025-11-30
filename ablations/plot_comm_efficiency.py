import matplotlib.pyplot as plt
import json
import os
import numpy as np

def get_model_size_params():
    # 8-16-64-10 architecture
    # Conv1: 1*8*5*5 + 8 = 208
    # Conv2: 8*16*5*5 + 16 = 3216
    # FC1: 256*64 + 64 = 16448
    # FC2: 64*10 + 10 = 650
    # Total: ~20,522 params
    return 20522

def calculate_bits_per_epoch(method, num_params, num_clients, compression_ratio=0.05):
    # 32-bit float = 32 bits
    
    if method == "fedavg":
        # Downlink: Server -> Client (Full Model, 32-bit)
        # Uplink: Client -> Server (Full Model, 32-bit)
        # Per Client: 2 * 32 * num_params
        return num_clients * 2 * 32 * num_params
        
    elif method == "cs-fl":
        # Downlink: Server -> Client (Full Model, 32-bit) - Initial only? 
        # Actually in our loop:
        # Phase 1 Downlink: y_global (m floats) -> m * 32
        # Phase 1 Uplink: y_local (m floats) -> m * 32
        # Phase 2 Downlink: r_global (num_params bits) -> num_params * 1
        # Phase 2 Uplink: r_local (num_params bits) -> num_params * 1
        
        m = int(compression_ratio * num_params)
        
        # Per Client Per Epoch:
        # P1: 32*m (Up) + 32*m (Down)
        # P2: 1*N (Up) + 1*N (Down)
        return num_clients * ( (64 * m) + (2 * num_params) )
        
    elif method == "1bit-cs-fl":
        # Phase 1 Downlink: z_global (m bits) -> m * 1
        # Phase 1 Uplink: z_local (m bits) -> m * 1
        # Phase 2 Downlink: r_global (num_params bits) -> num_params * 1
        # Phase 2 Uplink: r_local (num_params bits) -> num_params * 1
        
        m = int(compression_ratio * num_params)
        
        # Per Client Per Epoch:
        return num_clients * ( (2 * m) + (2 * num_params) )
        
    elif method == "sign-sgd":
        # Up: N bits
        # Down: N bits
        return num_clients * 2 * num_params

def main():
    results_dir = "results" # Adjust if needed
    methods = ["fedavg", "cs-fl", "1bit-cs-fl"]
    
    num_params = get_model_size_params()
    num_clients = 3
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        filename = f"metrics_comm_eff_{method}.json"
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
            
        with open(filepath, "r") as f:
            data = json.load(f)
            
        epochs = [d['epoch'] for d in data]
        accs = [d['accuracy'] for d in data]
        
        # Calculate cumulative bits
        bits_per_epoch = calculate_bits_per_epoch(method, num_params, num_clients)
        cumulative_bits = [e * bits_per_epoch / 8 / 1024 / 1024 for e in epochs] # Convert to MB
        
        plt.plot(cumulative_bits, accs, marker='o', label=method)
        
    plt.xlabel("Communication Cost (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Communication Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/communication_efficiency.png")
    print("Plot saved to plots/communication_efficiency.png")

if __name__ == "__main__":
    main()
