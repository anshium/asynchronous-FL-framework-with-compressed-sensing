from utils import run_experiment
import subprocess
import os

def main():
    # Methods to compare
    methods = ["fedavg", "cs-fl", "1bit-cs-fl"]
    
    # Common config
    base_updates = {
        "federated": {
            "num_epochs": 10
        }
    }
    
    for method in methods:
        updates = base_updates.copy()
        updates["federated"] = {"method": method}
        
        # Ensure consistent compression ratio for fair comparison if needed, 
        # or rely on base_config defaults which are tuned for each.
        # For this plot, we usually just run the methods and then calculate bits later.
        
        run_experiment(updates, f"comm_eff_{method}")
        
        # Cleanup
        client_clear_cmd = f"python clean_clients.py"
        subprocess.Popen(client_clear_cmd, shell=True)

if __name__ == "__main__":
    main()
