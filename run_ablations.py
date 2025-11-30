import yaml
import os
import subprocess
import time
import copy

def load_base_config():
    with open("configs/base_config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_experiment(config, exp_name):
    print(f"--- Starting Experiment: {exp_name} ---")
    
    # Save temp config
    temp_config_path = f"configs/temp_{exp_name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
        
    # Start Server
    server_cmd = f"python main.py --role server --config {temp_config_path}"
    server_proc = subprocess.Popen(server_cmd, shell=True)
    
    # Give server time to start
    time.sleep(2)
    
    # Start Clients (using spawn.py)
    spawn_cmd = f"python spawn.py --config {temp_config_path}"
    spawn_proc = subprocess.Popen(spawn_cmd, shell=True)
    
    # Wait for server to finish (it terminates clients)
    server_proc.wait()
    spawn_proc.wait()
    
    # Cleanup
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    print(f"--- Finished Experiment: {exp_name} ---\n")

def main():
    base_config = load_base_config()
    
    # Define Ablations
    # Example: Compare methods
    methods = ["cs-fl", "1bit-cs-fl", "fedavg"]
    
    # Example: Compare compression ratios for CS-FL
    # ratios = [0.1, 0.05, 0.01]
    
    for method in methods:
        config = copy.deepcopy(base_config)
        config['federated']['method'] = method
        config['federated']['num_epochs'] = 5 # Short run for testing
        
        exp_name = f"{method}_run"
        run_experiment(config, exp_name)

if __name__ == "__main__":
    main()
