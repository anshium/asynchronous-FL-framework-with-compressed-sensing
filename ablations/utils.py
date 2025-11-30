import yaml
import os
import subprocess
import time
import sys

def load_config(path="configs/base_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)

def run_experiment(config_updates, experiment_name, base_config_path="configs/base_config.yaml"):
    print(f"--- Starting Experiment: {experiment_name} ---")
    
    # 1. Load Base Config
    config = load_config(base_config_path)
    
    # 2. Apply Updates
    # Helper to update nested dict
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    config = update(config, config_updates)
    
    # Set experiment name for logging
    config['experiment_name'] = experiment_name
    config['paths']['results_dir'] = "results"
    
    # 3. Save Temp Config
    temp_config_path = f"configs/temp_{experiment_name}.yaml"
    save_config(config, temp_config_path)
    
    # 4. Start Server
    server_cmd = f"python main.py --role server --config {temp_config_path}"
    server_process = subprocess.Popen(server_cmd, shell=True)
    print(f"Server started (PID: {server_process.pid})")
    
    # 5. Start Clients (via spawn.py)
    # Give server a moment to start
    time.sleep(2)
    spawn_cmd = f"python spawn.py --config {temp_config_path}"
    spawn_process = subprocess.Popen(spawn_cmd, shell=True)
    
    # 6. Wait for completion
    try:
        server_process.wait()
        spawn_process.wait()
    except KeyboardInterrupt:
        server_process.terminate()
        spawn_process.terminate()
        
    print(f"--- Experiment {experiment_name} Completed ---\n")
    
    client_clear_cmd = f"python clean_clients.py"
    spawn_process = subprocess.Popen(client_clear_cmd, shell=True)

    # Cleanup
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
