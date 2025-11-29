import subprocess
import time
import sys
import yaml

def execute_command(command):
    return subprocess.Popen(command, shell=True)

def main():
    # Load config to get num_clients
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    num_clients = config['federated']['num_clients']
    processes = []

    print(f"Spawning {num_clients} clients...")
    for i in range(num_clients):
        cmd = f"python main.py --role client --id {i}"
        p = execute_command(cmd)
        processes.append(p)
        print(f"Started Client {i} (PID: {p.pid})")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("Terminating...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()