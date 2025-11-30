import subprocess
import argparse
import time
import sys
import yaml

def execute_command(command):
    return subprocess.Popen(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config to get num_clients
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    num_clients = config['federated']['num_clients']
    processes = []

    print(f"Spawning {num_clients} clients using config: {args.config}")
    for i in range(num_clients):
        cmd = f"python main.py --role client --id {i} --config {args.config}"
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