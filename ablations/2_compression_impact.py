from utils import run_experiment
import subprocess

def main():
    ratios = [0.01, 0.03, 0.05, 0.1]
    for r in ratios:
        updates = {
            "federated": {
                "method": "1bit-cs-fl"
            },
            "method_args": {
                "compression_ratio": r
            }
        }
        run_experiment(updates, f"compression_impact_{r}")

        client_clear_cmd = f"python clean_clients.py"
        spawn_process = subprocess.Popen(client_clear_cmd, shell=True)

if __name__ == "__main__":
    main()
