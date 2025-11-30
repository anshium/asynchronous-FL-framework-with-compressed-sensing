from utils import run_experiment

def main():
    thresholds = [0.001, 0.005, 0.01, 0.05]
    
    for t in thresholds:
        updates = {
            "federated": {
                "method": "1bit-cs-fl"
            },
            "method_args": {
                "sparsity_thresh": t
            }
        }
        run_experiment(updates, f"sparsity_sensitivity_{t}")

if __name__ == "__main__":
    main()
