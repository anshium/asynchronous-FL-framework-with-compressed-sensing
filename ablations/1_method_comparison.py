from utils import run_experiment

def main():
    methods = ["sign-sgd", "1bit-cs-fl", "fedavg"]
    
    for method in methods:
        updates = {
            "federated": {
                "method": method
            }
        }
        run_experiment(updates, f"method_comparison_{method}")

if __name__ == "__main__":
    main()
