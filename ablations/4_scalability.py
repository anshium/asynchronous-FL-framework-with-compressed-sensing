from utils import run_experiment

def main():
    client_counts = [3, 5, 10]
    
    for c in client_counts:
        updates = {
            "federated": {
                "method": "1bit-cs-fl",
                "num_clients": c
            }
        }
        run_experiment(updates, f"scalability_{c}_clients")

if __name__ == "__main__":
    main()
