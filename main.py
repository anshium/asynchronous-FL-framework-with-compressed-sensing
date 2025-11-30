import argparse
import yaml
import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

def load_config(path="configs/base_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--id", help="Client ID (required for client)", default="1")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.role == "server":
        from fedasynccs.server import serve
        serve(config)
    else:
        from fedasynccs.client import serve
        serve(args.id, config)