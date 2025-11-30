import grpc
from concurrent import futures
import threading
import random
import time
import torch
import io
import etcd3
from copy import deepcopy
import os

from proto import FedCS_pb2, FedCS_pb2_grpc
from fedasynccs.models import ModelHandler
from fedasynccs.dataset import FederatedDataset
from fedasynccs.compression import CompressedSensing

# Aggregation Helper Functions
def aggregate_weights(weight_dict, method):
    weight_list = list(weight_dict.values())
    if not weight_list: return None

    if method == "average":
        avg_weights = {}
        for key in weight_list[0].keys():
            avg_tensor = torch.stack([w[key].float() for w in weight_list]).mean(dim=0)
            avg_weights[key] = avg_tensor.long() if weight_list[0][key].dtype == torch.long else avg_tensor
        return avg_weights
        
    elif method == "average_flattened":
        return torch.mean(torch.stack(weight_list), dim=0)
        
    elif method == "average_flattened_sign":
        avg_weights = torch.sum(torch.tensor(weight_list, dtype=torch.float), dim=0)
        return [True if i >= 0 else False for i in torch.sign(avg_weights)]
        
    return None

class Server(FedCS_pb2_grpc.FederatedLearningServerServicer):
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.etcd = etcd3.client(host=config['etcd']['host'], port=int(config['etcd']['port']))
        self.watch_active = True
        
        # Auto-cleanup stale clients
        print("Cleaning up stale clients...")
        try:
            self.etcd.delete_prefix("/clients/")
        except Exception as e:
            print(f"Warning: Failed to clean up clients: {e}")

        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = FederatedDataset(config['paths']['data_dir'], mode="mnist") # or random/fashion
        self.model = ModelHandler(self.device, self.data)
        
        # Compressed Sensing Setup
        if "cs" in config['federated']['method']:
            # Calculate num params
            params = {k: v for k, v in self.model.get_weights().items()}
            num_params = len(torch.cat([v.flatten() for v in params.values()]))
            self.cs = CompressedSensing(num_params, self.device, config['method_args'], save_dir=config['paths']['data_dir'])
        else:
            self.cs = None

        self.start_etcd_watcher()

    def start_etcd_watcher(self):
        def watch_clients():
            events_iterator, cancel = self.etcd.watch_prefix("/clients/")
            for event in events_iterator:
                if isinstance(event, etcd3.events.PutEvent):
                    self.clients[event.key.decode("utf-8")] = event.value.decode("utf-8")
                    print(f"Client added: {event.key.decode('utf-8')}")
                elif isinstance(event, etcd3.events.DeleteEvent):
                    key = event.key.decode("utf-8")
                    if key in self.clients: del self.clients[key]
        
        # Load existing
        for val, meta in self.etcd.get_prefix("/clients/"):
            self.clients[meta.key.decode("utf-8")] = val.decode("utf-8")
            
        threading.Thread(target=watch_clients, daemon=True).start()

    def get_clients(self, n):
        while len(self.clients) < n:
            time.sleep(1)
        return random.sample(list(self.clients.values()), n)

    # RPC Wrappers
    def rpc_send_model(self, client_addr, model_bytes):
        with grpc.insecure_channel(client_addr) as channel:
            stub = FedCS_pb2_grpc.FederatedLearningClientStub(channel)
            return stub.SendModel(FedCS_pb2.ModelFile(content=model_bytes))

    def rpc_start_training(self, client_addr, method, res_dict):
        with grpc.insecure_channel(client_addr) as channel:
            stub = FedCS_pb2_grpc.FederatedLearningClientStub(channel)
            empty = FedCS_pb2.Empty()
            
            if method == "SignSGD":
                res = stub.SignSGD(empty)
                res_dict[client_addr] = [1 if i else -1 for i in res.content]
            elif method == "Phase1-CSFL":
                res = stub.Phase1_CSFL(empty)
                res_dict[client_addr] = torch.tensor(res.content)
            elif method == "Phase1-1B-CSFL":
                res = stub.Phase1_1B_CSFL(empty)
                res_dict[client_addr] = [1 if i else -1 for i in res.content]
            elif method == "Phase2":
                res = stub.Phase2(empty)
                res_dict[client_addr] = [1 if i else -1 for i in res.content]
            else: # FedAvg
                res = stub.StartTraining(empty)
                res_dict[client_addr] = torch.load(io.BytesIO(res.content))

    def rpc_update_weights(self, client_addr, weights, method):
        with grpc.insecure_channel(client_addr) as channel:
            stub = FedCS_pb2_grpc.FederatedLearningClientStub(channel)
            meta = (("type", method),)
            
            if method == "Phase1-CSFL":
                stub.UpdateClientWeights(FedCS_pb2.ModelFlat(content=weights.tolist()), metadata=meta)
            elif "Sign" in method or "Phase2" in method or "1B" in method:
                bool_weights = [True if i > 0 else False for i in weights]
                stub.UpdateClientWeightsSign(FedCS_pb2.ModelFlatSign(content=bool_weights), metadata=meta)

    def terminate_all_clients(self):
        print("Terminating all clients...")
        for client_addr in self.clients.values():
            try:
                with grpc.insecure_channel(client_addr) as channel:
                    stub = FedCS_pb2_grpc.FederatedLearningClientStub(channel)
                    stub.Terminate(FedCS_pb2.Empty())
            except Exception as e:
                print(f"Failed to terminate {client_addr}: {e}")

    def run(self):
        method = self.config['federated']['method']
        print(f"Starting Server with method: {method}")
        
        # Setup Logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{method}_{int(time.time())}.txt")
        print(f"Logging to {log_file}")
        
        def log(msg):
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        # Initial Save
        model_path = os.path.join(self.config['paths']['model_storage'], "global_model.pth")
        self.model.save(model_path)

        # Wait for clients
        log("Waiting for clients...")
        clients = self.get_clients(self.config['federated']['num_clients'])
        log(f"Clients connected: {len(clients)}")

        # Distribute Model ONCE for CS/Sign methods
        if method in ["cs-fl", "1bit-cs-fl", "sign-sgd"]:
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            threads = [threading.Thread(target=self.rpc_send_model, args=(c, model_bytes)) for c in clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
            log("Model distributed (Initial).")

        for epoch in range(self.config['federated']['num_epochs']):
            log(f"\n--- Epoch {epoch+1} ---")
            # Re-sample clients if needed, but for now we use the same set or re-fetch
            clients = self.get_clients(self.config['federated']['num_clients'])
            
            # 1. Distribute Model (FedAvg only)
            if method == "fedavg":
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                threads = [threading.Thread(target=self.rpc_send_model, args=(c, model_bytes)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                log("Model distributed.")

            # 2. Training Phase
            results = {}
            # Define protocol based on method
            if method == "fedavg":
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "normal", results)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                avg = aggregate_weights(results, "average")
                self.model.set_weights(avg)

            elif method == "sign-sgd":
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "SignSGD", results)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                avg_weights = aggregate_weights(results, "average_flattened_sign")
                
                # Send back global sign
                threads = [threading.Thread(target=self.rpc_update_weights, args=(c, avg_weights, "SignSGD")) for c in self.clients.values()]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                # Server Update
                r_tensor = torch.tensor([1 if i else -1 for i in avg_weights]).float().to(self.device)
                self._apply_flat_update(r_tensor, self.config['method_args']['lr_2'])

            elif method == "cs-fl":
                # Phase 1
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "Phase1-CSFL", results)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                y_global = aggregate_weights(results, "average_flattened")
                
                # Send back y_global
                threads = [threading.Thread(target=self.rpc_update_weights, args=(c, y_global, "Phase1-CSFL")) for c in self.clients.values()]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                # Server Update (Phase 1)
                s = self.cs.iht(self.cs.A, y_global.to(self.device))
                self._apply_flat_update(s, self.config['method_args']['lr_1'])
                
                # Phase 2
                results_p2 = {}
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "Phase2", results_p2)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                r_global = aggregate_weights(results_p2, "average_flattened_sign")
                
                 # Send back r_global
                threads = [threading.Thread(target=self.rpc_update_weights, args=(c, r_global, "Phase2")) for c in self.clients.values()]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                # Server Update (Phase 2)
                r_tensor = torch.tensor([1 if i else -1 for i in r_global]).float().to(self.device)
                self._apply_flat_update(r_tensor, self.config['method_args']['lr_2'])

            elif method == "1bit-cs-fl":
                # Phase 1
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "Phase1-1B-CSFL", results)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                z_global = aggregate_weights(results, "average_flattened_sign")
                
                if z_global is None:
                    log("⚠️ No valid updates received in Phase 2. Skipping.")
                    continue
                
                # Send back z_global
                threads = [threading.Thread(target=self.rpc_update_weights, args=(c, z_global, "Phase1-1B-CSFL")) for c in self.clients.values()]
                [t.start() for t in threads]
                [t.join() for t in threads]

                # Server Update (Phase 1)
                z_tensor = torch.tensor([1 if i else -1 for i in z_global]).float().to(self.device)
                s = self.cs.biht(self.cs.A, z_tensor)
                self._apply_flat_update(s, self.config['method_args']['lr_1'])

                # Phase 2
                results_p2 = {}
                threads = [threading.Thread(target=self.rpc_start_training, args=(c, "Phase2", results_p2)) for c in clients]
                [t.start() for t in threads]
                [t.join() for t in threads]
                
                r_global = aggregate_weights(results_p2, "average_flattened_sign")
                
                if r_global is None:
                    log("⚠️ No valid updates received in Phase 2. Skipping.")
                    continue
                 # Send back r_global
                threads = [threading.Thread(target=self.rpc_update_weights, args=(c, r_global, "Phase2")) for c in self.clients.values()]
                [t.start() for t in threads]
                [t.join() for t in threads]

                # Server Update (Phase 2)
                r_tensor = torch.tensor([1 if i else -1 for i in r_global]).float().to(self.device)

                # r_reconstructed = self.cs.biht(self.cs.A, r_tensor)
                self._apply_flat_update(r_tensor, self.config['method_args']['lr_2'])

            
            # Save and Eval
            self.model.save(model_path)
            loss, acc = self.model.evaluate()
            log(f"Epoch {epoch+1} Eval - Loss: {loss:.4f}, Acc: {acc:.2f}%")

        self.terminate_all_clients()

    def _apply_flat_update(self, flat_update, lr):
        w = self.model.get_weights()
        ptr = 0
        for key, v in w.items():
            numel = v.numel()
            update = flat_update[ptr : ptr + numel].reshape(v.shape)
            w[key] = w[key] + lr * update.to(v.device)
            ptr += numel
        self.model.set_weights(w)

def serve(config):
    server = Server(config)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    FedCS_pb2_grpc.add_FederatedLearningServerServicer_to_server(server, grpc_server)
    grpc_server.add_insecure_port(f"{config['server']['ip']}:{config['server']['port']}")
    grpc_server.start()
    
    try:
        server.run()
    finally:
        print("Stopping server...")
        grpc_server.stop(0)