import grpc
import torch
import io
import threading
import etcd3
import time
import sys
import os
from proto import FedCS_pb2, FedCS_pb2_grpc
from fedasynccs.models import ModelHandler
from fedasynccs.dataset import FederatedDataset
from fedasynccs.compression import CompressedSensing
from concurrent import futures

class Client(FedCS_pb2_grpc.FederatedLearningClientServicer):
    def __init__(self, client_id, port, config):
        self.client_id = client_id
        self.port = port
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Data & Model
        self.data = FederatedDataset(config['paths']['data_dir'], mode="mnist")
        self.model = ModelHandler(self.device, self.data)
        
        # Compressed Sensing
        if "cs" in config['federated']['method']:
            params = {k: v for k, v in self.model.get_weights().items()}
            num_params = len(torch.cat([v.flatten() for v in params.values()]))
            self.cs = CompressedSensing(num_params, self.device, config['method_args'], save_dir=config['paths']['data_dir'])
            
        self.register_with_etcd()

    def register_with_etcd(self):
        etcd = etcd3.client(host=self.config['etcd']['host'], port=int(self.config['etcd']['port']))
        key = f"/clients/{self.client_id}"
        value = f"localhost:{self.port}"
        etcd.put(key, value)
        print(f"Registered {self.client_id} at {value}")

    # RPC Implementations
    def SendModel(self, request, context):
        buffer = io.BytesIO(request.content)
        state_dict = torch.load(buffer, map_location=self.device)
        self.model.set_weights(state_dict)
        return FedCS_pb2.Status(ack=True)

    def StartTraining(self, request, context):
        # Baseline FedAvg
        self.model.train(int(self.client_id))
        
        buffer = io.BytesIO()
        torch.save(self.model.get_weights(), buffer)
        return FedCS_pb2.ModelFile(content=buffer.getvalue())

    def Phase1_CSFL(self, request, context):
        # 1. Store initial weights w_t
        self.w_prev = {k: v.clone() for k, v in self.model.get_weights().items()}
        
        # 2. Train to get w_t+1
        self.model.train(int(self.client_id))
        
        # 3. Calc gradient: w_t+1 - w_t
        w_curr = self.model.get_weights()
        grad_flat = torch.cat([(w_curr[k] - self.w_prev[k]).flatten() for k in w_curr])
        
        # 4. Sparsify
        self.s = self.cs.sparsify(grad_flat, self.config['method_args']['sparsity_thresh'])
        self.e = grad_flat - self.s # Store error for Phase 2

        # 5. CS Compression (A @ s)
        y = self.cs.A @ self.s.to(self.device)
        
        return FedCS_pb2.ModelFlat(content=y.cpu().numpy().tolist())

    def SignSGD(self, request, context):
        # 1. Store initial weights w_t
        self.w_prev = {k: v.clone() for k, v in self.model.get_weights().items()}
        
        # 2. Train to get w_t+1
        self.model.train(int(self.client_id))
        
        # 3. Calc gradient: w_t+1 - w_t
        w_curr = self.model.get_weights()
        grad_flat = torch.cat([(w_curr[k] - self.w_prev[k]).flatten() for k in w_curr])
        
        # 4. Sign
        return FedCS_pb2.ModelFlatSign(content=[True if i >= 0 else False for i in grad_flat])

    def Phase1_1B_CSFL(self, request, context):
        # 1. Store initial weights w_t
        self.w_prev = {k: v.clone() for k, v in self.model.get_weights().items()}
        
        # 2. Train to get w_t+1
        self.model.train(int(self.client_id))
        
        # 3. Calc gradient: w_t+1 - w_t
        w_curr = self.model.get_weights()
        grad_flat = torch.cat([(w_curr[k] - self.w_prev[k]).flatten() for k in w_curr])
        
        # 4. Sparsify
        self.s = self.cs.sparsify(grad_flat, self.config['method_args']['sparsity_thresh'])
        self.e = grad_flat - self.s # Store error for Phase 2

        # 5. CS Compression (A @ s)
        y = self.cs.A @ self.s.to(self.device)
        
        # 6. 1-bit quantization (sign)
        return FedCS_pb2.ModelFlatSign(content=[True if i >= 0 else False for i in y])

    def Phase2(self, request, context):
        # 1. Train again from updated weights
        self.model.train(int(self.client_id))
        
        # 2. Calculate new gradient
        w_curr = self.model.get_weights()
        # self.w_reconstructed is the w_t updated with global gradient from Phase 1
        grad_flat = torch.cat([(w_curr[k] - self.w_reconstructed[k]).flatten() for k in w_curr])
        
        # 3. Add error from Phase 1
        total_delta = grad_flat + self.e
        
        # 4. Sign of the total delta
        return FedCS_pb2.ModelFlatSign(content=[True if i >= 0 else False for i in total_delta])

    def UpdateClientWeightsSign(self, request, context):
        # Receive global sign vector, update local model
        # For 1-bit CS, this is the reconstructed gradient step
        signs = torch.tensor([1 if i else -1 for i in request.content]).float().to(self.device)
        
        # Server sends z_global (Phase 1) or r_global (Phase 2)
        # We need to distinguish based on metadata or infer?
        # The server sends metadata "type"
        
        method = dict(context.invocation_metadata())['type']
        
        if method == "Phase1-1B-CSFL":
            grad_update = self.cs.biht(self.cs.A, signs)
            lr = self.config['method_args']['lr_1']
            
            # Apply update to PREVIOUS weights (w_t)
            w = self.w_prev
            ptr = 0
            for key, v in w.items():
                numel = v.numel()
                upd = grad_update[ptr : ptr + numel].reshape(v.shape)
                w[key] = w[key] + lr * upd
                ptr += numel
                
            self.model.set_weights(w)
            self.w_reconstructed = {k: v.clone() for k, v in w.items()} # Save for Phase 2
            
        elif method == "Phase2":
            # Direct sign update
            grad_update = signs
            lr = self.config['method_args']['lr_2']
            
            # Apply to current weights (which are w_reconstructed from Phase 1)
            w = self.model.get_weights()
            ptr = 0
            for key, v in w.items():
                numel = v.numel()
                upd = grad_update[ptr : ptr + numel].reshape(v.shape)
                w[key] = w[key] + lr * upd
                ptr += numel
            self.model.set_weights(w)

        return FedCS_pb2.Status(ack=True)

    def Terminate(self, request, context):
        threading.Thread(target=self._delayed_exit, daemon=True).start()
        return FedCS_pb2.Empty()

    def _delayed_exit(self):
        time.sleep(0.5)
        print(f"Client {self.client_id} terminating...")
        os._exit(0)

def serve(client_id, config):
    port = int(config['server']['port']) + int(client_id) + 1
    client = Client(client_id, port, config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    FedCS_pb2_grpc.add_FederatedLearningClientServicer_to_server(client, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Client {client_id} listening on {port}")
    server.wait_for_termination()