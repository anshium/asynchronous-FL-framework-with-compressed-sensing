import argparse
import time
import logging
import grpc
import torch
import numpy as np

# Import generated gRPC modules
import federated_pb2
import federated_pb2_grpc

# Import local modules
import models
import dataset
from compression import CSCompressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Client %(message)s')

class FedACClient:
    def __init__(self, client_id, server_address, conf):
        self.client_id = client_id
        self.conf = conf
        self.device = models.get_device()
        
        # 1. gRPC Connection
        options = [('grpc.max_send_message_length', 50 * 1024 * 1024), 
                   ('grpc.max_receive_message_length', 50 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(server_address, options=options)
        self.stub = federated_pb2_grpc.FederatedLearningStub(self.channel)
        
        # 2. Data Loading
        logging.info(f"Loading {conf['dataset']} dataset...")
        train_ds, _ = dataset.get_dataset(conf['dataset'])
        
        partitions = dataset.partition_data_dirichlet(
            train_ds, 
            num_clients=conf['num_clients'], 
            alpha=conf['alpha'], 
            seed=42
        )
        my_indices = partitions[client_id]
        self.train_loader = dataset.get_dataloader(train_ds, my_indices, conf['batch_size'])
        self.num_samples = len(my_indices)
        
        # 3. Model Initialization
        self.model = models.get_model(conf['dataset']).to(self.device)
        
        # Helper to get total params
        self.total_params = sum(p.numel() for p in self.model.parameters())
        
        # 4. Compressed Sensing Module
        self.compressor = CSCompressor(
            original_dim=self.total_params, 
            compression_ratio=conf['compression_ratio'],
            device=self.device
        )
        
        # CS Error Feedback (Residual) - kept as Tensor on device
        self.cs_residual = torch.zeros(self.total_params, device=self.device)
        
        # 5. FedAC State Variables (kept as Tensors on device for speed)
        self.c_i = torch.zeros(self.total_params, device=self.device)
        self.c_tau = torch.zeros(self.total_params, device=self.device)

    def set_parameters_from_numpy(self, flat_numpy):
        """Load flat numpy array into model parameters."""
        tensor_data = torch.from_numpy(flat_numpy).to(self.device)
        torch.nn.utils.vector_to_parameters(tensor_data, self.model.parameters())

    def get_parameters_as_tensor(self):
        """Get flat tensor of current parameters."""
        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()

    def train_and_update(self):
        while True:
            try:
                # --- STEP 1: Pull Global Model & Correction Term ---
                logging.info("Requesting global model...")
                response = self.stub.GetGlobalModel(federated_pb2.Empty())
                
                global_round_id = response.round_id
                
                # Load Global Weights (x^t)
                global_weights_np = np.array(response.weights, dtype=np.float32)
                self.set_parameters_from_numpy(global_weights_np)
                
                # Keep a copy of start model (x_tau) for delta calculation
                start_model_tensor = self.get_parameters_as_tensor()
                
                # Load Global Correction (c^t)
                if len(response.global_control_variate) > 0:
                    c_tau_np = np.array(response.global_control_variate, dtype=np.float32)
                    self.c_tau = torch.from_numpy(c_tau_np).to(self.device)
                
                logging.info(f"Round {global_round_id}: Training on {self.num_samples} samples.")

                # --- STEP 2: Local Training with FedAC Correction ---
                if self.conf['use_1bit']:
                    # 1-bit reconstruction often requires smaller steps
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001) 
                else:
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=CONFIG['lr'])
                    
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.conf['lr'])
                self.model.train()
                
                # FedAC correction term: (c_tau - c_i)
                # We add this to the gradients at every step.
                correction_diff = self.c_tau - self.c_i
                
                steps = 0
                for epoch in range(self.conf['local_epochs']):
                    for data, target in self.train_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        
                        output = self.model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        
                        # --- FedAC Correction Injection ---
                        # w_{k+1} = w_k - lr * (grad + c_tau - c_i)
                        # We modify grad: p.grad = p.grad + (c_tau - c_i)
                        start_idx = 0
                        for p in self.model.parameters():
                            numel = p.numel()
                            if p.grad is not None:
                                layer_correction = correction_diff[start_idx:start_idx+numel].view(p.shape)
                                p.grad.add_(layer_correction)
                            start_idx += numel
                            
                        optimizer.step()
                        steps += 1

                # --- STEP 3: Post-Training Calculations (FedAC & CS) ---
                
                final_model_tensor = self.get_parameters_as_tensor()
                
                # A. Calculate Pseudo-Gradient / Actual Update
                # update_vector = x_start - x_final (The direction server needs to add if using server learning rate)
                # In FedBuff: delta = x_{i,K} - x_tau. 
                # Here we calculate: actual_update = x_final - x_start (This is the accumulated negative gradient)
                actual_update_vector = final_model_tensor - start_model_tensor
                
                # B. Update Local Control Variate (c_i)
                # Eq (14): c_new = c_i - c_tau + (x_start - x_final) / (K * eta)
                # Note: (x_start - x_final) / (K*eta) approximates the average gradient.
                K = steps
                eta = self.conf['lr']
                
                # Careful with signs:
                # avg_grad = (start - final) / (K * eta)
                avg_gradient = (start_model_tensor - final_model_tensor) / (K * eta)
                
                c_i_new = avg_gradient - correction_diff
                
                # Calculate delta for server: delta_c = c_new - c_old
                delta_c_i = c_i_new - self.c_i
                
                # Update local state
                self.c_i = c_i_new

                # C. Compress the Model Update (actual_update_vector)
                seed = int(time.time() * 1000) % 100000
                
                packed_payload, new_residual = self.compressor.compress(
                    update_vector=actual_update_vector, 
                    residual_vector=self.cs_residual, 
                    seed=seed, 
                    use_1bit=self.conf['use_1bit']
                )
                
                self.cs_residual = new_residual
                
                # Convert payload to bytes if it isn't already (for analog case)
                if self.conf['use_1bit']:
                    payload_bytes = packed_payload # It's already bytes from _pack_bits
                else:
                    payload_bytes = packed_payload.cpu().numpy().tobytes()

                # --- STEP 4: Push Update via gRPC ---
                logging.info("Sending compressed update...")
                
                update_msg = federated_pb2.ClientUpdate(
                    client_id=str(self.client_id),
                    base_model_round_id=global_round_id,
                    compressed_payload=payload_bytes,
                    measurement_seed=seed,
                    original_vector_len=self.total_params,
                    compressed_vector_len=self.compressor.M,
                    # Note: sending uncompressed delta_c_i is a bottleneck but required for FedAC correctness 
                    # unless we implement sparsification for control variates too.
                    delta_control_variate=delta_c_i.cpu().numpy().tolist(),
                    num_samples=self.num_samples,
                    is_1bit=self.conf['use_1bit']
                )
                
                self.stub.SubmitUpdate(update_msg)
                logging.info("Update submitted. Sleeping...")
                time.sleep(1) # Throttle

            except grpc.RpcError as e:
                logging.error(f"gRPC Error: {e}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error: {e}")
                time.sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--server', type=str, default='localhost:50051')
    args = parser.parse_args()

    config = {
        'dataset': 'mnist',
        'num_clients': 10,
        'alpha': 0.5,
        'batch_size': 32,
        'lr': 0.01,
        'local_epochs': 1,
        'compression_ratio': 0.1,
        'use_1bit': True
    }

    client = FedACClient(args.id, args.server, config)
    client.train_and_update()