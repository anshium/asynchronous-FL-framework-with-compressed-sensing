import torch
import os

class CompressedSensing:
    def __init__(self, num_params, device, args, save_dir="."):
        self.num_params = num_params
        self.device = device
        self.sparsity_thresh = args.get('sparsity_thresh', 0.005)
        self.compression_ratio = args.get('compression_ratio', 0.1)
        self.lr_1 = args.get('lr_1', 0.1)
        self.lr_2 = args.get('lr_2', 0.002)
        
        # Measurement matrix A
        self.A = None
        self.generate_measurement_matrix(save_dir)
        
    def generate_measurement_matrix(self, save_dir):
        m = int(self.compression_ratio * self.num_params)
        n = self.num_params
        save_file = os.path.join(save_dir, f"A_{m}_{n}.pt")
        
        if os.path.exists(save_file):
            try:
                self.A = torch.load(save_file, map_location=self.device)
                print(f"Loaded measurement matrix A from {save_file}")
                return
            except Exception as e:
                print(f"Failed to load {save_file}: {e}. Regenerating...")
        
        # Generate deterministically
        print(f"Generating A ({m}x{n})...")
        g = torch.Generator(device=self.device)
        g.manual_seed(42) # Fixed seed for consistency across clients
        self.A = torch.randn(m, n, generator=g, device=self.device)
        
        # Atomic write
        temp_file = f"{save_file}.tmp.{os.getpid()}"
        torch.save(self.A, temp_file)
        os.rename(temp_file, save_file)
        print(f"Saved A to {save_file}")
    
    @staticmethod
    def sparsify(grad, p):
        k = int(p * grad.numel())
        if k == 0:
            return torch.zeros_like(grad)
        values, indices = torch.topk(torch.abs(grad.flatten()), k)
        mask = torch.zeros_like(grad.flatten())
        mask[indices] = 1
        return (grad.flatten() * mask).view_as(grad) 
    
    def iht(self, A, y, max_iter=50):
        # Iterative Hard Thresholding
        mu = 1 / (torch.norm(self.A, 2) ** 2)
        x = torch.zeros(A.shape[1], device=self.device)
        for _ in range(max_iter):
            x = x + mu * A.T @ (y - A @ x)
            x = self.sparsify(x, self.sparsity_thresh)
        return x

    def biht(self, A, z, max_iter=50):
        # Binary Iterative Hard Thresholding
        x = torch.zeros(A.shape[1], device=self.device)
        for _ in range(max_iter):
            Ax = A @ x
            sign_err = torch.clamp(z * Ax, max=0)
            grad = A.T @ sign_err
            x = x - 0.1 * grad
            x = self.sparsify(x, self.sparsity_thresh)
        return x