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
        save_file = os.path.join(save_dir, "A.pt")
        m = int(self.compression_ratio * self.num_params)
        n = self.num_params
        
        if os.path.exists(save_file):
            self.A = torch.load(save_file, map_location=self.device)
            if self.A.shape != (m, n):
                print(f"Shape mismatch. Regenerating A: {m}x{n}")
                self.A = torch.randn(m, n).to(self.device)
                torch.save(self.A, save_file)
            else:
                print("Loaded measurement matrix A.")
        else:
            self.A = torch.randn(m, n).to(self.device)
            torch.save(self.A, save_file)
            print(f"Generated A ({m}x{n}) at {save_file}")
            
        # Cache spectral norm for IHT
        print("Calculating spectral norm of A (this may take a moment)...")
        self.mu = 1 / (torch.norm(self.A, 2) ** 2)
        print("Spectral norm calculated.")
    
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
        # Use cached mu
        x = torch.zeros(A.shape[1], device=self.device)
        for _ in range(max_iter):
            x = x + self.mu * A.T @ (y - A @ x)
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