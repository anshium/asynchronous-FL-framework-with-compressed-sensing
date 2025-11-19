import torch
import math
import numpy as np

class CSCompressor:
    def __init__(self, original_dim: int, compression_ratio: float, sparsity_ratio: float = None, device=None):
        self.N = original_dim
        self.M = int(original_dim * compression_ratio)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if sparsity_ratio is not None:
            self.k = int(self.N * sparsity_ratio)
        elif self.M < self.N:
            # Calculate sparsity k based on CS theory (approximate)
            self.k = int(self.M / (2 * math.log(self.N / self.M)))
        else:
            self.k = self.N
        
    def generate_measurement_matrix(self, seed: int) -> torch.Tensor:
        # Use a distinct generator to ensure reproducibility across client/server
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed)) # Ensure seed is int
        
        # Generate random Gaussian matrix
        A = torch.randn(self.M, self.N, generator=generator, device=self.device)
        # Normalize columns for RIP (Restricted Isometry Property) stability
        # A = A / math.sqrt(self.M) 
        return A

    def compress(self, update_vector: torch.Tensor, residual_vector: torch.Tensor, seed: int, use_1bit: bool = True):
        # Ensure inputs are on the correct device
        update_vector = update_vector.to(self.device)
        
        if residual_vector is not None:
            residual_vector = residual_vector.to(self.device)
            h = update_vector + residual_vector
        else:
            h = update_vector
        
        # 1. Sparsify (Top-K Error Feedback)
        k_val = self.k
        if k_val > 0 and k_val < self.N:
            abs_h = torch.abs(h)
            # kthvalue finds the k-th smallest, so for top-k largest we need (N - k + 1)
            threshold = torch.kthvalue(abs_h.flatten(), self.N - k_val + 1).values
            s = torch.where(abs_h >= threshold, h, torch.zeros_like(h))
        else:
            s = h # No sparsification if k is invalid or full dimension
            
        # 2. Update Residual (e = h - s)
        new_residual = h - s
        
        # 3. Generate Matrix A
        A = self.generate_measurement_matrix(seed)
        
        # 4. Linear Compression: y = As
        y = torch.matmul(A, s.flatten())
        
        if use_1bit:
            # Component 2: 1-Bit Quantization
            # z = sign(y). If 0, map to 1 (binary {-1, 1})
            z = torch.sign(y)
            z[z == 0] = 1 
            payload = self._pack_bits(z)
        else:
            # Component 1: Analog CS
            payload = y
            
        return payload, new_residual.detach()

    def reconstruct(self, payload, seed: int, use_1bit: bool = True, iterations: int = 20) -> torch.Tensor:
        A = self.generate_measurement_matrix(seed)
        
        if use_1bit:
            if isinstance(payload, bytes):
                z = self._unpack_bits(payload, self.M).to(self.device)
            else:
                z = payload.to(self.device)
            return self._biht(z, A, iterations)
        else:
            y = payload.to(self.device)
            return self._iht(y, A, iterations)

    def _biht(self, z: torch.Tensor, A: torch.Tensor, iterations: int) -> torch.Tensor:
        """Binary Iterative Hard Thresholding"""
        x_hat = torch.zeros(self.N, device=self.device)
        
        # Use calculated mu for stability (matches notebook implementation)
        spectral_norm = torch.linalg.norm(A, ord=2)
        mu = 1 / (spectral_norm ** 2)
        
        for _ in range(iterations):
            # Ax
            Ax = torch.matmul(A, x_hat)
            
            # Gradient: A.T * (sign(Ax) - z)
            # z is {-1, 1}. 
            # If sign(Ax) != z, we get error.
            sign_diff = torch.sign(Ax) - z
            
            gradient = torch.matmul(A.T, sign_diff)
            x_hat -= mu * gradient
            
            # Hard Thresholding
            x_hat = self._hard_threshold(x_hat)
            
        return x_hat

    def _iht(self, y: torch.Tensor, A: torch.Tensor, iterations: int) -> torch.Tensor:
        """Iterative Hard Thresholding"""
        x_hat = torch.zeros(self.N, device=self.device)
        
        spectral_norm = torch.linalg.norm(A, ord=2)
        mu = 1 / (spectral_norm ** 2)
        
        for _ in range(iterations):
            # Gradient step: x = x + mu * A.T @ (y - A @ x)
            residual = y - torch.matmul(A, x_hat)
            grad_step = torch.matmul(A.T, residual)
            x_hat += mu * grad_step
            
            x_hat = self._hard_threshold(x_hat)
            
        return x_hat

    def _hard_threshold(self, x: torch.Tensor) -> torch.Tensor:
        k_val = self.k
        if k_val <= 0 or k_val >= self.N: return x
        
        abs_x = x.abs()
        threshold = torch.kthvalue(abs_x.flatten(), self.N - k_val + 1).values
        return torch.where(abs_x >= threshold, x, torch.zeros_like(x))

    def _pack_bits(self, sign_tensor: torch.Tensor) -> bytes:
        # Convert {-1, 1} -> {0, 1} for packing
        # 1 -> 1, -1 -> 0
        bits = (sign_tensor > 0).cpu().numpy().astype(np.uint8)
        return np.packbits(bits).tobytes()

    def _unpack_bits(self, byte_data: bytes, length: int) -> torch.Tensor:
        bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        bits = bits[:length]
        # 0 -> -1, 1 -> 1
        signs = torch.from_numpy(bits.astype(np.float32)).to(self.device)
        signs = 2 * signs - 1
        return signs