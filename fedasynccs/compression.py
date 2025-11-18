import torch
import math

def generate_measurement_matrix(m, n, device='cpu'):
    """
    Generates a Gaussian random measurement matrix A of size (m, n).
    Entries are drawn from N(0, 1/m).
    """
    # Using standard normal distribution scaled by 1/sqrt(m)
    # This ensures the matrix satisfies Restricted Isometry Property (RIP) with high probability

    # TODO: I feel that the A should be same on client and server side. Passing it alongwith the compressed update would defeat the purpose of the compression.
    # SOLUTIONS: What if we keep A fixed? What if we generate A from the same random seed on every client and server?

    A = torch.randn(m, n, device=device) / math.sqrt(m)
    return A

def sparsify_update(update_vector, sparsity_ratio):
    """
    Keeps only the top-k elements (largest magnitude) of the update vector.
    
    Args:
        update_vector: The flattened model update (1D tensor).
        sparsity_ratio: Fraction of elements to keep (e.g., 0.01 for 1%).
    
    Returns:
        sparse_vector: Vector with same shape, but non-top-k elements set to 0.
    """
    k = int(update_vector.numel() * sparsity_ratio)
    if k == 0:
        k = 1 # Ensure at least one element is kept
        
    # Find the k-th largest value by magnitude
    # topk returns values and indices, we need the smallest value in the top k
    threshold = torch.topk(update_vector.abs(), k).values[-1]
    
    # Zero out elements with magnitude less than threshold
    sparse_vector = update_vector.clone()
    sparse_vector[update_vector.abs() < threshold] = 0
    
    return sparse_vector

def compress_1bit(sparse_vector, measurement_matrix):
    """
    Compresses the sparse vector using the measurement matrix and 1-bit quantization.
    
    Formula: z = sign(A @ s)
    
    Args:
        sparse_vector: The N-dimensional sparse update.
        measurement_matrix: The MxN random matrix.
        
    Returns:
        compressed_vector: The M-dimensional binary vector (+1 or -1).
    """
    # Linear projection
    y = torch.matmul(measurement_matrix, sparse_vector)
    
    # 1-bit Quantization (Sign function)
    z = torch.sign(y)
    
    # Handle zeros: sign(0) = 0, but we need bits (+1/-1). Map 0 to 1 arbitrarily.
    z[z == 0] = 1 
    
    return z

def biht_reconstruction(z, A, original_size, sparsity_ratio, iterations=20, step_size=1.0):
    """
    Binary Iterative Hard Thresholding (BIHT) to reconstruct the sparse vector 
    from 1-bit measurements.
    
    Algorithm:
        x_{k+1} = HardThreshold(x_k - step * A.T @ (sign(A @ x_k) - z))
        Note: The gradient of the one-sided L1 loss implies moving towards consistency with z.
        The standard BIHT step is often: x = x + step * A.T * (z - sign(Ax))
        
    Args:
        z: Received compressed 1-bit vector (M).
        A: Measurement matrix (MxN).
        original_size: N (number of parameters).
        sparsity_ratio: To determine k for Hard Thresholding.
        iterations: Number of BIHT steps.
    
    Returns:
        x_hat: Reconstructed sparse vector.
    """
    k = int(original_size * sparsity_ratio)
    if k == 0: k = 1
    
    # Initialize estimate x to zero
    x_hat = torch.zeros(original_size, device=z.device)
    
    # Precompute transpose for efficiency if A is large
    A_T = A.t()
    
    for i in range(iterations):
        # 1. Calculate consistency with measurements
        Ax = torch.matmul(A, x_hat)
        z_hat = torch.sign(Ax)
        z_hat[z_hat == 0] = 1
        
        # 2. Compute gradient-like term: A.T * (z - z_hat)
        # We want x such that sign(Ax) matches z.
        residual = z - z_hat
        grad = torch.matmul(A_T, residual)
        
        # 3. Gradient Step
        x_next = x_hat + (step_size / 2) * grad # Dividing by 2 is a common heuristic normalization
        
        # 4. Hard Thresholding (Project onto sparsity constraint)
        # Keep only top-k elements of x_next
        threshold = torch.topk(x_next.abs(), k).values[-1]
        x_next[x_next.abs() < threshold] = 0
        
        x_hat = x_next
        
    return x_hat

if __name__ == "__main__":
    print("=== Running Compression Module Check ===")
    
    N = 1000
    M = 200
    SPARSITY = 0.01
    DEVICE = 'cpu'
    if torch.cuda.is_available(): DEVICE = 'cuda'
    elif torch.backends.mps.is_available(): DEVICE = 'mps'
    
    print(f"Device: {DEVICE}")
    print(f"Original Dim: {N}, Compressed Dim: {M}, Sparsity: {SPARSITY}")
    
    # 1. Create a synthetic sparse update vector
    original_vector = torch.randn(N, device=DEVICE)
    original_sparse = sparsify_update(original_vector, SPARSITY)
    print(f"Original Non-zeros: {torch.count_nonzero(original_sparse).item()}")
    
    # 2. Generate Matrix
    A = generate_measurement_matrix(M, N, device=DEVICE)
    
    # 3. Compress
    z = compress_1bit(original_sparse, A)
    print(f"Compressed shape: {z.shape}, Values: {torch.unique(z)}")
    
    # 4. Reconstruct
    print("Reconstructing with BIHT...")
    recovered_vector = biht_reconstruction(z, A, N, SPARSITY, iterations=50)
    
    # 5. Evaluate
    # Cosine similarity is a good metric for update vectors (direction matters more than scale in FL)
    cos_sim = torch.nn.functional.cosine_similarity(original_sparse.unsqueeze(0), recovered_vector.unsqueeze(0))
    print(f"Reconstruction Cosine Similarity: {cos_sim.item():.4f}")
    
    if cos_sim.item() > 0.6:
        print("SUCCESS: Reconstruction is highly correlated with original.")
    else:
        print("WARNING: Reconstruction correlation is low. Check parameters.")
        
    print("=== Check Complete ===")
