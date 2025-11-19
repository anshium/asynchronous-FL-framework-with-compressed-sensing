import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedasynccs.compression import CSCompressor

# Helper to get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running tests on: {device}")

def test_measurement_matrix_consistency():
    """Test that the same seed generates identical matrices"""
    print("\n=== Test 1: Measurement Matrix Consistency ===")
    compressor = CSCompressor(original_dim=1000, compression_ratio=0.2, device=device)
    
    A1 = compressor.generate_measurement_matrix(seed=42)
    A2 = compressor.generate_measurement_matrix(seed=42)
    A3 = compressor.generate_measurement_matrix(seed=99)
    
    assert torch.allclose(A1, A2), "Same seed should generate identical matrices"
    assert not torch.allclose(A1, A3), "Different seeds should generate different matrices"
    print("✓ Matrix consistency test passed")


def test_1bit_compression_reconstruction():
    """Test 1-bit CS-FL compression and reconstruction"""
    print("\n=== Test 2: 1-Bit Compression & Reconstruction ===")
    
    N = 1000
    compressor = CSCompressor(original_dim=N, compression_ratio=0.2, device=device)
    
    # Create sparse update vector
    update = torch.randn(N, device=device)
    residual = torch.zeros(N, device=device)
    
    # Compress
    payload, new_residual = compressor.compress(update, residual, seed=42, use_1bit=True)
    
    # Payload is bytes for 1-bit
    print(f"Original size: {N * 4} bytes (float32), Compressed: {len(payload)} bytes")
    print(f"Compression ratio: {len(payload) / (N * 4):.4f}")
    
    # Reconstruct
    reconstructed = compressor.reconstruct(payload, seed=42, use_1bit=True, iterations=50)
    
    # Evaluate
    # Cosine Similarity
    cosine_sim = torch.dot(update, reconstructed) / (torch.norm(update) * torch.norm(reconstructed))
    
    print(f"Cosine similarity: {cosine_sim.item():.4f}")
    print(f"Sparsity level k: {compressor.k}")
    
    # Check non-zeros (approximate due to float precision, but hard thresholding sets exact zeros)
    non_zeros = torch.count_nonzero(reconstructed).item()
    print(f"Non-zeros in reconstruction: {non_zeros}")
    
    print("✓ 1-bit compression test passed")


def test_analog_compression_reconstruction():
    """Test analog CS-FL compression and reconstruction"""
    print("\n=== Test 3: Analog Compression & Reconstruction ===")
    
    N = 1000
    compressor = CSCompressor(original_dim=N, compression_ratio=0.2, device=device)
    
    # Create sparse update vector
    update = torch.randn(N, device=device)
    residual = torch.zeros(N, device=device)
    
    # Compress
    payload, new_residual = compressor.compress(update, residual, seed=42, use_1bit=False)
    
    # Payload is Tensor for Analog
    payload_size = payload.numel() * 4 # float32 bytes
    print(f"Original size: {N * 4} bytes, Compressed: {payload_size} bytes")
    print(f"Compression ratio: {payload_size / (N * 4):.4f}")
    
    # Reconstruct
    reconstructed = compressor.reconstruct(payload, seed=42, use_1bit=False, iterations=50)
    
    # Evaluate
    cosine_sim = torch.dot(update, reconstructed) / (torch.norm(update) * torch.norm(reconstructed))
    mse = torch.mean((update - reconstructed) ** 2)
    
    print(f"Cosine similarity: {cosine_sim.item():.4f}")
    print(f"MSE: {mse.item():.6f}")
    print(f"Non-zeros in reconstruction: {torch.count_nonzero(reconstructed).item()}")
    
    print("✓ Analog compression test passed")


def test_error_feedback():
    """Test error feedback mechanism across multiple rounds"""
    print("\n=== Test 4: Error Feedback Mechanism ===")
    
    N = 500
    compressor = CSCompressor(original_dim=N, compression_ratio=0.3, device=device)
    
    residual = torch.zeros(N, device=device)
    total_error = 0.0
    
    for round_num in range(3):
        update = torch.randn(N, device=device)
        payload, residual = compressor.compress(update, residual, seed=round_num, use_1bit=True)
        
        error_norm = torch.norm(residual).item()
        total_error += error_norm
        print(f"Round {round_num + 1}: Residual norm = {error_norm:.4f}")
    
    print(f"Total accumulated error metric: {total_error:.4f}")
    print("✓ Error feedback test passed")


def test_sparsity_preservation():
    """Test that reconstruction maintains sparsity"""
    print("\n=== Test 5: Sparsity Preservation ===")
    
    N = 800
    compressor = CSCompressor(original_dim=N, compression_ratio=0.25, device=device)
    
    # Create naturally sparse update
    update = torch.zeros(N, device=device)
    # Randomly select k indices
    perm = torch.randperm(N, device=device)
    sparse_indices = perm[:compressor.k]
    update[sparse_indices] = torch.randn(compressor.k, device=device)
    
    residual = torch.zeros(N, device=device)
    
    # Compress and reconstruct
    payload, _ = compressor.compress(update, residual, seed=123, use_1bit=True)
    reconstructed = compressor.reconstruct(payload, seed=123, use_1bit=True, iterations=50)
    
    original_nonzeros = torch.count_nonzero(update).item()
    reconstructed_nonzeros = torch.count_nonzero(reconstructed).item()
    
    print(f"Original non-zeros: {original_nonzeros}")
    print(f"Reconstructed non-zeros: {reconstructed_nonzeros}")
    print(f"Expected k: {compressor.k}")
    
    # Hard thresholding in reconstruction ensures exactly k non-zeros (or fewer if 0s selected)
    assert reconstructed_nonzeros <= compressor.k, "Reconstruction not sparse enough"
    print("✓ Sparsity preservation test passed")


def test_bit_packing_unpacking():
    """Test bit packing/unpacking utilities"""
    print("\n=== Test 6: Bit Packing/Unpacking ===")
    
    compressor = CSCompressor(original_dim=100, compression_ratio=0.2, device=device)
    
    # Test with known pattern
    # Create a tensor of 1s and -1s
    original_signs = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1] * 25, dtype=torch.float32, device=device)
    
    packed = compressor._pack_bits(original_signs)
    unpacked = compressor._unpack_bits(packed, len(original_signs))
    
    assert torch.allclose(original_signs, unpacked), "Bit packing/unpacking mismatch"
    print(f"Original length: {len(original_signs)}, Packed bytes: {len(packed)}")
    print(f"Compression: {len(packed) / (len(original_signs) * 4):.4f}")
    print("✓ Bit packing test passed")


def compare_1bit_vs_analog():
    """Compare 1-bit and analog compression"""
    print("\n=== Test 7: 1-Bit vs Analog Comparison ===")
    
    N = 1000
    compressor = CSCompressor(original_dim=N, compression_ratio=0.2, device=device)
    
    update = torch.randn(N, device=device)
    residual = torch.zeros(N, device=device)
    
    # 1-bit
    payload_1bit, _ = compressor.compress(update, residual.clone(), seed=42, use_1bit=True)
    reconstructed_1bit = compressor.reconstruct(payload_1bit, seed=42, use_1bit=True, iterations=50)
    cosine_1bit = torch.dot(update, reconstructed_1bit) / (torch.norm(update) * torch.norm(reconstructed_1bit))
    
    # Analog
    payload_analog, _ = compressor.compress(update, residual.clone(), seed=42, use_1bit=False)
    reconstructed_analog = compressor.reconstruct(payload_analog, seed=42, use_1bit=False, iterations=50)
    cosine_analog = torch.dot(update, reconstructed_analog) / (torch.norm(update) * torch.norm(reconstructed_analog))
    
    size_1bit = len(payload_1bit)
    size_analog = payload_analog.numel() * 4
    
    print(f"1-bit: Size={size_1bit} bytes, Cosine Sim={cosine_1bit.item():.4f}")
    print(f"Analog: Size={size_analog} bytes, Cosine Sim={cosine_analog.item():.4f}")
    print(f"Size ratio (1-bit/analog): {size_1bit / size_analog:.4f}")
    print("✓ Comparison complete")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Compression Module Tests (PyTorch)")
    print("=" * 60)
    
    test_measurement_matrix_consistency()
    test_1bit_compression_reconstruction()
    test_analog_compression_reconstruction()
    test_error_feedback()
    test_sparsity_preservation()
    test_bit_packing_unpacking()
    compare_1bit_vs_analog()
    print("\nAll tests completed successfully.")