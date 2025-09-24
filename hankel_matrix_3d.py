"""
3D Hankel Matrix Implementation with Delay Embedding
Creates Hankel matrices with shape (n_batches, delay_embedding_dim, window_length)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def build_3d_hankel_matrix(signal, window_len, delay_embedding_dim, stride=None):
    """
    Build 3D Hankel matrix with delay embedding
    
    Args:
        signal: Input time series signal
        window_len: Length of each window (e.g., 512)
        delay_embedding_dim: Number of overlapping windows per batch (e.g., 10)
        stride: Step size between batches. If None, uses delay_embedding_dim
    
    Returns:
        hankel_matrix: Shape (n_batches, delay_embedding_dim, window_len)
        batch_starts: Starting indices for each batch
    """
    T = len(signal)
    
    if stride is None:
        stride = delay_embedding_dim
    
    # Calculate number of batches
    # Each batch needs (delay_embedding_dim - 1) * stride + window_len samples
    samples_per_batch = (delay_embedding_dim - 1) * stride + window_len
    max_start = T - samples_per_batch
    batch_starts = list(range(0, max_start + 1, stride))
    
    n_batches = len(batch_starts)
    
    # Initialize 3D Hankel matrix
    hankel_matrix = np.zeros((n_batches, delay_embedding_dim, window_len), dtype=np.float32)
    
    # Fill the matrix
    for batch_idx, start in enumerate(batch_starts):
        for delay_idx in range(delay_embedding_dim):
            window_start = start + delay_idx * stride
            window_end = window_start + window_len
            
            if window_end <= T:
                hankel_matrix[batch_idx, delay_idx, :] = signal[window_start:window_end]
    
    return hankel_matrix, batch_starts


class Hankel3DDataset(Dataset):
    """Dataset for 3D Hankel matrix with delay embedding"""
    
    def __init__(self, signal, window_len=512, delay_embedding_dim=10, stride=None, 
                 normalize=True, shuffle=True):
        self.signal = signal.astype(np.float32)
        self.window_len = window_len
        self.delay_embedding_dim = delay_embedding_dim
        self.stride = stride if stride is not None else delay_embedding_dim
        
        # Build 3D Hankel matrix
        self.hankel_matrix, self.batch_starts = build_3d_hankel_matrix(
            self.signal, window_len, delay_embedding_dim, self.stride
        )
        
        print(f"3D Hankel matrix shape: {self.hankel_matrix.shape}")
        print(f"Number of batches: {self.hankel_matrix.shape[0]}")
        print(f"Delay embedding dimension: {self.hankel_matrix.shape[1]}")
        print(f"Window length: {self.hankel_matrix.shape[2]}")
        
        # Normalize if requested
        if normalize:
            self.mean = self.hankel_matrix.mean()
            self.std = self.hankel_matrix.std() + 1e-8
            self.hankel_matrix = (self.hankel_matrix - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
        
        # Shuffle batches if requested
        if shuffle:
            self.shuffle_indices()
        else:
            self.indices = np.arange(len(self.hankel_matrix))
    
    def shuffle_indices(self):
        """Shuffle the batch indices for generalization"""
        self.indices = np.random.permutation(len(self.hankel_matrix))
        print(f"Shuffled {len(self.indices)} batches")
    
    def __len__(self):
        return len(self.hankel_matrix)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return torch.from_numpy(self.hankel_matrix[actual_idx]).float()


def reconstruct_from_3d_hankel(hankel_matrix, batch_starts, stride, T, window_fn=None, mean=0.0, std=1.0):
    """
    Reconstruct full signal from 3D Hankel matrix
    
    Args:
        hankel_matrix: Shape (n_batches, delay_embedding_dim, window_len)
        batch_starts: Starting indices for each batch
        stride: Step size used in construction
        T: Target signal length
        window_fn: Optional windowing function
        mean: Mean used for normalization (for denormalization)
        std: Std used for normalization (for denormalization)
    
    Returns:
        reconstructed_signal: Reconstructed time series
    """
    recon_agg = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    
    n_batches, delay_dim, window_len = hankel_matrix.shape
    
    if window_fn is None:
        window = np.ones(window_len, dtype=np.float32)
    else:
        window = window_fn(window_len).astype(np.float32)
    
    # Reconstruct by averaging overlapping windows
    for batch_idx, batch_start in enumerate(batch_starts):
        for delay_idx in range(delay_dim):
            window_start = batch_start + delay_idx * stride
            window_end = window_start + window_len
            
            if window_end <= T:
                win_data = hankel_matrix[batch_idx, delay_idx, :] * window
                recon_agg[window_start:window_end] += win_data
                counts[window_start:window_end] += window
    
    # Average overlapping regions
    mask = counts > 0
    reconstructed_signal = np.zeros_like(recon_agg)
    reconstructed_signal[mask] = recon_agg[mask] / counts[mask]
    
    # Denormalize the reconstructed signal
    reconstructed_signal = reconstructed_signal * std + mean
    
    return reconstructed_signal


def demonstrate_hankel_construction():
    """Demonstrate the Hankel matrix construction with a simple example"""
    print("=== Hankel Matrix Construction Demo ===")
    
    # Create simple test signal
    signal = np.arange(100)  # [0, 1, 2, ..., 99]
    window_len = 10
    delay_embedding_dim = 5
    stride = 5
    
    print(f"Signal length: {len(signal)}")
    print(f"Window length: {window_len}")
    print(f"Delay embedding dimension: {delay_embedding_dim}")
    print(f"Stride: {stride}")
    print()
    
    # Build Hankel matrix
    hankel_matrix, batch_starts = build_3d_hankel_matrix(
        signal, window_len, delay_embedding_dim, stride
    )
    
    print(f"Hankel matrix shape: {hankel_matrix.shape}")
    print(f"Batch starts: {batch_starts}")
    print()
    
    # Show first batch
    print("First batch (delay embedding):")
    for i in range(delay_embedding_dim):
        start = batch_starts[0] + i * stride
        end = start + window_len
        print(f"  Window {i}: indices [{start}:{end}] = {signal[start:end]}")
    
    print()
    print("Second batch:")
    for i in range(delay_embedding_dim):
        start = batch_starts[1] + i * stride
        end = start + window_len
        print(f"  Window {i}: indices [{start}:{end}] = {signal[start:end]}")
    
    # Test reconstruction
    reconstructed = reconstruct_from_3d_hankel(
        hankel_matrix, batch_starts, stride, len(signal)
    )
    
    print(f"\nReconstruction error: {np.mean((signal - reconstructed)**2):.6f}")


if __name__ == "__main__":
    demonstrate_hankel_construction()