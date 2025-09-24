"""
Complete Example: 3D Hankel Matrix with (1489, 10, 512) dimensions
Demonstrates the stride-based delay embedding approach
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from src.core.lorenz import generate_lorenz_full


def create_1489_10_512_hankel():
    """Create exactly (1489, 10, 512) Hankel matrix"""
    
    # Calculate exact signal length needed for 1489 batches
    window_len = 512
    delay_embedding_dim = 10
    stride = 1
    target_batches = 1489
    
    # Required signal length calculation
    required_length = (target_batches - 1) + (delay_embedding_dim - 1) * stride + window_len
    dt = 0.01
    T = required_length * dt
    
    print(f"Creating Hankel matrix with dimensions (1489, 10, 512)")
    print(f"Required signal length: {required_length}")
    print(f"Time duration: {T:.2f}")
    
    # Generate Lorenz signal with exact length
    traj, t = generate_lorenz_full(T=T, dt=dt)
    x = traj[:, 0]
    
    print(f"Generated signal length: {len(x)}")
    
    # Create 3D Hankel matrix dataset
    dataset = Hankel3DDataset(
        signal=x, 
        window_len=window_len, 
        delay_embedding_dim=delay_embedding_dim, 
        stride=stride,
        normalize=True, 
        shuffle=True
    )
    
    print(f"Hankel matrix shape: {dataset.hankel_matrix.shape}")
    print(f"✓ Successfully created ({dataset.hankel_matrix.shape[0]}, {dataset.hankel_matrix.shape[1]}, {dataset.hankel_matrix.shape[2]}) Hankel matrix")
    
    return dataset, x, t


def demonstrate_delay_embedding(dataset):
    """Show how the delay embedding works"""
    print("\n=== Delay Embedding Demonstration ===")
    
    # Show first few batches
    for batch_idx in range(3):
        print(f"\nBatch {batch_idx}:")
        batch_start = dataset.batch_starts[batch_idx]
        
        for delay_idx in range(dataset.delay_embedding_dim):
            window_start = batch_start + delay_idx * dataset.stride
            window_end = window_start + dataset.window_len
            print(f"  Delay {delay_idx}: window [{window_start}:{window_end}]")


def test_reconstruction(dataset, original_signal):
    """Test signal reconstruction"""
    print("\n=== Reconstruction Test ===")
    
    reconstructed = reconstruct_from_3d_hankel(
        dataset.hankel_matrix, 
        dataset.batch_starts, 
        dataset.stride, 
        len(original_signal)
    )
    
    mse = np.mean((original_signal - reconstructed)**2)
    mae = np.mean(np.abs(original_signal - reconstructed))
    
    print(f"Reconstruction MSE: {mse:.6e}")
    print(f"Reconstruction MAE: {mae:.6e}")
    print(f"Signal range: [{original_signal.min():.3f}, {original_signal.max():.3f}]")
    
    return reconstructed


def visualize_results(original_signal, reconstructed_signal, time_axis):
    """Visualize original vs reconstructed signal"""
    plt.figure(figsize=(15, 8))
    
    # Plot full signals
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, original_signal, label='Original', alpha=0.7)
    plt.plot(time_axis, reconstructed_signal, label='Reconstructed', alpha=0.7)
    plt.title('Full Signal Reconstruction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot zoomed view
    plt.subplot(2, 2, 2)
    zoom_start, zoom_end = 500, 1000
    plt.plot(time_axis[zoom_start:zoom_end], original_signal[zoom_start:zoom_end], 
             label='Original', alpha=0.7)
    plt.plot(time_axis[zoom_start:zoom_end], reconstructed_signal[zoom_start:zoom_end], 
             label='Reconstructed', alpha=0.7)
    plt.title('Zoomed Reconstruction (samples 500-1000)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(2, 2, 3)
    error = original_signal - reconstructed_signal
    plt.plot(time_axis, error, color='red', alpha=0.7)
    plt.title('Reconstruction Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    # Plot scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(original_signal, reconstructed_signal, alpha=0.5, s=1)
    plt.plot([original_signal.min(), original_signal.max()], 
             [original_signal.min(), original_signal.max()], 'r--', alpha=0.7)
    plt.title('Original vs Reconstructed Scatter')
    plt.xlabel('Original')
    plt.ylabel('Reconstructed')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("=== 3D Hankel Matrix Implementation ===")
    print("Creating (1489, 10, 512) Hankel matrix with delay embedding\n")
    
    # Create the Hankel matrix
    dataset, original_signal, time_axis = create_1489_10_512_hankel()
    
    # Demonstrate delay embedding
    demonstrate_delay_embedding(dataset)
    
    # Test reconstruction
    reconstructed_signal = test_reconstruction(dataset, original_signal)
    
    # Visualize results
    visualize_results(original_signal, reconstructed_signal, time_axis)
    
    print("\n=== Summary ===")
    print(f"✓ Created Hankel matrix with shape: {dataset.hankel_matrix.shape}")
    print(f"✓ Delay embedding dimension: {dataset.hankel_matrix.shape[1]}")
    print(f"✓ Window length: {dataset.hankel_matrix.shape[2]}")
    print(f"✓ Stride-based construction: stride={dataset.stride}")
    print(f"✓ Shuffling enabled for generalization")
    print(f"✓ Signal reconstruction implemented")


if __name__ == "__main__":
    main()
