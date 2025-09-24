"""
3D Hankel Matrix with Stride = 5
Creates Hankel matrices with stride-based delay embedding using stride = 5
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full


def create_hankel_with_stride_5():
    """Create 3D Hankel matrix with stride = 5"""
    
    # Generate Lorenz signal
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]
    
    print(f"Signal length: {len(x)}")
    
    # Create 3D Hankel matrix with stride = 5
    dataset = Hankel3DDataset(
        signal=x, 
        window_len=512, 
        delay_embedding_dim=10, 
        stride=5,  # Stride = 5 as requested
        normalize=True, 
        shuffle=True
    )
    
    print(f"Hankel matrix shape: {dataset.hankel_matrix.shape}")
    print(f"Number of batches: {dataset.hankel_matrix.shape[0]}")
    print(f"Delay embedding dimension: {dataset.hankel_matrix.shape[1]}")
    print(f"Window length: {dataset.hankel_matrix.shape[2]}")
    print(f"Stride: {dataset.stride}")
    
    return dataset, x, t


def demonstrate_stride_5_pattern(dataset):
    """Show the delay embedding pattern with stride = 5"""
    print("\n=== Delay Embedding Pattern (Stride = 5) ===")
    
    # Show first few batches
    for batch_idx in range(3):
        print(f"\nBatch {batch_idx}:")
        batch_start = dataset.batch_starts[batch_idx]
        
        for delay_idx in range(dataset.delay_embedding_dim):
            window_start = batch_start + delay_idx * dataset.stride
            window_end = window_start + dataset.window_len
            print(f"  Delay {delay_idx}: window [{window_start}:{window_end}]")
    
    print(f"\nPattern explanation:")
    print(f"- Each batch contains {dataset.delay_embedding_dim} overlapping windows")
    print(f"- Within each batch, windows are shifted by {dataset.stride} samples")
    print(f"- Between batches, the starting position shifts by {dataset.stride} samples")
    print(f"- This creates less overlap compared to stride=1, but maintains delay embedding structure")


def compare_stride_effects():
    """Compare different stride values"""
    print("\n=== Stride Comparison ===")
    
    # Generate signal
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]
    
    strides = [1, 5, 10]
    
    for stride in strides:
        dataset = Hankel3DDataset(
            signal=x, 
            window_len=512, 
            delay_embedding_dim=10, 
            stride=stride,
            normalize=True, 
            shuffle=False  # Don't shuffle for comparison
        )
        
        print(f"Stride {stride}: Hankel shape {dataset.hankel_matrix.shape}")
        print(f"  Number of batches: {dataset.hankel_matrix.shape[0]}")
        print(f"  Total windows: {dataset.hankel_matrix.shape[0] * dataset.hankel_matrix.shape[1]}")


def test_reconstruction_quality(dataset, original_signal):
    """Test reconstruction quality with stride = 5"""
    print("\n=== Reconstruction Quality Test ===")
    
    reconstructed = reconstruct_from_3d_hankel(
        dataset.hankel_matrix, 
        dataset.batch_starts, 
        dataset.stride, 
        len(original_signal),
        mean=dataset.mean,
        std=dataset.std
    )
    
    mse = np.mean((original_signal - reconstructed)**2)
    mae = np.mean(np.abs(original_signal - reconstructed))
    max_error = np.max(np.abs(original_signal - reconstructed))
    
    print(f"Reconstruction MSE: {mse:.6e}")
    print(f"Reconstruction MAE: {mae:.6e}")
    print(f"Maximum error: {max_error:.6e}")
    print(f"Signal range: [{original_signal.min():.3f}, {original_signal.max():.3f}]")
    
    return reconstructed


def visualize_stride_5_results(original_signal, reconstructed_signal, time_axis):
    """Visualize results for stride = 5"""
    plt.figure(figsize=(15, 10))
    
    # Plot full signals
    plt.subplot(3, 2, 1)
    plt.plot(time_axis, original_signal, label='Original', alpha=0.7, linewidth=1)
    plt.plot(time_axis, reconstructed_signal, label='Reconstructed (Stride=5)', alpha=0.7, linewidth=1)
    plt.title('Full Signal Reconstruction (Stride = 5)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot zoomed view
    plt.subplot(3, 2, 2)
    zoom_start, zoom_end = 500, 1000
    plt.plot(time_axis[zoom_start:zoom_end], original_signal[zoom_start:zoom_end], 
             label='Original', alpha=0.7, linewidth=1)
    plt.plot(time_axis[zoom_start:zoom_end], reconstructed_signal[zoom_start:zoom_end], 
             label='Reconstructed', alpha=0.7, linewidth=1)
    plt.title('Zoomed Reconstruction (samples 500-1000)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(3, 2, 3)
    error = original_signal - reconstructed_signal
    plt.plot(time_axis, error, color='red', alpha=0.7, linewidth=1)
    plt.title('Reconstruction Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    # Plot scatter plot
    plt.subplot(3, 2, 4)
    plt.scatter(original_signal, reconstructed_signal, alpha=0.5, s=1)
    plt.plot([original_signal.min(), original_signal.max()], 
             [original_signal.min(), original_signal.max()], 'r--', alpha=0.7)
    plt.title('Original vs Reconstructed Scatter')
    plt.xlabel('Original')
    plt.ylabel('Reconstructed')
    plt.grid(True, alpha=0.3)
    
    # Plot error histogram
    plt.subplot(3, 2, 5)
    plt.hist(error, bins=50, alpha=0.7, color='red')
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot power spectral density comparison
    plt.subplot(3, 2, 6)
    from scipy import signal
    f_orig, psd_orig = signal.welch(original_signal, fs=100, nperseg=256)
    f_recon, psd_recon = signal.welch(reconstructed_signal, fs=100, nperseg=256)
    plt.semilogy(f_orig, psd_orig, label='Original', alpha=0.7)
    plt.semilogy(f_recon, psd_recon, label='Reconstructed', alpha=0.7)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("=== 3D Hankel Matrix with Stride = 5 ===")
    print("Creating Hankel matrix with stride-based delay embedding\n")
    
    # Create the Hankel matrix
    dataset, original_signal, time_axis = create_hankel_with_stride_5()
    
    # Demonstrate the pattern
    demonstrate_stride_5_pattern(dataset)
    
    # Compare different strides
    compare_stride_effects()
    
    # Test reconstruction
    reconstructed_signal = test_reconstruction_quality(dataset, original_signal)
    
    # Visualize results
    visualize_stride_5_results(original_signal, reconstructed_signal, time_axis)
    
    print("\n=== Summary ===")
    print(f"✓ Created Hankel matrix with shape: {dataset.hankel_matrix.shape}")
    print(f"✓ Delay embedding dimension: {dataset.hankel_matrix.shape[1]}")
    print(f"✓ Window length: {dataset.hankel_matrix.shape[2]}")
    print(f"✓ Stride: {dataset.stride}")
    print(f"✓ Number of batches: {dataset.hankel_matrix.shape[0]}")
    print(f"✓ Shuffling enabled for generalization")
    print(f"✓ Signal reconstruction implemented")


if __name__ == "__main__":
    main()
