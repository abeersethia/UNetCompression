"""
Reconstructed Manifold Analysis
Analyzes how well the latent space can reconstruct the original 3D Hankel matrix
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns

from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full


def create_autoencoder_model(input_dim, latent_dim, output_dim):
    """Create encoder-decoder model"""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, latent_dim)
    )
    
    decoder = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, output_dim)
    )
    
    return encoder, decoder


def train_autoencoder(encoder, decoder, hankel_data, n_epochs=100, lr=1e-3):
    """Train the autoencoder"""
    print("Training autoencoder...")
    
    # Convert to tensor
    hankel_tensor = torch.from_numpy(hankel_data).float()
    
    # Create optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        latent = encoder(hankel_tensor)
        reconstructed = decoder(latent)
        
        # Compute loss
        loss = criterion(reconstructed, hankel_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    return losses


def analyze_reconstruction_quality(original_hankel, reconstructed_hankel, dataset):
    """Analyze the quality of reconstruction"""
    print("\n=== Reconstruction Quality Analysis ===")
    
    # Flatten for analysis
    orig_flat = original_hankel.reshape(original_hankel.shape[0], -1)
    recon_flat = reconstructed_hankel.reshape(reconstructed_hankel.shape[0], -1)
    
    # Basic statistics
    print(f"Original Hankel shape: {original_hankel.shape}")
    print(f"Reconstructed Hankel shape: {reconstructed_hankel.shape}")
    
    # Per-sample reconstruction quality
    mse_per_sample = np.mean((orig_flat - recon_flat)**2, axis=1)
    mae_per_sample = np.mean(np.abs(orig_flat - recon_flat), axis=1)
    
    print(f"\nPer-sample reconstruction metrics:")
    print(f"  MSE - Mean: {mse_per_sample.mean():.6f}, Std: {mse_per_sample.std():.6f}")
    print(f"  MAE - Mean: {mae_per_sample.mean():.6f}, Std: {mae_per_sample.std():.6f}")
    
    # Overall reconstruction quality
    overall_mse = mean_squared_error(orig_flat.flatten(), recon_flat.flatten())
    overall_mae = mean_absolute_error(orig_flat.flatten(), recon_flat.flatten())
    
    print(f"\nOverall reconstruction metrics:")
    print(f"  MSE: {overall_mse:.6f}")
    print(f"  MAE: {overall_mae:.6f}")
    print(f"  RMSE: {np.sqrt(overall_mse):.6f}")
    
    # Correlation analysis
    correlation = pearsonr(orig_flat.flatten(), recon_flat.flatten())[0]
    print(f"  Correlation: {correlation:.6f}")
    
    # Per-dimension analysis
    print(f"\nPer-dimension reconstruction quality:")
    for i in range(min(5, orig_flat.shape[1])):
        dim_corr = pearsonr(orig_flat[:, i], recon_flat[:, i])[0]
        dim_mse = mean_squared_error(orig_flat[:, i], recon_flat[:, i])
        print(f"  Dim {i}: MSE={dim_mse:.6f}, Corr={dim_corr:.6f}")
    
    return mse_per_sample, mae_per_sample, overall_mse, overall_mae, correlation


def visualize_reconstruction(original_hankel, reconstructed_hankel, dataset, original_signal, time_axis):
    """Create comprehensive visualizations of reconstruction quality"""
    print("\n=== Creating Reconstruction Visualizations ===")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 1. Original vs Reconstructed Hankel matrix (first batch)
    axes[0, 0].imshow(original_hankel[0], aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_title('Original Hankel Matrix (Batch 0)')
    axes[0, 0].set_xlabel('Window Length')
    axes[0, 0].set_ylabel('Delay Embedding')
    
    axes[0, 1].imshow(reconstructed_hankel[0], aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_title('Reconstructed Hankel Matrix (Batch 0)')
    axes[0, 1].set_xlabel('Window Length')
    axes[0, 1].set_ylabel('Delay Embedding')
    
    # 2. Reconstruction error heatmap
    error = np.abs(original_hankel[0] - reconstructed_hankel[0])
    im = axes[0, 2].imshow(error, aspect='auto', cmap='Reds')
    axes[0, 2].set_title('Reconstruction Error (Batch 0)')
    axes[0, 2].set_xlabel('Window Length')
    axes[0, 2].set_ylabel('Delay Embedding')
    plt.colorbar(im, ax=axes[0, 2], label='Absolute Error')
    
    # 3. Reconstruction error distribution
    axes[0, 3].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 3].set_xlabel('Absolute Error')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_title('Error Distribution (Batch 0)')
    axes[0, 3].grid(True, alpha=0.3)
    
    # 4. Original vs Reconstructed signal reconstruction
    orig_recon_signal = reconstruct_from_3d_hankel(
        original_hankel, dataset.batch_starts, dataset.stride, len(original_signal),
        mean=dataset.mean, std=dataset.std
    )
    
    recon_recon_signal = reconstruct_from_3d_hankel(
        reconstructed_hankel, dataset.batch_starts, dataset.stride, len(original_signal),
        mean=dataset.mean, std=dataset.std
    )
    
    axes[1, 0].plot(time_axis, original_signal, label='Original Signal', alpha=0.8, linewidth=1.5)
    axes[1, 0].plot(time_axis, orig_recon_signal, label='From Original Hankel', alpha=0.8, linewidth=1.5)
    axes[1, 0].plot(time_axis, recon_recon_signal, label='From Reconstructed Hankel', alpha=0.8, linewidth=1.5)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Signal Reconstruction Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Zoomed signal comparison
    zoom_start, zoom_end = 500, 1000
    axes[1, 1].plot(time_axis[zoom_start:zoom_end], original_signal[zoom_start:zoom_end], 
                    label='Original', alpha=0.8, linewidth=1.5)
    axes[1, 1].plot(time_axis[zoom_start:zoom_end], recon_recon_signal[zoom_start:zoom_end], 
                    label='Reconstructed', alpha=0.8, linewidth=1.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Zoomed Signal Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scatter plot: Original vs Reconstructed
    orig_flat = original_hankel.reshape(original_hankel.shape[0], -1)
    recon_flat = reconstructed_hankel.reshape(reconstructed_hankel.shape[0], -1)
    
    # Sample some points for visualization
    sample_indices = np.random.choice(orig_flat.shape[0], min(1000, orig_flat.shape[0]), replace=False)
    axes[1, 2].scatter(orig_flat[sample_indices, 0], recon_flat[sample_indices, 0], 
                       alpha=0.5, s=10)
    axes[1, 2].plot([orig_flat.min(), orig_flat.max()], [orig_flat.min(), orig_flat.max()], 
                    'r--', alpha=0.7)
    axes[1, 2].set_xlabel('Original Value')
    axes[1, 2].set_ylabel('Reconstructed Value')
    axes[1, 2].set_title('Original vs Reconstructed Scatter')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Per-sample reconstruction error
    mse_per_sample = np.mean((orig_flat - recon_flat)**2, axis=1)
    axes[1, 3].plot(mse_per_sample, alpha=0.8, linewidth=1.5)
    axes[1, 3].set_xlabel('Sample Index')
    axes[1, 3].set_ylabel('MSE')
    axes[1, 3].set_title('Per-Sample Reconstruction Error')
    axes[1, 3].grid(True, alpha=0.3)
    
    # 8. Latent space visualization
    with torch.no_grad():
        encoder, _ = create_autoencoder_model(orig_flat.shape[1], 32, orig_flat.shape[1])
        latent_orig = encoder(torch.from_numpy(orig_flat).float()).numpy()
        latent_recon = encoder(torch.from_numpy(recon_flat).float()).numpy()
    
    axes[2, 0].scatter(latent_orig[:, 0], latent_orig[:, 1], c='blue', alpha=0.7, s=20, label='Original')
    axes[2, 0].scatter(latent_recon[:, 0], latent_recon[:, 1], c='red', alpha=0.7, s=20, label='Reconstructed')
    axes[2, 0].set_xlabel('Latent Dim 1')
    axes[2, 0].set_ylabel('Latent Dim 2')
    axes[2, 0].set_title('Latent Space Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 9. Reconstruction error over time
    signal_error = np.abs(original_signal - recon_recon_signal)
    axes[2, 1].plot(time_axis, signal_error, alpha=0.8, linewidth=1.5, color='red')
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Absolute Error')
    axes[2, 1].set_title('Signal Reconstruction Error Over Time')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 10. Power spectral density comparison
    from scipy import signal
    f_orig, psd_orig = signal.welch(original_signal, fs=100, nperseg=256)
    f_recon, psd_recon = signal.welch(recon_recon_signal, fs=100, nperseg=256)
    axes[2, 2].semilogy(f_orig, psd_orig, label='Original', alpha=0.8, linewidth=1.5)
    axes[2, 2].semilogy(f_recon, psd_recon, label='Reconstructed', alpha=0.8, linewidth=1.5)
    axes[2, 2].set_xlabel('Frequency (Hz)')
    axes[2, 2].set_ylabel('PSD')
    axes[2, 2].set_title('Power Spectral Density')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # 11. Reconstruction quality metrics
    overall_mse = mean_squared_error(orig_flat.flatten(), recon_flat.flatten())
    overall_mae = mean_absolute_error(orig_flat.flatten(), recon_flat.flatten())
    correlation = pearsonr(orig_flat.flatten(), recon_flat.flatten())[0]
    
    axes[2, 3].text(0.1, 0.8, f'MSE: {overall_mse:.6f}', transform=axes[2, 3].transAxes, fontsize=12)
    axes[2, 3].text(0.1, 0.7, f'MAE: {overall_mae:.6f}', transform=axes[2, 3].transAxes, fontsize=12)
    axes[2, 3].text(0.1, 0.6, f'RMSE: {np.sqrt(overall_mse):.6f}', transform=axes[2, 3].transAxes, fontsize=12)
    axes[2, 3].text(0.1, 0.5, f'Correlation: {correlation:.6f}', transform=axes[2, 3].transAxes, fontsize=12)
    axes[2, 3].text(0.1, 0.4, f'Compression: {orig_flat.shape[1]/32:.1f}:1', transform=axes[2, 3].transAxes, fontsize=12)
    axes[2, 3].set_title('Reconstruction Metrics')
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return orig_recon_signal, recon_recon_signal


def main():
    """Main execution function for reconstructed manifold analysis"""
    print("=== Reconstructed Manifold Analysis ===")
    print("Analyzing reconstruction quality from latent space\n")
    
    # Generate Lorenz signal
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]
    
    # Create 3D Hankel matrix dataset
    dataset = Hankel3DDataset(
        signal=x, 
        window_len=512, 
        delay_embedding_dim=10, 
        stride=5,
        normalize=True, 
        shuffle=False
    )
    
    print(f"Hankel matrix shape: {dataset.hankel_matrix.shape}")
    
    # Prepare data for autoencoder
    hankel_data = dataset.hankel_matrix
    n_batches, delay_dim, window_len = hankel_data.shape
    flattened_data = hankel_data.reshape(n_batches, -1)
    
    input_dim = flattened_data.shape[1]
    latent_dim = 32
    output_dim = input_dim
    
    print(f"Input dimension: {input_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Compression ratio: {input_dim/latent_dim:.1f}:1")
    
    # Create and train autoencoder
    encoder, decoder = create_autoencoder_model(input_dim, latent_dim, output_dim)
    losses = train_autoencoder(encoder, decoder, flattened_data, n_epochs=100)
    
    # Get reconstructed data
    with torch.no_grad():
        latent = encoder(torch.from_numpy(flattened_data).float())
        reconstructed_flat = decoder(latent).numpy()
    
    # Reshape back to original Hankel matrix shape
    reconstructed_hankel = reconstructed_flat.reshape(hankel_data.shape)
    
    # Analyze reconstruction quality
    mse_per_sample, mae_per_sample, overall_mse, overall_mae, correlation = analyze_reconstruction_quality(
        hankel_data, reconstructed_hankel, dataset
    )
    
    # Create visualizations
    orig_recon_signal, recon_recon_signal = visualize_reconstruction(
        hankel_data, reconstructed_hankel, dataset, x, t
    )
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"✓ Original Hankel shape: {hankel_data.shape}")
    print(f"✓ Reconstructed Hankel shape: {reconstructed_hankel.shape}")
    print(f"✓ Compression ratio: {input_dim/latent_dim:.1f}:1")
    print(f"✓ Overall MSE: {overall_mse:.6f}")
    print(f"✓ Overall MAE: {overall_mae:.6f}")
    print(f"✓ Correlation: {correlation:.6f}")
    print(f"✓ Signal reconstruction MSE: {mean_squared_error(x, recon_recon_signal):.6f}")
    print(f"✓ Ready for manifold analysis!")


if __name__ == "__main__":
    main()
