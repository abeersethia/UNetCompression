"""
Reversed Adaptive Noise Training for 3D Hankel Matrix Autoencoder
Starts with low noise and gradually increases to high noise during training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def get_reversed_adaptive_noise_std(epoch, max_epochs, base_std=0.1):
    """
    Reversed adaptive noise function
    Starts with low noise and gradually increases to high noise
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        base_std: Maximum noise standard deviation
    
    Returns:
        noise_std: Current noise standard deviation
    """
    return base_std * (epoch / max_epochs) + 0.01


def create_improved_autoencoder(input_dim, latent_dim=64):
    """Create improved autoencoder with BatchNorm and better architecture"""
    
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(256),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, latent_dim)
    )
    
    decoder = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(256),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, input_dim)
    )
    
    return encoder, decoder


def train_with_reversed_adaptive_noise(encoder, decoder, train_data, test_data, 
                                     n_epochs=150, base_noise_std=0.1):
    """
    Train autoencoder with reversed adaptive noise
    
    Args:
        encoder: Encoder network
        decoder: Decoder network
        train_data: Training data
        test_data: Test data
        n_epochs: Number of training epochs
        base_noise_std: Maximum noise standard deviation
    
    Returns:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        noise_levels: Noise levels per epoch
        best_model_state: Best model state dict
    """
    
    train_tensor = torch.from_numpy(train_data).float()
    test_tensor = torch.from_numpy(test_data).float()
    
    # Improved optimizer and scheduler
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), 
                                lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    noise_levels = []
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_model_state = None
    
    print(f'Training with REVERSED adaptive noise (0.01 → {base_noise_std:.2f})...')
    
    for epoch in range(n_epochs):
        # Get reversed adaptive noise level
        noise_std = get_reversed_adaptive_noise_std(epoch, n_epochs, base_noise_std)
        noise_levels.append(noise_std)
        
        # Training with reversed adaptive noise
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        
        # Add Gaussian noise to input (but not to target)
        noisy_input = train_tensor + torch.randn_like(train_tensor) * noise_std
        train_latent = encoder(noisy_input)
        train_reconstructed = decoder(train_latent)
        train_loss = criterion(train_reconstructed, train_tensor)
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()
        
        # Validation without noise
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_latent = encoder(test_tensor)
            val_reconstructed = decoder(val_latent)
            val_loss = criterion(val_reconstructed, test_tensor)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                'encoder': encoder.state_dict().copy(),
                'decoder': decoder.state_dict().copy()
            }
        else:
            patience_counter += 1
        
        if (epoch + 1) % 30 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}, Noise: {noise_std:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses, noise_levels, best_model_state


def evaluate_model(encoder, decoder, train_data, test_data, dataset, train_batch_starts, test_batch_starts):
    """Evaluate the trained model"""
    
    train_tensor = torch.from_numpy(train_data).float()
    test_tensor = torch.from_numpy(test_data).float()
    
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        
        train_latent = encoder(train_tensor)
        train_reconstructed = decoder(train_latent).numpy()
        
        test_latent = encoder(test_tensor)
        test_reconstructed = decoder(test_latent).numpy()
    
    # Reshape back to Hankel matrix shape
    n_batches, delay_dim, window_len = dataset.hankel_matrix.shape
    train_recon_hankel = train_reconstructed.reshape(len(train_data), delay_dim, window_len)
    test_recon_hankel = test_reconstructed.reshape(len(test_data), delay_dim, window_len)
    
    # Reconstruct signals
    train_recon_signal = reconstruct_from_3d_hankel(
        train_recon_hankel, train_batch_starts, dataset.stride, len(dataset.signal),
        mean=dataset.mean, std=dataset.std
    )
    
    test_recon_signal = reconstruct_from_3d_hankel(
        test_recon_hankel, test_batch_starts, dataset.stride, len(dataset.signal),
        mean=dataset.mean, std=dataset.std
    )
    
    return train_recon_signal, test_recon_signal


def create_comprehensive_visualization(t, x, train_recon_signal, test_recon_signal, 
                                     train_losses, val_losses, noise_levels,
                                     train_mse, test_mse, train_corr, test_corr,
                                     overfitting_ratio, train_split):
    """Create comprehensive visualization of results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    train_time_end = int(len(x) * train_split)
    train_time = t[:train_time_end]
    test_time = t[train_time_end:]
    
    # 1. Signal reconstruction with train/test split
    axes[0, 0].plot(t, x, label='Original Signal', alpha=0.8, linewidth=1.5, color='black')
    axes[0, 0].plot(t, train_recon_signal, label='Train Reconstructed', alpha=0.8, linewidth=1.5, color='blue')
    axes[0, 0].plot(t, test_recon_signal, label='Test Reconstructed', alpha=0.8, linewidth=1.5, color='red')
    
    # Add vertical line to separate train/test
    axes[0, 0].axvline(x=train_time_end * 0.01, color='gray', linestyle='--', alpha=0.8, linewidth=3)
    axes[0, 0].text(train_time_end * 0.01, axes[0, 0].get_ylim()[1]*0.9, 'Train/Test Split', 
                    rotation=90, verticalalignment='top', fontsize=12, color='gray', fontweight='bold')
    
    # Add background colors for train/test regions
    axes[0, 0].axvspan(0, train_time_end * 0.01, alpha=0.1, color='blue', label='Train Region')
    axes[0, 0].axvspan(train_time_end * 0.01, t[-1], alpha=0.1, color='red', label='Test Region')
    
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Signal Reconstruction with REVERSED Adaptive Noise\\n(Low → High Noise)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Train segment only
    axes[0, 1].plot(train_time, x[:train_time_end], label='Original Train', alpha=0.8, linewidth=1.5, color='black')
    axes[0, 1].plot(train_time, train_recon_signal[:train_time_end], label='Reconstructed Train', alpha=0.8, linewidth=1.5, color='blue')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Train Segment ({train_split*100:.0f}%)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Test segment only
    axes[0, 2].plot(test_time, x[train_time_end:], label='Original Test', alpha=0.8, linewidth=1.5, color='black')
    axes[0, 2].plot(test_time, test_recon_signal[train_time_end:], label='Reconstructed Test', alpha=0.8, linewidth=1.5, color='red')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].set_title(f'Test Segment ({(1-train_split)*100:.0f}%)', fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training curves
    axes[1, 0].plot(train_losses, label='Train Loss', alpha=0.8, linewidth=1.5, color='blue')
    axes[1, 0].plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=1.5, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Curves\\n(Reversed Adaptive Noise)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 5. REVERSED adaptive noise schedule
    axes[1, 1].plot(noise_levels, alpha=0.8, linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Noise Standard Deviation')
    axes[1, 1].set_title('REVERSED Adaptive Noise Schedule\\n(0.01 → 0.11)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 0.12)
    
    # 6. Reconstruction error
    train_error = np.abs(x[:train_time_end] - train_recon_signal[:train_time_end])
    test_error = np.abs(x[train_time_end:] - test_recon_signal[train_time_end:])
    
    axes[1, 2].plot(train_time, train_error, label='Train Error', alpha=0.8, linewidth=1.5, color='blue')
    axes[1, 2].plot(test_time, test_error, label='Test Error', alpha=0.8, linewidth=1.5, color='red')
    axes[1, 2].axvline(x=train_time_end * 0.01, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Absolute Error')
    axes[1, 2].set_title('Reconstruction Error\\n(Reversed Noise)', fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Error distribution
    axes[2, 0].hist(train_error, bins=50, alpha=0.7, label='Train Error', color='blue', density=True)
    axes[2, 0].hist(test_error, bins=50, alpha=0.7, label='Test Error', color='red', density=True)
    axes[2, 0].set_xlabel('Absolute Error')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title('Error Distribution\\n(Reversed Noise)', fontsize=14)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Scatter plot comparison
    axes[2, 1].scatter(x[:train_time_end], train_recon_signal[:train_time_end], alpha=0.5, s=10, color='blue', label='Train')
    axes[2, 1].scatter(x[train_time_end:], test_recon_signal[train_time_end:], alpha=0.5, s=10, color='red', label='Test')
    axes[2, 1].plot([x.min(), x.max()], [x.min(), x.max()], 'k--', alpha=0.7)
    axes[2, 1].set_xlabel('Original Signal')
    axes[2, 1].set_ylabel('Reconstructed Signal')
    axes[2, 1].set_title('Original vs Reconstructed\\n(Reversed Noise)', fontsize=14)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Performance metrics
    axes[2, 2].axis('off')
    metrics_text = f'''REVERSED Adaptive Noise Results:

TRAIN SEGMENT ({train_split*100:.0f}%):
MSE: {train_mse:.6f}
MAE: {np.mean(np.abs(x[:train_time_end] - train_recon_signal[:train_time_end])):.6f}
Correlation: {train_corr:.6f}

TEST SEGMENT ({(1-train_split)*100:.0f}%):
MSE: {test_mse:.6f}
MAE: {np.mean(np.abs(x[train_time_end:] - test_recon_signal[train_time_end:])):.6f}
Correlation: {test_corr:.6f}

OVERFITTING ANALYSIS:
Overfitting Ratio: {overfitting_ratio:.2f}

GENERALIZATION GAP:
MSE Difference: {abs(test_mse - train_mse):.6f}

REVERSED NOISE SCHEDULE:
Start: {noise_levels[0]:.4f} (low)
End: {noise_levels[-1]:.4f} (high)
Schedule: Linear increase

APPROACH:
✓ Start with clean data (low noise)
✓ Gradually increase noise
✓ Learn fine details first
✓ Build robustness to noise'''

    axes[2, 2].text(0.05, 0.95, metrics_text, transform=axes[2, 2].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("=== Reversed Adaptive Noise Training ===")
    print("Starting with low noise and gradually increasing to high noise\\n")
    
    # Generate signal and create dataset
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]
    
    dataset = Hankel3DDataset(
        signal=x, 
        window_len=512, 
        delay_embedding_dim=10, 
        stride=5,
        normalize=True, 
        shuffle=True
    )
    
    # Prepare data
    hankel_data = dataset.hankel_matrix
    n_batches, delay_dim, window_len = hankel_data.shape
    flattened_data = hankel_data.reshape(n_batches, -1)
    
    input_dim = flattened_data.shape[1]
    latent_dim = 64
    
    # Train/test split
    train_split = 0.7
    n_train_batches = int(n_batches * train_split)
    n_test_batches = n_batches - n_train_batches
    
    train_indices = dataset.indices[:n_train_batches]
    test_indices = dataset.indices[n_train_batches:]
    
    train_data = flattened_data[train_indices]
    test_data = flattened_data[test_indices]
    
    train_batch_starts = [dataset.batch_starts[i] for i in train_indices]
    test_batch_starts = [dataset.batch_starts[i] for i in test_indices]
    
    print(f"Data split: {len(train_data)} train, {len(test_data)} test batches")
    
    # Create and train model
    encoder, decoder = create_improved_autoencoder(input_dim, latent_dim)
    
    train_losses, val_losses, noise_levels, best_model_state = train_with_reversed_adaptive_noise(
        encoder, decoder, train_data, test_data, n_epochs=150, base_noise_std=0.1
    )
    
    # Load best model
    encoder.load_state_dict(best_model_state['encoder'])
    decoder.load_state_dict(best_model_state['decoder'])
    
    # Evaluate model
    train_recon_signal, test_recon_signal = evaluate_model(
        encoder, decoder, train_data, test_data, dataset, train_batch_starts, test_batch_starts
    )
    
    # Calculate metrics
    train_time_end = int(len(x) * train_split)
    
    train_mse = mean_squared_error(x[:train_time_end], train_recon_signal[:train_time_end])
    train_mae = mean_absolute_error(x[:train_time_end], train_recon_signal[:train_time_end])
    train_corr = pearsonr(x[:train_time_end], train_recon_signal[:train_time_end])[0]
    
    test_mse = mean_squared_error(x[train_time_end:], test_recon_signal[train_time_end:])
    test_mae = mean_absolute_error(x[train_time_end:], test_recon_signal[train_time_end:])
    test_corr = pearsonr(x[train_time_end:], test_recon_signal[train_time_end:])[0]
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    overfitting_ratio = final_val_loss / final_train_loss
    
    # Create visualization
    create_comprehensive_visualization(
        t, x, train_recon_signal, test_recon_signal,
        train_losses, val_losses, noise_levels,
        train_mse, test_mse, train_corr, test_corr,
        overfitting_ratio, train_split
    )
    
    # Print results
    print("\\n=== REVERSED ADAPTIVE NOISE RESULTS ===")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Train Correlation: {train_corr:.6f}")
    print(f"Test Correlation: {test_corr:.6f}")
    print(f"Generalization gap: {abs(test_mse - train_mse):.6f}")
    print(f"Overfitting ratio: {overfitting_ratio:.2f}")
    print(f"Noise range: {noise_levels[0]:.4f} → {noise_levels[-1]:.4f}")
    print()
    print("NOISE SCHEDULE COMPARISON:")
    print("Original (High→Low): 0.11 → 0.01")
    print("Reversed (Low→High): 0.01 → 0.11")


if __name__ == "__main__":
    main()
