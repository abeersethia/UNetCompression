"""
Main execution script for Direct Manifold-to-Signal Attractor Reconstruction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from direct_manifold_model import DirectManifoldAutoencoder
from hankel_dataset import HankelMatrixDataset
from lorenz import generate_lorenz_full
from direct_manifold_training import train_direct_manifold_model, reconstruct_direct_signal


def main():
    """Main execution function for direct manifold-to-signal reconstruction"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate Lorenz system data
    print("Generating Lorenz system data...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]  # Only x-component for attractor reconstruction
    y = traj[:, 1]  # Keep y for visualization
    z = traj[:, 2]  # Keep z for reference
    
    print(f"Generated signal length: {len(x)}")
    print(f"Using only x(t) component for direct manifold reconstruction")
    
    # Create Hankel matrix dataset
    print("Creating Hankel matrix dataset...")
    window_len = 512  # Window size for Hankel matrix
    stride = 1        # Overlapping windows
    
    dataset = HankelMatrixDataset(x, window_len=window_len, stride=stride, normalize=True)
    
    # Get the actual Hankel matrix shape
    hankel_matrix = dataset.inputs
    print(f"Hankel matrix shape: {hankel_matrix.shape}")
    
    # Create dataloader
    hankel_tensor = torch.from_numpy(hankel_matrix).float()
    dataloader = DataLoader([hankel_tensor], batch_size=1, shuffle=False)
    
    # Initialize direct manifold model
    print("Initializing Direct Manifold Autoencoder...")
    hankel_size = hankel_matrix.size  # Total elements in Hankel matrix
    signal_length = len(x)  # Length of target time signal
    
    model = DirectManifoldAutoencoder(
        hankel_size=hankel_size,
        signal_length=signal_length,
        latent_dim=32
    )
    
    print(f"Hankel matrix size: {hankel_size}")
    print(f"Target signal length: {signal_length}")
    print(f"Compression ratio: {hankel_size/signal_length:.1f}:1")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    print("Training Direct Manifold Autoencoder...")
    model = train_direct_manifold_model(
        model, 
        dataloader, 
        target_signal=x,  # Original time signal as target
        n_epochs=100,
        lr=1e-3,
        device=device
    )
    
    # Reconstruct signal directly from manifold
    print("Reconstructing signal directly from manifold...")
    recon, latent = reconstruct_direct_signal(model, dataloader, device=device)
    
    print(f"Reconstructed signal shape: {recon.shape}")
    print(f"Latent manifold shape: {latent.shape}")
    
    # Plot time-series reconstruction
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, label="Original", alpha=0.7)
    plt.plot(t, recon[0], label="Reconstructed (Direct)", alpha=0.7)
    plt.legend()
    plt.title("Lorenz x(t) Reconstruction - Direct Manifold-to-Signal")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot original vs reconstructed attractor
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Original", alpha=0.6)
    plt.plot(recon[0], y, label="Reconstructed (Direct)", alpha=0.6)
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Lorenz Attractor (Original vs Direct Reconstruction)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate and print reconstruction metrics
    mse = np.mean((x - recon[0]) ** 2)
    mae = np.mean(np.abs(x - recon[0]))
    
    print(f"\nReconstruction Metrics:")
    print(f"Mean Squared Error: {mse:.6e}")
    print(f"Mean Absolute Error: {mae:.6e}")
    
    # Save model
    torch.save(model.state_dict(), 'direct_manifold_autoencoder.pth')
    print("Model saved as 'direct_manifold_autoencoder.pth'")


if __name__ == "__main__":
    main()
