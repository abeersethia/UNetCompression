"""
Analysis script to examine reconstruction artifacts and quality
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from direct_manifold_model import DirectManifoldAutoencoder
from hankel_dataset import HankelMatrixDataset
from lorenz import generate_lorenz_full
from direct_manifold_training import reconstruct_direct_signal


def analyze_reconstruction_quality():
    """Analyze the quality of reconstruction and identify artifacts"""
    
    # Generate Lorenz system data
    print("Generating Lorenz system data...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    x = traj[:, 0]  # Only x-component
    
    # Create dataset and dataloader
    window_len = 512
    stride = 1
    dataset = HankelMatrixDataset(x, window_len=window_len, stride=stride, normalize=True)
    hankel_tensor = torch.from_numpy(dataset.inputs).float()
    dataloader = DataLoader([hankel_tensor], batch_size=1, shuffle=False)
    
    # Load trained model
    print("Loading trained model...")
    hankel_size = dataset.inputs.size
    signal_length = len(x)
    
    model = DirectManifoldAutoencoder(
        hankel_size=hankel_size,
        signal_length=signal_length,
        latent_dim=32
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('direct_manifold_autoencoder.pth', map_location='cpu'))
    model.eval()
    
    # Reconstruct signal
    print("Reconstructing signal...")
    recon, latent = reconstruct_direct_signal(model, dataloader, device='cpu')
    recon_signal = recon[0]  # Remove batch dimension
    
    # Calculate detailed metrics
    print("\n=== RECONSTRUCTION QUALITY ANALYSIS ===")
    
    # Basic metrics
    mse = np.mean((x - recon_signal) ** 2)
    mae = np.mean(np.abs(x - recon_signal))
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error (MSE): {mse:.6e}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6e}")
    print(f"Mean Absolute Error (MAE): {mae:.6e}")
    
    # Signal-to-Noise Ratio
    signal_power = np.mean(x ** 2)
    noise_power = np.mean((x - recon_signal) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")
    
    # Correlation coefficient
    correlation = np.corrcoef(x, recon_signal)[0, 1]
    print(f"Correlation Coefficient: {correlation:.6f}")
    
    # Frequency domain analysis
    print("\n=== FREQUENCY DOMAIN ANALYSIS ===")
    
    # FFT of original and reconstructed
    fft_original = np.fft.fft(x)
    fft_recon = np.fft.fft(recon_signal)
    freqs = np.fft.fftfreq(len(x), d=0.01)
    
    # Power spectral density
    psd_original = np.abs(fft_original) ** 2
    psd_recon = np.abs(fft_recon) ** 2
    
    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain comparison
    axes[0, 0].plot(t, x, label='Original', alpha=0.7, linewidth=1)
    axes[0, 0].plot(t, recon_signal, label='Reconstructed', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Time Domain Comparison')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('x(t)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error signal
    error_signal = x - recon_signal
    axes[0, 1].plot(t, error_signal, color='red', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Reconstruction Error (Original - Reconstructed)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency domain comparison
    axes[1, 0].semilogy(freqs[:len(freqs)//2], psd_original[:len(freqs)//2], 
                       label='Original', alpha=0.7)
    axes[1, 0].semilogy(freqs[:len(freqs)//2], psd_recon[:len(freqs)//2], 
                       label='Reconstructed', alpha=0.7)
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Reconstructed
    axes[1, 1].scatter(x, recon_signal, alpha=0.5, s=1)
    axes[1, 1].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', alpha=0.8)
    axes[1, 1].set_title(f'Original vs Reconstructed (R² = {correlation**2:.4f})')
    axes[1, 1].set_xlabel('Original x(t)')
    axes[1, 1].set_ylabel('Reconstructed x(t)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Zoomed view of a specific region
    print("\n=== ZOOMED ANALYSIS (Time 5-10 seconds) ===")
    start_idx = int(5.0 / 0.01)  # 5 seconds
    end_idx = int(10.0 / 0.01)   # 10 seconds
    
    plt.figure(figsize=(12, 6))
    plt.plot(t[start_idx:end_idx], x[start_idx:end_idx], 
             label='Original', alpha=0.8, linewidth=2)
    plt.plot(t[start_idx:end_idx], recon_signal[start_idx:end_idx], 
             label='Reconstructed', alpha=0.8, linewidth=2)
    plt.plot(t[start_idx:end_idx], error_signal[start_idx:end_idx], 
             label='Error', alpha=0.6, linewidth=1, color='red')
    plt.title('Zoomed View: Original vs Reconstructed (5-10 seconds)')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Artifact analysis
    print("\n=== ARTIFACT ANALYSIS ===")
    
    # High-frequency noise analysis
    high_freq_mask = np.abs(freqs) > 10  # Frequencies > 10 Hz
    high_freq_error = np.sum(psd_original[high_freq_mask]) - np.sum(psd_recon[high_freq_mask])
    print(f"High-frequency (>10Hz) power difference: {high_freq_error:.6e}")
    
    # Low-frequency preservation
    low_freq_mask = np.abs(freqs) <= 5  # Frequencies <= 5 Hz
    low_freq_correlation = np.corrcoef(
        psd_original[low_freq_mask], psd_recon[low_freq_mask]
    )[0, 1]
    print(f"Low-frequency (≤5Hz) PSD correlation: {low_freq_correlation:.6f}")
    
    # Peak detection and comparison
    from scipy.signal import find_peaks
    peaks_orig, _ = find_peaks(x, height=np.std(x))
    peaks_recon, _ = find_peaks(recon_signal, height=np.std(recon_signal))
    
    print(f"Number of peaks in original: {len(peaks_orig)}")
    print(f"Number of peaks in reconstructed: {len(peaks_recon)}")
    
    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr_db,
        'correlation': correlation,
        'error_signal': error_signal,
        'recon_signal': recon_signal
    }


if __name__ == "__main__":
    results = analyze_reconstruction_quality()
