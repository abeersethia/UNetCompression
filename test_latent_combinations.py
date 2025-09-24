"""
Test Different Latent Space Combinations for X-Only Manifold Reconstruction

This script tests various combinations of latent dimensions (B, D, L) where:
- B: Batch size (fixed at 289)
- D: Feature dimensions (variable)
- L: Compressed signal length (variable)

Author: AI Assistant
Date: 2024
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time

# Import the corrected reconstructor
from x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def test_latent_combination(latent_d, latent_l, max_epochs=100, verbose=False):
    """
    Test a specific latent space combination
    
    Args:
        latent_d (int): Number of feature dimensions
        latent_l (int): Compressed signal length
        max_epochs (int): Maximum training epochs
        verbose (bool): Whether to print progress
    
    Returns:
        dict: Results including metrics and timing
    """
    print(f"\n=== TESTING LATENT COMBINATION: D={latent_d}, L={latent_l} ===")
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor
    reconstructor = XOnlyManifoldReconstructorCorrected(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        latent_d=latent_d,
        latent_l=latent_l,
        train_split=0.7
    )
    
    # Prepare data
    start_time = time.time()
    n_batches = reconstructor.prepare_data(traj, t)
    
    # Train the model
    training_start = time.time()
    training_history = reconstructor.train(max_epochs=max_epochs, verbose=verbose)
    training_time = time.time() - training_start
    
    # Reconstruct the manifold
    reconstruction_start = time.time()
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    reconstruction_time = time.time() - reconstruction_start
    
    # Get latent representations
    latent_all = reconstructor.get_latent_representations()
    
    total_time = time.time() - start_time
    
    # Calculate compression ratios
    temporal_compression = 512 / latent_l
    spatial_compression = 10 / latent_d
    total_compression = (512 * 10) / (latent_d * latent_l)
    
    results = {
        'latent_d': latent_d,
        'latent_l': latent_l,
        'latent_shape': latent_all.shape,
        'correlations': metrics['correlations'],
        'mse': metrics['mse'],
        'mean_error': metrics['mean_error'],
        'std_error': metrics['std_error'],
        'max_error': metrics['max_error'],
        'training_time': training_time,
        'reconstruction_time': reconstruction_time,
        'total_time': total_time,
        'temporal_compression': temporal_compression,
        'spatial_compression': spatial_compression,
        'total_compression': total_compression,
        'best_val_loss': training_history['best_val_loss'],
        'final_train_loss': training_history['train_losses'][-1],
        'final_val_loss': training_history['val_losses'][-1]
    }
    
    print(f"Results for D={latent_d}, L={latent_l}:")
    print(f"  X correlation: {metrics['correlations']['X']:.4f}")
    print(f"  Y correlation: {metrics['correlations']['Y']:.4f}")
    print(f"  Z correlation: {metrics['correlations']['Z']:.4f}")
    print(f"  Mean error: {metrics['mean_error']:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Compression: {total_compression:.1f}:1")
    
    return results

def main():
    """Test different latent space combinations"""
    print("=== TESTING DIFFERENT LATENT SPACE COMBINATIONS ===")
    print("Testing various (D, L) combinations for X-only manifold reconstruction")
    print()
    
    # Define test combinations
    test_combinations = [
        # (D, L) combinations
        (16, 64),   # High compression
        (16, 128),  # Medium compression
        (16, 256),  # Low compression
        (32, 64),   # High compression
        (32, 128),  # Medium compression (baseline)
        (32, 256),  # Low compression
        (64, 64),   # High compression
        (64, 128),  # Medium compression
        (64, 256),  # Low compression
    ]
    
    results = []
    
    for latent_d, latent_l in test_combinations:
        try:
            result = test_latent_combination(latent_d, latent_l, max_epochs=100, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"Error testing D={latent_d}, L={latent_l}: {e}")
            continue
    
    # Create comparison visualization
    create_comparison_plots(results)
    
    # Print summary table
    print_summary_table(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for different latent combinations"""
    print(f"\n=== CREATING COMPARISON PLOTS ===")
    
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for plotting
    combinations = [f"D={r['latent_d']}, L={r['latent_l']}" for r in results]
    x_corrs = [r['correlations']['X'] for r in results]
    y_corrs = [r['correlations']['Y'] for r in results]
    z_corrs = [r['correlations']['Z'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    training_times = [r['training_time'] for r in results]
    compressions = [r['total_compression'] for r in results]
    
    # 1. Correlation comparison
    x_pos = np.arange(len(combinations))
    width = 0.25
    
    axes[0, 0].bar(x_pos - width, x_corrs, width, label='X (input)', alpha=0.7, color='blue')
    axes[0, 0].bar(x_pos, y_corrs, width, label='Y (reconstructed)', alpha=0.7, color='green')
    axes[0, 0].bar(x_pos + width, z_corrs, width, label='Z (reconstructed)', alpha=0.7, color='red')
    
    axes[0, 0].set_xlabel('Latent Combination')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Correlation vs Latent Combination')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean error vs combination
    axes[0, 1].bar(x_pos, mean_errors, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Latent Combination')
    axes[0, 1].set_ylabel('Mean Reconstruction Error')
    axes[0, 1].set_title('Reconstruction Error vs Latent Combination')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training time vs combination
    axes[0, 2].bar(x_pos, training_times, alpha=0.7, color='purple')
    axes[0, 2].set_xlabel('Latent Combination')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time vs Latent Combination')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Compression ratio vs combination
    axes[1, 0].bar(x_pos, compressions, alpha=0.7, color='brown')
    axes[1, 0].set_xlabel('Latent Combination')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Compression Ratio vs Latent Combination')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(combinations, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Error vs Compression scatter
    axes[1, 1].scatter(compressions, mean_errors, s=100, alpha=0.7, c='red')
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('Mean Reconstruction Error')
    axes[1, 1].set_title('Error vs Compression Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add labels to scatter points
    for i, combo in enumerate(combinations):
        axes[1, 1].annotate(combo, (compressions[i], mean_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 6. Correlation vs Compression scatter
    axes[1, 2].scatter(compressions, y_corrs, s=100, alpha=0.7, c='green', label='Y correlation')
    axes[1, 2].scatter(compressions, z_corrs, s=100, alpha=0.7, c='red', label='Z correlation')
    axes[1, 2].set_xlabel('Compression Ratio')
    axes[1, 2].set_ylabel('Correlation')
    axes[1, 2].set_title('Reconstructed Correlation vs Compression')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_combinations_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved as 'latent_combinations_comparison.png'")

def print_summary_table(results):
    """Print a summary table of all results"""
    print(f"\n=== SUMMARY TABLE ===")
    print(f"{'Combination':<15} {'X Corr':<8} {'Y Corr':<8} {'Z Corr':<8} {'Mean Err':<10} {'Compression':<12} {'Time(s)':<8}")
    print("-" * 80)
    
    for r in results:
        combo = f"D={r['latent_d']}, L={r['latent_l']}"
        print(f"{combo:<15} {r['correlations']['X']:<8.4f} {r['correlations']['Y']:<8.4f} "
              f"{r['correlations']['Z']:<8.4f} {r['mean_error']:<10.4f} "
              f"{r['total_compression']:<12.1f} {r['training_time']:<8.1f}")
    
    # Find best combinations
    print(f"\n=== BEST COMBINATIONS ===")
    
    # Best overall correlation (Y + Z average)
    best_corr_idx = max(range(len(results)), 
                       key=lambda i: (results[i]['correlations']['Y'] + results[i]['correlations']['Z']) / 2)
    best_corr = results[best_corr_idx]
    print(f"Best Correlation: D={best_corr['latent_d']}, L={best_corr['latent_l']} "
          f"(Y={best_corr['correlations']['Y']:.4f}, Z={best_corr['correlations']['Z']:.4f})")
    
    # Best compression
    best_comp_idx = max(range(len(results)), key=lambda i: results[i]['total_compression'])
    best_comp = results[best_comp_idx]
    print(f"Best Compression: D={best_comp['latent_d']}, L={best_comp['latent_l']} "
          f"({best_comp['total_compression']:.1f}:1)")
    
    # Best error
    best_error_idx = min(range(len(results)), key=lambda i: results[i]['mean_error'])
    best_error = results[best_error_idx]
    print(f"Best Error: D={best_error['latent_d']}, L={best_error['latent_l']} "
          f"({best_error['mean_error']:.4f})")
    
    # Fastest training
    fastest_idx = min(range(len(results)), key=lambda i: results[i]['training_time'])
    fastest = results[fastest_idx]
    print(f"Fastest Training: D={fastest['latent_d']}, L={fastest['latent_l']} "
          f"({fastest['training_time']:.1f}s)")

if __name__ == "__main__":
    results = main()
