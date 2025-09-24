"""
Test Latent Combinations D = 8 to 16 for X-Only Manifold Reconstruction

This script tests various combinations of latent dimensions (B, D, L) where:
- B: Batch size (fixed at 289)
- D: Feature dimensions (8 to 16)
- L: Compressed signal length (64, 128, 256)

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from src.core.lorenz import generate_lorenz_full
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time

def test_latent_combination(latent_d, latent_l, max_epochs=80, verbose=False):
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
    """Test latent combinations D = 8 to 16"""
    print("=== TESTING LATENT COMBINATIONS D = 8 TO 16 ===")
    print("Testing various (D, L) combinations for X-only manifold reconstruction")
    print("Focus: D = 8, 10, 12, 14, 16 with L = 64, 128, 256")
    print()
    
    # Define test combinations focused on D = 8 to 16
    test_combinations = [
        # (D, L) combinations
        (8, 64),    # High compression
        (8, 128),   # Medium compression
        (8, 256),   # Low compression
        (10, 64),   # High compression
        (10, 128),  # Medium compression
        (10, 256),  # Low compression
        (12, 64),   # High compression
        (12, 128),  # Medium compression
        (12, 256),  # Low compression
        (14, 64),   # High compression
        (14, 128),  # Medium compression
        (14, 256),  # Low compression
        (16, 64),   # High compression
        (16, 128),  # Medium compression
        (16, 256),  # Low compression
    ]
    
    results = []
    
    for latent_d, latent_l in test_combinations:
        try:
            result = test_latent_combination(latent_d, latent_l, max_epochs=80, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"Error testing D={latent_d}, L={latent_l}: {e}")
            continue
    
    # Create comparison visualization
    create_comparison_plots(results)
    
    # Print summary table
    print_summary_table(results)
    
    # Analyze trends
    analyze_trends(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for different latent combinations"""
    print(f"\n=== CREATING COMPARISON PLOTS ===")
    
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Extract data for plotting
    combinations = [f"D={r['latent_d']}, L={r['latent_l']}" for r in results]
    x_corrs = [r['correlations']['X'] for r in results]
    y_corrs = [r['correlations']['Y'] for r in results]
    z_corrs = [r['correlations']['Z'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    training_times = [r['training_time'] for r in results]
    compressions = [r['total_compression'] for r in results]
    d_values = [r['latent_d'] for r in results]
    l_values = [r['latent_l'] for r in results]
    
    # 1. Correlation vs D (grouped by L)
    for l_val in [64, 128, 256]:
        l_mask = [l == l_val for l in l_values]
        d_vals = [d for d, mask in zip(d_values, l_mask) if mask]
        y_corr_vals = [y for y, mask in zip(y_corrs, l_mask) if mask]
        z_corr_vals = [z for z, mask in zip(z_corrs, l_mask) if mask]
        
        axes[0, 0].plot(d_vals, y_corr_vals, 'o-', label=f'Y (L={l_val})', alpha=0.8, linewidth=2)
        axes[0, 0].plot(d_vals, z_corr_vals, 's-', label=f'Z (L={l_val})', alpha=0.8, linewidth=2)
    
    axes[0, 0].set_xlabel('D (Feature Dimensions)')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Reconstructed Correlation vs D\n(Grouped by L)', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(7, 17)
    
    # 2. Error vs D (grouped by L)
    for l_val in [64, 128, 256]:
        l_mask = [l == l_val for l in l_values]
        d_vals = [d for d, mask in zip(d_values, l_mask) if mask]
        error_vals = [e for e, mask in zip(mean_errors, l_mask) if mask]
        
        axes[0, 1].plot(d_vals, error_vals, 'o-', label=f'L={l_val}', alpha=0.8, linewidth=2)
    
    axes[0, 1].set_xlabel('D (Feature Dimensions)')
    axes[0, 1].set_ylabel('Mean Reconstruction Error')
    axes[0, 1].set_title('Reconstruction Error vs D\n(Grouped by L)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(7, 17)
    
    # 3. Training Time vs D (grouped by L)
    for l_val in [64, 128, 256]:
        l_mask = [l == l_val for l in l_values]
        d_vals = [d for d, mask in zip(d_values, l_mask) if mask]
        time_vals = [t for t, mask in zip(training_times, l_mask) if mask]
        
        axes[0, 2].plot(d_vals, time_vals, 'o-', label=f'L={l_val}', alpha=0.8, linewidth=2)
    
    axes[0, 2].set_xlabel('D (Feature Dimensions)')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time vs D\n(Grouped by L)', fontsize=12)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(7, 17)
    
    # 4. Compression vs D (grouped by L)
    for l_val in [64, 128, 256]:
        l_mask = [l == l_val for l in l_values]
        d_vals = [d for d, mask in zip(d_values, l_mask) if mask]
        comp_vals = [c for c, mask in zip(compressions, l_mask) if mask]
        
        axes[1, 0].plot(d_vals, comp_vals, 'o-', label=f'L={l_val}', alpha=0.8, linewidth=2)
    
    axes[1, 0].set_xlabel('D (Feature Dimensions)')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Compression Ratio vs D\n(Grouped by L)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(7, 17)
    
    # 5. Error vs Compression scatter
    colors = ['red' if l == 64 else 'green' if l == 128 else 'blue' for l in l_values]
    scatter = axes[1, 1].scatter(compressions, mean_errors, c=colors, s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('Mean Reconstruction Error')
    axes[1, 1].set_title('Error vs Compression Trade-off\n(Color: L)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='L=64'),
                      Patch(facecolor='green', label='L=128'),
                      Patch(facecolor='blue', label='L=256')]
    axes[1, 1].legend(handles=legend_elements)
    
    # 6. Correlation vs Compression scatter
    axes[1, 2].scatter(compressions, y_corrs, c=colors, s=100, alpha=0.7, marker='o', label='Y correlation')
    axes[1, 2].scatter(compressions, z_corrs, c=colors, s=100, alpha=0.7, marker='s', label='Z correlation')
    axes[1, 2].set_xlabel('Compression Ratio')
    axes[1, 2].set_ylabel('Correlation')
    axes[1, 2].set_title('Reconstructed Correlation vs Compression\n(Color: L)', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d8_to_d16_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved as 'd8_to_d16_comparison.png'")

def print_summary_table(results):
    """Print a summary table of all results"""
    print(f"\n=== SUMMARY TABLE (D = 8 TO 16) ===")
    print(f"{'Combination':<15} {'X Corr':<8} {'Y Corr':<8} {'Z Corr':<8} {'Mean Err':<10} {'Compression':<12} {'Time(s)':<8}")
    print("-" * 80)
    
    for r in results:
        combo = f"D={r['latent_d']}, L={r['latent_l']}"
        print(f"{combo:<15} {r['correlations']['X']:<8.4f} {r['correlations']['Y']:<8.4f} "
              f"{r['correlations']['Z']:<8.4f} {r['mean_error']:<10.4f} "
              f"{r['total_compression']:<12.1f} {r['training_time']:<8.1f}")
    
    # Find best combinations
    print(f"\n=== BEST COMBINATIONS (D = 8 TO 16) ===")
    
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

def analyze_trends(results):
    """Analyze trends in the results"""
    print(f"\n=== TREND ANALYSIS (D = 8 TO 16) ===")
    
    # Group by L values
    l64_results = [r for r in results if r['latent_l'] == 64]
    l128_results = [r for r in results if r['latent_l'] == 128]
    l256_results = [r for r in results if r['latent_l'] == 256]
    
    def analyze_group(group, l_val):
        if not group:
            return
        group.sort(key=lambda x: x['latent_d'])
        d_vals = [r['latent_d'] for r in group]
        y_corrs = [r['correlations']['Y'] for r in group]
        z_corrs = [r['correlations']['Z'] for r in group]
        errors = [r['mean_error'] for r in group]
        
        print(f"\nL = {l_val}:")
        print(f"  D range: {min(d_vals)} to {max(d_vals)}")
        print(f"  Y correlation range: {min(y_corrs):.4f} to {max(y_corrs):.4f}")
        print(f"  Z correlation range: {min(z_corrs):.4f} to {max(z_corrs):.4f}")
        print(f"  Error range: {min(errors):.4f} to {max(errors):.4f}")
        
        # Calculate correlation between D and performance
        d_y_corr = np.corrcoef(d_vals, y_corrs)[0, 1]
        d_z_corr = np.corrcoef(d_vals, z_corrs)[0, 1]
        d_error_corr = np.corrcoef(d_vals, errors)[0, 1]
        
        print(f"  D vs Y correlation: {d_y_corr:.4f}")
        print(f"  D vs Z correlation: {d_z_corr:.4f}")
        print(f"  D vs Error correlation: {d_error_corr:.4f}")
    
    analyze_group(l64_results, 64)
    analyze_group(l128_results, 128)
    analyze_group(l256_results, 256)
    
    # Overall trends
    print(f"\nOVERALL TRENDS:")
    all_d_vals = [r['latent_d'] for r in results]
    all_y_corrs = [r['correlations']['Y'] for r in results]
    all_z_corrs = [r['correlations']['Z'] for r in results]
    all_errors = [r['mean_error'] for r in results]
    
    d_y_corr = np.corrcoef(all_d_vals, all_y_corrs)[0, 1]
    d_z_corr = np.corrcoef(all_d_vals, all_z_corrs)[0, 1]
    d_error_corr = np.corrcoef(all_d_vals, all_errors)[0, 1]
    
    print(f"  D vs Y correlation (all): {d_y_corr:.4f}")
    print(f"  D vs Z correlation (all): {d_z_corr:.4f}")
    print(f"  D vs Error correlation (all): {d_error_corr:.4f}")
    
    if d_y_corr > 0.1:
        print(f"  → Increasing D improves Y reconstruction")
    elif d_y_corr < -0.1:
        print(f"  → Increasing D degrades Y reconstruction")
    else:
        print(f"  → D has minimal effect on Y reconstruction")
    
    if d_z_corr > 0.1:
        print(f"  → Increasing D improves Z reconstruction")
    elif d_z_corr < -0.1:
        print(f"  → Increasing D degrades Z reconstruction")
    else:
        print(f"  → D has minimal effect on Z reconstruction")

if __name__ == "__main__":
    results = main()
