"""
Quick Test and Visualization for D = 8 to 16 Latent Combinations

This script runs a focused test on key combinations and creates specific visualizations:
1. Correlation vs Latent Combination (X input, Y reconstructed, Z reconstructed)
2. Reconstruction Error vs Latent Combination
3. Training Time vs Latent Combination
4. Compression Ratio vs Latent Combination
5. Error vs Compression Trade-off
6. Reconstruction Correlation vs Compression

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time

# Import the corrected reconstructor
from x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def quick_test(latent_d, latent_l, max_epochs=60):
    """Quick test with reduced epochs for faster execution"""
    print(f"Testing D={latent_d}, L={latent_l}...")
    
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
    reconstructor.prepare_data(traj, t)
    
    # Train the model
    training_start = time.time()
    training_history = reconstructor.train(max_epochs=max_epochs, verbose=False)
    training_time = time.time() - training_start
    
    # Reconstruct the manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Calculate compression ratios
    temporal_compression = 512 / latent_l
    spatial_compression = 10 / latent_d
    total_compression = (512 * 10) / (latent_d * latent_l)
    
    results = {
        'latent_d': latent_d,
        'latent_l': latent_l,
        'correlations': metrics['correlations'],
        'mean_error': metrics['mean_error'],
        'training_time': training_time,
        'total_compression': total_compression,
        'temporal_compression': temporal_compression,
        'spatial_compression': spatial_compression
    }
    
    print(f"  X: {metrics['correlations']['X']:.4f}, Y: {metrics['correlations']['Y']:.4f}, Z: {metrics['correlations']['Z']:.4f}, Error: {metrics['mean_error']:.4f}")
    
    return results

def main():
    """Run quick tests and create visualizations"""
    print("=== QUICK TEST D = 8 TO 16 ===")
    print("Running focused tests with reduced epochs for faster execution")
    print()
    
    # Key combinations to test (reduced set for speed)
    test_combinations = [
        (8, 64), (8, 128), (8, 256),
        (10, 64), (10, 128), (10, 256),
        (12, 64), (12, 128), (12, 256),
        (14, 64), (14, 128), (14, 256),
        (16, 64), (16, 128), (16, 256),
    ]
    
    results = []
    
    for latent_d, latent_l in test_combinations:
        try:
            result = quick_test(latent_d, latent_l, max_epochs=60)
            results.append(result)
        except Exception as e:
            print(f"Error testing D={latent_d}, L={latent_l}: {e}")
            continue
    
    # Create the specific visualizations requested
    create_requested_plots(results)
    
    return results

def create_requested_plots(results):
    """Create the specific plots requested by the user"""
    print(f"\n=== CREATING REQUESTED VISUALIZATIONS ===")
    
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    combinations = [f"D={r['latent_d']}, L={r['latent_l']}" for r in results]
    x_corrs = [r['correlations']['X'] for r in results]
    y_corrs = [r['correlations']['Y'] for r in results]
    z_corrs = [r['correlations']['Z'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    training_times = [r['training_time'] for r in results]
    compressions = [r['total_compression'] for r in results]
    
    # Create figure with 6 subplots as requested
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 1. Correlation vs Latent Combination (X input, Y reconstructed, Z reconstructed)
    x_pos = np.arange(len(combinations))
    width = 0.25
    
    bars1 = axes[0, 0].bar(x_pos - width, x_corrs, width, label='X (input)', alpha=0.8, color='blue')
    bars2 = axes[0, 0].bar(x_pos, y_corrs, width, label='Y (reconstructed)', alpha=0.8, color='green')
    bars3 = axes[0, 0].bar(x_pos + width, z_corrs, width, label='Z (reconstructed)', alpha=0.8, color='red')
    
    axes[0, 0].set_xlabel('Latent Combination')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Correlation vs Latent Combination\n(X input, Y reconstructed, Z reconstructed)', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Add correlation values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 2. Reconstruction Error vs Latent Combination
    bars = axes[0, 1].bar(x_pos, mean_errors, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Latent Combination')
    axes[0, 1].set_ylabel('Reconstruction Error')
    axes[0, 1].set_title('Reconstruction Error vs Latent Combination', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add error values on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 3. Training Time vs Latent Combination
    bars = axes[0, 2].bar(x_pos, training_times, alpha=0.8, color='purple')
    axes[0, 2].set_xlabel('Latent Combination')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time vs Latent Combination', fontsize=12)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(combinations, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add time values on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.1f}s', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 4. Compression Ratio vs Latent Combination
    bars = axes[1, 0].bar(x_pos, compressions, alpha=0.8, color='brown')
    axes[1, 0].set_xlabel('Latent Combination')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Compression Ratio vs Latent Combination', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(combinations, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add compression values on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.1f}:1', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 5. Error vs Compression Trade-off
    scatter = axes[1, 1].scatter(compressions, mean_errors, s=100, alpha=0.8, c='red', edgecolors='black')
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('Reconstruction Error')
    axes[1, 1].set_title('Error vs Compression Trade-off', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add labels to scatter points
    for i, combo in enumerate(combinations):
        axes[1, 1].annotate(combo, (compressions[i], mean_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 6. Reconstruction Correlation vs Compression
    axes[1, 2].scatter(compressions, y_corrs, s=100, alpha=0.8, c='green', marker='o', label='Y correlation', edgecolors='black')
    axes[1, 2].scatter(compressions, z_corrs, s=100, alpha=0.8, c='red', marker='s', label='Z correlation', edgecolors='black')
    axes[1, 2].set_xlabel('Compression Ratio')
    axes[1, 2].set_ylabel('Correlation')
    axes[1, 2].set_title('Reconstruction Correlation vs Compression', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)
    
    # Add labels to scatter points
    for i, combo in enumerate(combinations):
        axes[1, 2].annotate(combo, (compressions[i], y_corrs[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('d8_to_d16_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis plots saved as 'd8_to_d16_analysis.png'")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Tested {len(results)} combinations:")
    for r in results:
        print(f"  D={r['latent_d']}, L={r['latent_l']}: X={r['correlations']['X']:.3f}, Y={r['correlations']['Y']:.3f}, Z={r['correlations']['Z']:.3f}, Error={r['mean_error']:.3f}")

if __name__ == "__main__":
    results = main()
