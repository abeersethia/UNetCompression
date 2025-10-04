"""
Lorenz Attractor Reconstruction Visualization
Shows 3D reconstructions from each architecture
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def test_architecture_quick(ArchClass, name, traj, t, max_epochs=50, pbar=None):
    """Quick test of architecture for visualization"""
    if pbar:
        pbar.set_description(f"Testing {name}")
    
    if name == "EDGeNet":
        reconstructor = ArchClass(
            window_len=512, delay_embedding_dim=10, stride=5,
            compressed_t=256, train_split=0.7
        )
        reconstructor.prepare_data(traj, t)
        training_history = reconstructor.train(max_epochs=max_epochs, patience=1000, verbose=False)
    elif name == "Corrected":
        reconstructor = ArchClass(
            window_len=512, delay_embedding_dim=10, stride=5,
            latent_d=32, latent_l=128, train_split=0.7
        )
        reconstructor.prepare_data(traj, t)
        training_history = reconstructor.train(max_epochs=max_epochs, verbose=False)
    else:
        reconstructor = ArchClass(
            window_len=512, delay_embedding_dim=10, stride=5,
            compressed_t=256, train_split=0.7
        )
        reconstructor.prepare_data(traj, t)
        training_history = reconstructor.train(max_epochs=max_epochs, verbose=False)
    
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    if pbar:
        pbar.update(1)
    
    return {
        'name': name,
        'original': original_attractor,
        'reconstructed': reconstructed_attractor,
        'metrics': metrics,
        'training_history': training_history
    }

def create_lorenz_reconstruction_visualization():
    """Create comprehensive Lorenz attractor reconstruction visualization"""
    print("="*80)
    print("LORENZ ATTRACTOR RECONSTRUCTION VISUALIZATION")
    print("="*80)
    print("X-Only Manifold Reconstruction from Each Architecture")
    print()
    
    # Generate Lorenz attractor
    print("🔄 Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    print(f"✅ Trajectory shape: {traj.shape}")
    
    # Test all architectures with progress bar
    architectures = [
        (DirectManifoldMLPReconstructor, "MLP"),
        (DirectManifoldLSTMReconstructor, "LSTM"),
        (DirectManifoldCausalAEReconstructor, "CausalAE"),
        (DirectManifoldEDGeNetReconstructor, "EDGeNet"),
        (XOnlyManifoldReconstructorCorrected, "Corrected")
    ]
    
    print("🔄 Testing all architectures...")
    results = []
    with tqdm(total=len(architectures), desc="Architecture Testing", unit="arch") as pbar:
        for ArchClass, name in architectures:
            try:
                result = test_architecture_quick(ArchClass, name, traj, t, max_epochs=50, pbar=pbar)
                results.append(result)
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
                pbar.update(1)
                continue
    
    print("🔄 Creating Lorenz attractor reconstruction visualization...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Original Lorenz Attractor (3D) - Top Left
    ax1 = fig.add_subplot(3, 6, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.8, linewidth=1, color='blue', label='Original')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=100, label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Lorenz Attractor\n(3D)', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2-6. Individual Architecture Reconstructions (3D)
    for idx, result in enumerate(results):
        name = result['name']
        original = result['original']
        reconstructed = result['reconstructed']
        metrics = result['metrics']
        
        # 3D Reconstruction
        ax = fig.add_subplot(3, 6, idx + 2, projection='3d')
        ax.plot(original[:, 0], original[:, 1], original[:, 2], 
                alpha=0.6, linewidth=0.8, color='blue', label='Original')
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
                alpha=0.8, linewidth=0.8, color='red', label='Reconstructed')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add performance metrics to title
        avg_corr = np.mean([metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']])
        ax.set_title(f'{name}\nAvg Corr: {avg_corr:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    # 7-11. 2D X-Y Projections
    for idx, result in enumerate(results):
        name = result['name']
        original = result['original']
        reconstructed = result['reconstructed']
        
        ax = fig.add_subplot(3, 6, idx + 7)
        ax.plot(original[:, 0], original[:, 1], alpha=0.8, linewidth=1, color='blue', label='Original')
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{name} - X-Y Projection', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 12-16. 2D Y-Z Projections
    for idx, result in enumerate(results):
        name = result['name']
        original = result['original']
        reconstructed = result['reconstructed']
        
        ax = fig.add_subplot(3, 6, idx + 12)
        ax.plot(original[:, 1], original[:, 2], alpha=0.8, linewidth=1, color='blue', label='Original')
        ax.plot(reconstructed[:, 1], reconstructed[:, 2], alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f'{name} - Y-Z Projection', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 17. Overlay Comparison (3D)
    ax17 = fig.add_subplot(3, 6, 17, projection='3d')
    # Find best performing architecture
    best_result = max(results, key=lambda r: np.mean([r['metrics']['correlations']['X'], 
                                                     r['metrics']['correlations']['Y'], 
                                                     r['metrics']['correlations']['Z']]))
    
    ax17.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6, linewidth=1, color='blue', label='Original')
    ax17.plot(best_result['reconstructed'][:, 0], best_result['reconstructed'][:, 1], best_result['reconstructed'][:, 2], 
             alpha=0.8, linewidth=1, color='red', label=f'Best ({best_result["name"]})')
    ax17.set_xlabel('X')
    ax17.set_ylabel('Y')
    ax17.set_zlabel('Z')
    ax17.set_title('Best Reconstruction Overlay', fontsize=12, fontweight='bold')
    ax17.legend()
    
    # 18. Performance Summary
    ax18 = fig.add_subplot(3, 6, 18)
    ax18.axis('off')
    
    # Create performance summary text
    performance_text = "PERFORMANCE SUMMARY:\n\n"
    for result in results:
        name = result['name']
        metrics = result['metrics']
        avg_corr = np.mean([metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']])
        performance_text += f"{name}:\n"
        performance_text += f"  X: {metrics['correlations']['X']:.3f}\n"
        performance_text += f"  Y: {metrics['correlations']['Y']:.3f}\n"
        performance_text += f"  Z: {metrics['correlations']['Z']:.3f}\n"
        performance_text += f"  Avg: {avg_corr:.3f}\n"
        performance_text += f"  Error: {metrics['mean_error']:.2f}\n\n"
    
    ax18.text(0.05, 0.95, performance_text, transform=ax18.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Lorenz Attractor Reconstruction Comparison\nX-Only Manifold Reconstruction', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('plots/lorenz_attractor_reconstructions.png', dpi=300, bbox_inches='tight')
    print("✅ Lorenz attractor reconstructions saved to: plots/lorenz_attractor_reconstructions.png")
    
    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED RECONSTRUCTION RESULTS")
    print("="*80)
    
    for result in results:
        name = result['name']
        metrics = result['metrics']
        avg_corr = np.mean([metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']])
        
        print(f"\n🏗️  {name}:")
        print(f"   X Correlation: {metrics['correlations']['X']:.4f}")
        print(f"   Y Correlation: {metrics['correlations']['Y']:.4f}")
        print(f"   Z Correlation: {metrics['correlations']['Z']:.4f}")
        print(f"   Average Correlation: {avg_corr:.4f}")
        print(f"   Mean Error: {metrics['mean_error']:.4f}")
        params = result['training_history'].get('total_parameters', 0)
        print(f"   Parameters: {params:,}" if params > 0 else "   Parameters: N/A")
    
    # Find and highlight best performers
    best_overall = max(results, key=lambda r: np.mean([r['metrics']['correlations']['X'], 
                                                      r['metrics']['correlations']['Y'], 
                                                      r['metrics']['correlations']['Z']]))
    best_x = max(results, key=lambda r: r['metrics']['correlations']['X'])
    best_y = max(results, key=lambda r: r['metrics']['correlations']['Y'])
    best_z = max(results, key=lambda r: r['metrics']['correlations']['Z'])
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"   Overall: {best_overall['name']} (avg: {np.mean([best_overall['metrics']['correlations']['X'], best_overall['metrics']['correlations']['Y'], best_overall['metrics']['correlations']['Z']]):.4f})")
    print(f"   X Component: {best_x['name']} ({best_x['metrics']['correlations']['X']:.4f})")
    print(f"   Y Component: {best_y['name']} ({best_y['metrics']['correlations']['Y']:.4f})")
    print(f"   Z Component: {best_z['name']} ({best_z['metrics']['correlations']['Z']:.4f})")
    
    plt.show()
    return results

def main():
    """Main function"""
    results = create_lorenz_reconstruction_visualization()
    return results

if __name__ == "__main__":
    results = main()
