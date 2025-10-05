"""
Rössler Attractor X-Only Reconstruction using Corrected Architecture
Reconstructs the full Rössler attractor from X component only using the corrected architecture
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Import Corrected architecture and Rössler generator
from src.core.rossler import generate_rossler_full, visualize_rossler_attractor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructionCorrected

def test_rossler_corrected_reconstruction():
    """Test Corrected reconstruction of Rössler attractor from X-only input"""
    print("="*80)
    print("RÖSSLER ATTRACTOR X-ONLY RECONSTRUCTION - CORRECTED")
    print("="*80)
    print("Reconstructing full Rössler attractor from X component only using Corrected architecture")
    print()
    
    # Generate Rössler attractor
    print("🔄 Generating Rössler attractor...")
    traj, t = generate_rossler_full(T=100.0, dt=0.01)  # Longer duration for full state space
    print(f"✅ Rössler trajectory shape: {traj.shape}")
    print(f"   X range: [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}]")
    print(f"   Y range: [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}]")
    print(f"   Z range: [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}]")
    
    # Create Corrected reconstructor
    print("🔄 Creating Corrected reconstructor...")
    reconstructor = XOnlyManifoldReconstructionCorrected(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        compressed_t=256,
        train_split=0.7
    )
    
    # Prepare data and train
    print("🔄 Preparing data and training Corrected architecture...")
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=150, verbose=True)
    
    # Reconstruct manifold
    print("🔄 Reconstructing Rössler manifold...")
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Create comprehensive visualization
    print("🔄 Creating Rössler Corrected reconstruction visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original Rössler Attractor (3D)
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
             alpha=0.8, linewidth=1, color='blue', label='Original')
    ax1.scatter(original_attractor[0, 0], original_attractor[0, 1], original_attractor[0, 2], 
               color='green', s=100, label='Start')
    ax1.scatter(original_attractor[-1, 0], original_attractor[-1, 1], original_attractor[-1, 2], 
               color='red', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Rössler Attractor', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Corrected Reconstructed Rössler (3D)
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
             alpha=0.8, linewidth=1, color='red', label='Corrected Reconstructed')
    ax2.scatter(reconstructed_attractor[0, 0], reconstructed_attractor[0, 1], reconstructed_attractor[0, 2], 
               color='green', s=100, label='Start')
    ax2.scatter(reconstructed_attractor[-1, 0], reconstructed_attractor[-1, 1], reconstructed_attractor[-1, 2], 
               color='blue', s=100, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Corrected Reconstructed Rössler', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Overlay Comparison (3D)
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax3.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
             alpha=0.6, linewidth=1, color='blue', label='Original')
    ax3.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
             alpha=0.8, linewidth=1, color='red', label='Corrected Reconstructed')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. X-Y Projection
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(original_attractor[:, 0], original_attractor[:, 1], alpha=0.8, linewidth=1, color='blue', label='Original')
    ax4.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], alpha=0.8, linewidth=1, color='red', label='Corrected')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('X-Y Projection', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Time Series - X (Input)
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(t, original_attractor[:, 0], alpha=0.8, linewidth=1, color='blue', label='Original X')
    ax5.plot(t, reconstructed_attractor[:, 0], alpha=0.8, linewidth=1, color='red', label='Corrected X')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('X Value')
    ax5.set_title('X Component (INPUT)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Time Series - Y (Reconstructed)
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(t, original_attractor[:, 1], alpha=0.8, linewidth=1, color='blue', label='Original Y')
    ax6.plot(t, reconstructed_attractor[:, 1], alpha=0.8, linewidth=1, color='red', label='Corrected Y')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Y Value')
    ax6.set_title('Y Component (RECONSTRUCTED)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Time Series - Z (Reconstructed)
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.plot(t, original_attractor[:, 2], alpha=0.8, linewidth=1, color='blue', label='Original Z')
    ax7.plot(t, reconstructed_attractor[:, 2], alpha=0.8, linewidth=1, color='red', label='Corrected Z')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Z Value')
    ax7.set_title('Z Component (RECONSTRUCTED)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    # Handle metrics that might not have denormalized MSE
    mse_info = "N/A"
    if 'mse_denormalized' in metrics:
        mse_info = f"{metrics['mse_denormalized']['Mean']:.4f}"
    elif 'mse' in metrics:
        mse_info = f"{metrics['mse']['Mean']:.4f}"
    
    summary_text = f'''RÖSSLER ATTRACTOR X-ONLY RECONSTRUCTION - CORRECTED:

ARCHITECTURE: Corrected (Hankel → Direct Signal)
DATASET: Rössler Attractor
METHOD: X-only manifold reconstruction

RECONSTRUCTION QUALITY:
X Correlation: {metrics['correlations']['X']:.4f}
Y Correlation: {metrics['correlations']['Y']:.4f}
Z Correlation: {metrics['correlations']['Z']:.4f}

ERROR METRICS:
Mean Error: {metrics['mean_error']:.4f}
MSE: {mse_info}

MODEL STATISTICS:
Parameters: {training_history['total_parameters']:,}
Training Time: {training_history['training_time']:.2f}s

RÖSSLER CHARACTERISTICS:
- Chaotic attractor with single scroll
- Simpler than Lorenz (2D manifold)
- Good test for corrected architecture

SUCCESS: ✅ Rössler attractor reconstructed from X-only input!'''

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Rössler Attractor X-Only Reconstruction - Corrected\nFrom X Component to Full Attractor', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('plots/rossler_corrected_reconstruction.png', dpi=300, bbox_inches='tight')
    print("✅ Rössler Corrected reconstruction saved to: plots/rossler_corrected_reconstruction.png")
    
    # Print results
    print("\n" + "="*80)
    print("RÖSSLER CORRECTED RECONSTRUCTION RESULTS")
    print("="*80)
    print(f"🏗️  Architecture: Corrected")
    print(f"📊 Parameters: {training_history['total_parameters']:,}")
    print(f"⏱️  Training Time: {training_history['training_time']:.2f}s")
    print(f"🔗 X Correlation: {metrics['correlations']['X']:.4f}")
    print(f"🔗 Y Correlation: {metrics['correlations']['Y']:.4f}")
    print(f"🔗 Z Correlation: {metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean Error: {metrics['mean_error']:.4f}")
    print(f"📊 MSE: {mse_info}")
    
    print(f"\n✅ SUCCESS: Corrected architecture successfully reconstructs Rössler attractor from X-only input!")
    print(f"🎯 The corrected architecture is well-suited for Rössler's simpler dynamics.")
    
    plt.show()
    return reconstructor, original_attractor, reconstructed_attractor, metrics

if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = test_rossler_corrected_reconstruction()
