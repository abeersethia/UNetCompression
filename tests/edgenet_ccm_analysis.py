"""
EDGeNet CCM Analysis
Compute Convergent Cross Mapping (CCM) from reconstructed manifold for EDGeNet
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import pandas as pd

# Import EDGeNet architecture
from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor

def embed_time_series(x, E, tau=1):
    """
    Create delay embedding of time series
    
    Args:
        x: Time series data
        E: Embedding dimension
        tau: Time delay
    
    Returns:
        Embedded time series
    """
    N = len(x)
    embedded = np.zeros((N - (E-1)*tau, E))
    
    for i in range(E):
        embedded[:, i] = x[i*tau:N - (E-1-i)*tau]
    
    return embedded

def find_neighbors(embedded, k=5):
    """
    Find k nearest neighbors for each point in embedded space
    
    Args:
        embedded: Embedded time series
        k: Number of neighbors
    
    Returns:
        Indices of nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embedded)
    distances, indices = nbrs.kneighbors(embedded)
    return indices[:, 1:]  # Exclude self

def ccm_prediction(x, y, E=3, tau=1, L=None):
    """
    Perform CCM prediction from x to y
    
    Args:
        x: Source time series
        y: Target time series
        E: Embedding dimension
        tau: Time delay
        L: Library size (if None, use all data)
    
    Returns:
        CCM correlation and prediction
    """
    # Create embeddings
    x_embedded = embed_time_series(x, E, tau)
    y_embedded = embed_time_series(y, E, tau)
    
    # Use same length for both
    min_len = min(len(x_embedded), len(y_embedded))
    x_embedded = x_embedded[:min_len]
    y_embedded = y_embedded[:min_len]
    
    # Set library size
    if L is None:
        L = len(x_embedded)
    else:
        L = min(L, len(x_embedded))
    
    # Find neighbors in x space
    x_library = x_embedded[:L]
    x_test = x_embedded[L:]
    y_test = y_embedded[L:]
    
    if len(x_test) == 0:
        return 0.0, np.array([])
    
    # Find neighbors for test points
    nbrs = NearestNeighbors(n_neighbors=min(5, L), algorithm='ball_tree').fit(x_library)
    distances, indices = nbrs.kneighbors(x_test)
    
    # Weighted prediction
    predictions = np.zeros(len(x_test))
    weights = np.exp(-distances / np.mean(distances))
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    for i in range(len(x_test)):
        neighbor_indices = indices[i]
        neighbor_weights = weights[i]
        
        # Predict y from neighbors in x space
        y_neighbors = y_embedded[neighbor_indices]
        predictions[i] = np.sum(neighbor_weights * y_neighbors[:, 0])  # Use first component
    
    # Calculate correlation
    if len(predictions) > 1:
        correlation, _ = pearsonr(predictions, y_test[:, 0])
    else:
        correlation = 0.0
    
    return correlation, predictions

def ccm_analysis(x, y, E=3, tau=1, L_range=None):
    """
    Perform CCM analysis with varying library sizes
    
    Args:
        x: Source time series
        y: Target time series
        E: Embedding dimension
        tau: Time delay
        L_range: Range of library sizes to test
    
    Returns:
        Dictionary with CCM results
    """
    if L_range is None:
        L_range = np.logspace(2, np.log10(len(x)//2), 10).astype(int)
    
    correlations = []
    library_sizes = []
    
    for L in L_range:
        if L < 10:  # Skip very small library sizes
            continue
            
        corr, _ = ccm_prediction(x, y, E, tau, L)
        correlations.append(corr)
        library_sizes.append(L)
    
    return {
        'correlations': np.array(correlations),
        'library_sizes': np.array(library_sizes),
        'max_correlation': np.max(correlations) if correlations else 0.0,
        'convergence': np.mean(correlations[-3:]) if len(correlations) >= 3 else 0.0
    }

def analyze_edgenet_ccm():
    """Analyze CCM for EDGeNet reconstructed manifold"""
    print("="*80)
    print("EDGENET CCM ANALYSIS")
    print("="*80)
    print("Computing Convergent Cross Mapping from reconstructed manifold")
    print()
    
    # Generate Lorenz attractor
    print("🔄 Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    print(f"✅ Trajectory shape: {traj.shape}")
    
    # Create EDGeNet reconstructor
    print("🔄 Creating EDGeNet reconstructor...")
    reconstructor = DirectManifoldEDGeNetReconstructor(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        compressed_t=256,
        train_split=0.7
    )
    
    # Prepare data and train
    print("🔄 Preparing data and training EDGeNet...")
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=250, patience=1000, verbose=False)
    
    # Reconstruct manifold
    print("🔄 Reconstructing manifold...")
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Extract time series
    x_orig = original_attractor[:, 0]
    y_orig = original_attractor[:, 1]
    z_orig = original_attractor[:, 2]
    
    x_recon = reconstructed_attractor[:, 0]
    y_recon = reconstructed_attractor[:, 1]
    z_recon = reconstructed_attractor[:, 2]
    
    print(f"✅ Original time series length: {len(x_orig)}")
    print(f"✅ Reconstructed time series length: {len(x_recon)}")
    
    # Perform CCM analysis
    print("\n🔄 Performing CCM analysis...")
    
    # CCM from original data
    print("  Analyzing original Lorenz attractor...")
    orig_xy = ccm_analysis(x_orig, y_orig, E=3, tau=1)
    orig_xz = ccm_analysis(x_orig, z_orig, E=3, tau=1)
    orig_yx = ccm_analysis(y_orig, x_orig, E=3, tau=1)
    orig_yz = ccm_analysis(y_orig, z_orig, E=3, tau=1)
    orig_zx = ccm_analysis(z_orig, x_orig, E=3, tau=1)
    orig_zy = ccm_analysis(z_orig, y_orig, E=3, tau=1)
    
    # CCM from reconstructed data
    print("  Analyzing EDGeNet reconstructed manifold...")
    recon_xy = ccm_analysis(x_recon, y_recon, E=3, tau=1)
    recon_xz = ccm_analysis(x_recon, z_recon, E=3, tau=1)
    recon_yx = ccm_analysis(y_recon, x_recon, E=3, tau=1)
    recon_yz = ccm_analysis(y_recon, z_recon, E=3, tau=1)
    recon_zx = ccm_analysis(z_recon, x_recon, E=3, tau=1)
    recon_zy = ccm_analysis(z_recon, y_recon, E=3, tau=1)
    
    # Create CCM comparison visualization
    print("🔄 Creating CCM visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. CCM Convergence plots
    ax1.plot(orig_xy['library_sizes'], orig_xy['correlations'], 'b-', label='X→Y (Original)', linewidth=2)
    ax1.plot(orig_xz['library_sizes'], orig_xz['correlations'], 'g-', label='X→Z (Original)', linewidth=2)
    ax1.plot(recon_xy['library_sizes'], recon_xy['correlations'], 'b--', label='X→Y (EDGeNet)', linewidth=2)
    ax1.plot(recon_xz['library_sizes'], recon_xz['correlations'], 'g--', label='X→Z (EDGeNet)', linewidth=2)
    ax1.set_xlabel('Library Size')
    ax1.set_ylabel('CCM Correlation')
    ax1.set_title('CCM Convergence: X as Driver', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. CCM Matrix comparison
    orig_ccm_matrix = np.array([
        [1.0, orig_xy['max_correlation'], orig_xz['max_correlation']],
        [orig_yx['max_correlation'], 1.0, orig_yz['max_correlation']],
        [orig_zx['max_correlation'], orig_zy['max_correlation'], 1.0]
    ])
    
    recon_ccm_matrix = np.array([
        [1.0, recon_xy['max_correlation'], recon_xz['max_correlation']],
        [recon_yx['max_correlation'], 1.0, recon_yz['max_correlation']],
        [recon_zx['max_correlation'], recon_zy['max_correlation'], 1.0]
    ])
    
    im1 = ax2.imshow(orig_ccm_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('CCM Matrix (Original)', fontsize=14, fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['X', 'Y', 'Z'])
    ax2.set_yticklabels(['X', 'Y', 'Z'])
    plt.colorbar(im1, ax=ax2, label='CCM Correlation')
    
    # Add correlation values to matrix
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{orig_ccm_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold', color='white' if abs(orig_ccm_matrix[i, j]) > 0.5 else 'black')
    
    im2 = ax3.imshow(recon_ccm_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title('CCM Matrix (EDGeNet Reconstructed)', fontsize=14, fontweight='bold')
    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(['X', 'Y', 'Z'])
    ax3.set_yticklabels(['X', 'Y', 'Z'])
    plt.colorbar(im2, ax=ax3, label='CCM Correlation')
    
    # Add correlation values to matrix
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{recon_ccm_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold', color='white' if abs(recon_ccm_matrix[i, j]) > 0.5 else 'black')
    
    # 4. CCM Comparison bar chart
    directions = ['X→Y', 'X→Z', 'Y→X', 'Y→Z', 'Z→X', 'Z→Y']
    orig_ccm_values = [orig_xy['max_correlation'], orig_xz['max_correlation'], 
                      orig_yx['max_correlation'], orig_yz['max_correlation'],
                      orig_zx['max_correlation'], orig_zy['max_correlation']]
    recon_ccm_values = [recon_xy['max_correlation'], recon_xz['max_correlation'],
                       recon_yx['max_correlation'], recon_yz['max_correlation'],
                       recon_zx['max_correlation'], recon_zy['max_correlation']]
    
    x_pos = np.arange(len(directions))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, orig_ccm_values, width, label='Original', color='skyblue', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, recon_ccm_values, width, label='EDGeNet', color='lightcoral', alpha=0.7)
    
    ax4.set_xlabel('Causal Direction')
    ax4.set_ylabel('CCM Correlation')
    ax4.set_title('CCM Comparison: Original vs EDGeNet', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(directions, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars1, orig_ccm_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, recon_ccm_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/edgenet_ccm_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ CCM analysis saved to: plots/edgenet_ccm_analysis.png")
    
    # Print results
    print("\n" + "="*80)
    print("CCM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📊 Original Lorenz Attractor CCM:")
    print(f"   X→Y: {orig_xy['max_correlation']:.4f}")
    print(f"   X→Z: {orig_xz['max_correlation']:.4f}")
    print(f"   Y→X: {orig_yx['max_correlation']:.4f}")
    print(f"   Y→Z: {orig_yz['max_correlation']:.4f}")
    print(f"   Z→X: {orig_zx['max_correlation']:.4f}")
    print(f"   Z→Y: {orig_zy['max_correlation']:.4f}")
    
    print(f"\n🤖 EDGeNet Reconstructed CCM:")
    print(f"   X→Y: {recon_xy['max_correlation']:.4f}")
    print(f"   X→Z: {recon_xz['max_correlation']:.4f}")
    print(f"   Y→X: {recon_yx['max_correlation']:.4f}")
    print(f"   Y→Z: {recon_yz['max_correlation']:.4f}")
    print(f"   Z→X: {recon_zx['max_correlation']:.4f}")
    print(f"   Z→Y: {recon_zy['max_correlation']:.4f}")
    
    # Calculate CCM preservation
    ccm_preservation = []
    for i, direction in enumerate(directions):
        orig_val = orig_ccm_values[i]
        recon_val = recon_ccm_values[i]
        if orig_val != 0:
            preservation = recon_val / orig_val
        else:
            preservation = 0
        ccm_preservation.append(preservation)
    
    print(f"\n🎯 CCM Preservation (EDGeNet/Original):")
    for i, direction in enumerate(directions):
        print(f"   {direction}: {ccm_preservation[i]:.4f}")
    
    avg_preservation = np.mean(ccm_preservation)
    print(f"\n✅ Average CCM Preservation: {avg_preservation:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'Direction': directions,
        'Original_CCM': orig_ccm_values,
        'EDGeNet_CCM': recon_ccm_values,
        'Preservation': ccm_preservation
    })
    results_df.to_csv('plots/edgenet_ccm_results.csv', index=False)
    print(f"📊 CCM results saved to: plots/edgenet_ccm_results.csv")
    
    plt.show()
    return results_df

if __name__ == "__main__":
    results_df = analyze_edgenet_ccm()
