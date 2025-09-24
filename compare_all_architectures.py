"""
Compare All X-Only Manifold Reconstruction Architectures

This script runs and compares all 4 autoencoder architectures:
1. MLP Autoencoder
2. LSTM Autoencoder  
3. CausalAE (Causal CNN)
4. EDGeNet

Provides comprehensive comparison including:
- Parameter counts
- FLOPs estimation
- Reconstruction quality
- Training time
- Manifold visualization

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd

# Import all architectures
from x_only_manifold_reconstruction_mlp import XOnlyManifoldReconstructorMLP
from x_only_manifold_reconstruction_lstm import XOnlyManifoldReconstructorLSTM
from x_only_manifold_reconstruction_causalae import XOnlyManifoldReconstructorCausalAE
from x_only_manifold_reconstruction_edgenet import XOnlyManifoldReconstructorEDGeNet
from lorenz import generate_lorenz_full

def run_architecture_comparison(max_epochs=60, verbose=False):
    """Run comparison of all 4 architectures"""
    print("=== COMPARING ALL X-ONLY MANIFOLD RECONSTRUCTION ARCHITECTURES ===")
    print("Architectures: MLP, LSTM, CausalAE, EDGeNet")
    print(f"Training epochs: {max_epochs}")
    print()
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Define common parameters
    common_params = {
        'window_len': 512,
        'delay_embedding_dim': 10,
        'stride': 5,
        'latent_d': 32,
        'latent_l': 128,
        'train_split': 0.7
    }
    
    # Initialize all reconstructors
    architectures = {
        'MLP': XOnlyManifoldReconstructorMLP(**common_params),
        'LSTM': XOnlyManifoldReconstructorLSTM(**common_params),
        'CausalAE': XOnlyManifoldReconstructorCausalAE(**common_params),
        'EDGeNet': XOnlyManifoldReconstructorEDGeNet(**common_params)
    }
    
    results = {}
    
    for arch_name, reconstructor in architectures.items():
        print(f"\n{'='*50}")
        print(f"TESTING {arch_name.upper()} ARCHITECTURE")
        print(f"{'='*50}")
        
        try:
            # Prepare data
            start_time = time.time()
            reconstructor.prepare_data(traj, t)
            
            # Train model
            training_start = time.time()
            training_history = reconstructor.train(max_epochs=max_epochs, verbose=verbose)
            training_time = time.time() - training_start
            
            # Reconstruct manifold
            reconstruction_start = time.time()
            original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
            reconstruction_time = time.time() - reconstruction_start
            
            total_time = time.time() - start_time
            
            # Store results
            results[arch_name] = {
                'reconstructor': reconstructor,
                'original_attractor': original_attractor,
                'reconstructed_attractor': reconstructed_attractor,
                'metrics': metrics,
                'training_history': training_history,
                'training_time': training_time,
                'reconstruction_time': reconstruction_time,
                'total_time': total_time,
                'parameters': training_history['total_parameters'],
                'flops': training_history['estimated_flops']
            }
            
            print(f"✅ {arch_name} completed successfully")
            print(f"   Parameters: {training_history['total_parameters']:,}")
            print(f"   FLOPs: {training_history['estimated_flops']:,}")
            print(f"   Training time: {training_time:.2f}s")
            print(f"   X correlation: {metrics['correlations']['X']:.4f}")
            print(f"   Y correlation: {metrics['correlations']['Y']:.4f}")
            print(f"   Z correlation: {metrics['correlations']['Z']:.4f}")
            print(f"   Mean error: {metrics['mean_error']:.4f}")
            
        except Exception as e:
            print(f"❌ {arch_name} failed: {e}")
            results[arch_name] = None
    
    return results

def create_comprehensive_comparison(results):
    """Create comprehensive comparison visualization"""
    print(f"\n=== CREATING COMPREHENSIVE COMPARISON ===")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if len(successful_results) == 0:
        print("No successful results to compare")
        return
    
    # Create comparison figure
    fig = plt.figure(figsize=(24, 20))
    
    # 1. 3D Manifold Comparison (2x2 grid)
    arch_names = list(successful_results.keys())
    for i, arch_name in enumerate(arch_names):
        ax = fig.add_subplot(4, 4, i+1, projection='3d')
        result = successful_results[arch_name]
        original = result['original_attractor']
        reconstructed = result['reconstructed_attractor']
        
        ax.plot(original[:, 0], original[:, 1], original[:, 2], 
                alpha=0.6, linewidth=1, color='blue', label='Original')
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
                alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{arch_name} Manifold Reconstruction', fontsize=12)
        if i == 0:
            ax.legend()
    
    # 2. Correlation Comparison
    ax5 = fig.add_subplot(4, 4, 5)
    arch_names = list(successful_results.keys())
    x_corrs = [successful_results[arch]['metrics']['correlations']['X'] for arch in arch_names]
    y_corrs = [successful_results[arch]['metrics']['correlations']['Y'] for arch in arch_names]
    z_corrs = [successful_results[arch]['metrics']['correlations']['Z'] for arch in arch_names]
    
    x_pos = np.arange(len(arch_names))
    width = 0.25
    
    ax5.bar(x_pos - width, x_corrs, width, label='X (input)', alpha=0.8, color='blue')
    ax5.bar(x_pos, y_corrs, width, label='Y (reconstructed)', alpha=0.8, color='green')
    ax5.bar(x_pos + width, z_corrs, width, label='Z (reconstructed)', alpha=0.8, color='red')
    
    ax5.set_xlabel('Architecture')
    ax5.set_ylabel('Correlation')
    ax5.set_title('Correlation Comparison', fontsize=12)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(arch_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # 3. Error Comparison
    ax6 = fig.add_subplot(4, 4, 6)
    errors = [successful_results[arch]['metrics']['mean_error'] for arch in arch_names]
    colors = ['skyblue', 'lightgreen', 'lightyellow', 'lightcyan'][:len(arch_names)]
    
    bars = ax6.bar(arch_names, errors, color=colors, alpha=0.8)
    ax6.set_xlabel('Architecture')
    ax6.set_ylabel('Mean Reconstruction Error')
    ax6.set_title('Reconstruction Error Comparison', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Parameter Count Comparison
    ax7 = fig.add_subplot(4, 4, 7)
    params = [successful_results[arch]['parameters'] for arch in arch_names]
    
    bars = ax7.bar(arch_names, params, color=colors, alpha=0.8)
    ax7.set_xlabel('Architecture')
    ax7.set_ylabel('Parameters')
    ax7.set_title('Parameter Count Comparison', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{param/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    # 5. FLOPs Comparison
    ax8 = fig.add_subplot(4, 4, 8)
    flops = [successful_results[arch]['flops'] for arch in arch_names]
    
    bars = ax8.bar(arch_names, flops, color=colors, alpha=0.8)
    ax8.set_xlabel('Architecture')
    ax8.set_ylabel('FLOPs')
    ax8.set_title('FLOPs Comparison', fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    for bar, flop in zip(bars, flops):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{flop/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 6. Training Time Comparison
    ax9 = fig.add_subplot(4, 4, 9)
    times = [successful_results[arch]['training_time'] for arch in arch_names]
    
    bars = ax9.bar(arch_names, times, color=colors, alpha=0.8)
    ax9.set_xlabel('Architecture')
    ax9.set_ylabel('Training Time (seconds)')
    ax9.set_title('Training Time Comparison', fontsize=12)
    ax9.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 7. Efficiency Analysis (Error vs Parameters)
    ax10 = fig.add_subplot(4, 4, 10)
    ax10.scatter(params, errors, s=200, c=colors[:len(arch_names)], alpha=0.8)
    ax10.set_xlabel('Parameters')
    ax10.set_ylabel('Mean Reconstruction Error')
    ax10.set_title('Efficiency Analysis\n(Error vs Parameters)', fontsize=12)
    ax10.grid(True, alpha=0.3)
    
    for i, arch in enumerate(arch_names):
        ax10.annotate(arch, (params[i], errors[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # 8. Performance vs Complexity (Correlation vs FLOPs)
    ax11 = fig.add_subplot(4, 4, 11)
    avg_corrs = [(y_corrs[i] + z_corrs[i]) / 2 for i in range(len(arch_names))]
    ax11.scatter(flops, avg_corrs, s=200, c=colors[:len(arch_names)], alpha=0.8)
    ax11.set_xlabel('FLOPs')
    ax11.set_ylabel('Avg Reconstructed Correlation (Y+Z)/2')
    ax11.set_title('Performance vs Complexity\n(Correlation vs FLOPs)', fontsize=12)
    ax11.grid(True, alpha=0.3)
    
    for i, arch in enumerate(arch_names):
        ax11.annotate(arch, (flops[i], avg_corrs[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # 9-12. Individual time series comparisons
    for i, arch_name in enumerate(arch_names):
        if i >= 4:
            break
        ax = fig.add_subplot(4, 4, 12+i)
        result = successful_results[arch_name]
        original = result['original_attractor']
        reconstructed = result['reconstructed_attractor']
        t = np.linspace(0, 20.0, len(original))
        
        # Plot Y component comparison
        ax.plot(t[:1000], original[:1000, 1], alpha=0.8, linewidth=1, color='blue', label='Original Y')
        ax.plot(t[:1000], reconstructed[:1000, 1], alpha=0.8, linewidth=1, color='red', label='Reconstructed Y')
        ax.set_xlabel('Time')
        ax.set_ylabel('Y Value')
        ax.set_title(f'{arch_name} Y Component\n(Correlation: {y_corrs[i]:.3f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('all_architectures_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive comparison saved as 'all_architectures_comparison.png'")

def print_summary_table(results):
    """Print a comprehensive summary table"""
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if len(successful_results) == 0:
        print("No successful results to summarize")
        return
    
    # Create summary data
    summary_data = []
    for arch_name, result in successful_results.items():
        metrics = result['metrics']
        summary_data.append({
            'Architecture': arch_name,
            'Parameters': f"{result['parameters']:,}",
            'FLOPs': f"{result['flops']:,}",
            'Train Time (s)': f"{result['training_time']:.1f}",
            'X Corr': f"{metrics['correlations']['X']:.4f}",
            'Y Corr': f"{metrics['correlations']['Y']:.4f}",
            'Z Corr': f"{metrics['correlations']['Z']:.4f}",
            'Mean Error': f"{metrics['mean_error']:.4f}",
            'Best Val Loss': f"{result['training_history']['best_val_loss']:.6f}"
        })
    
    # Create DataFrame for nice printing
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Find best performers
    print(f"\n{'='*80}")
    print("BEST PERFORMERS")
    print(f"{'='*80}")
    
    # Best correlation
    y_corrs = [successful_results[arch]['metrics']['correlations']['Y'] for arch in successful_results.keys()]
    z_corrs = [successful_results[arch]['metrics']['correlations']['Z'] for arch in successful_results.keys()]
    avg_corrs = [(y + z) / 2 for y, z in zip(y_corrs, z_corrs)]
    best_corr_idx = np.argmax(avg_corrs)
    best_corr_arch = list(successful_results.keys())[best_corr_idx]
    
    print(f"🏆 Best Correlation: {best_corr_arch} (Avg Y+Z: {avg_corrs[best_corr_idx]:.4f})")
    
    # Best error
    errors = [successful_results[arch]['metrics']['mean_error'] for arch in successful_results.keys()]
    best_error_idx = np.argmin(errors)
    best_error_arch = list(successful_results.keys())[best_error_idx]
    
    print(f"🎯 Best Error: {best_error_arch} (Error: {errors[best_error_idx]:.4f})")
    
    # Most efficient (best correlation per parameter)
    params = [successful_results[arch]['parameters'] for arch in successful_results.keys()]
    efficiency = [corr / param * 1e6 for corr, param in zip(avg_corrs, params)]  # Correlation per million parameters
    best_eff_idx = np.argmax(efficiency)
    best_eff_arch = list(successful_results.keys())[best_eff_idx]
    
    print(f"⚡ Most Efficient: {best_eff_arch} (Corr/M Params: {efficiency[best_eff_idx]:.4f})")
    
    # Fastest
    times = [successful_results[arch]['training_time'] for arch in successful_results.keys()]
    fastest_idx = np.argmin(times)
    fastest_arch = list(successful_results.keys())[fastest_idx]
    
    print(f"🚀 Fastest Training: {fastest_arch} (Time: {times[fastest_idx]:.1f}s)")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"• For best quality: Use {best_corr_arch}")
    print(f"• For best efficiency: Use {best_eff_arch}")
    print(f"• For fastest training: Use {fastest_arch}")
    print(f"• For balanced performance: Consider trade-offs between quality, speed, and complexity")

def main():
    """Main function for architecture comparison"""
    print("🚀 Starting comprehensive architecture comparison...")
    
    # Run comparison
    results = run_architecture_comparison(max_epochs=60, verbose=False)
    
    # Create visualizations
    create_comprehensive_comparison(results)
    
    # Print summary
    print_summary_table(results)
    
    return results

if __name__ == "__main__":
    results = main()
