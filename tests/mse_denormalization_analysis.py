"""
MSE Denormalization Analysis
Check MSE values with and without denormalization to understand the true reconstruction quality
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def denormalize_signal(signal, mean, std):
    """Denormalize a signal using mean and std"""
    return signal * std + mean

def calculate_denormalized_mse(original, reconstructed, dataset):
    """Calculate MSE after denormalizing the reconstructed signal"""
    # Get normalization parameters from dataset
    mean = dataset.mean
    std = dataset.std
    
    # Denormalize reconstructed signal
    reconstructed_denorm = denormalize_signal(reconstructed, mean, std)
    
    # Calculate MSE on denormalized values
    mse_x = mean_squared_error(original[:, 0], reconstructed_denorm[:, 0])
    mse_y = mean_squared_error(original[:, 1], reconstructed_denorm[:, 1])
    mse_z = mean_squared_error(original[:, 2], reconstructed_denorm[:, 2])
    
    return {
        'X': mse_x,
        'Y': mse_y,
        'Z': mse_z,
        'mean': (mse_x + mse_y + mse_z) / 3
    }

def test_architecture_with_denormalization(ArchClass, name, traj, t, max_epochs=50):
    """Test architecture and calculate both normalized and denormalized MSE"""
    print(f"Testing {name}...")
    
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
    
    # Calculate denormalized MSE
    denorm_mse = calculate_denormalized_mse(original_attractor, reconstructed_attractor, reconstructor.dataset_x)
    
    return {
        'name': name,
        'original': original_attractor,
        'reconstructed': reconstructed_attractor,
        'metrics': metrics,
        'training_history': training_history,
        'denormalized_mse': denorm_mse,
        'reconstructor': reconstructor
    }

def main():
    """Main analysis function"""
    print("="*80)
    print("MSE DENORMALIZATION ANALYSIS")
    print("="*80)
    print("Comparing normalized vs denormalized MSE values")
    print()
    
    # Generate Lorenz attractor
    print("🔄 Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    print(f"✅ Trajectory shape: {traj.shape}")
    
    # Check original signal statistics
    print("\n📊 Original Signal Statistics:")
    print(f"X range: [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}], std: {traj[:, 0].std():.2f}")
    print(f"Y range: [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}], std: {traj[:, 1].std():.2f}")
    print(f"Z range: [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}], std: {traj[:, 2].std():.2f}")
    
    # Test all architectures
    architectures = [
        (DirectManifoldMLPReconstructor, "MLP"),
        (DirectManifoldLSTMReconstructor, "LSTM"),
        (DirectManifoldCausalAEReconstructor, "CausalAE"),
        (DirectManifoldEDGeNetReconstructor, "EDGeNet"),
        (XOnlyManifoldReconstructorCorrected, "Corrected")
    ]
    
    print("\n🔄 Testing all architectures...")
    results = []
    with tqdm(total=len(architectures), desc="Architecture Testing", unit="arch") as pbar:
        for ArchClass, name in architectures:
            try:
                result = test_architecture_with_denormalization(ArchClass, name, traj, t, max_epochs=50)
                results.append(result)
                pbar.update(1)
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
                pbar.update(1)
                continue
    
    # Create comparison table
    print("\n" + "="*100)
    print("MSE COMPARISON: NORMALIZED vs DENORMALIZED")
    print("="*100)
    
    print(f"{'Architecture':<12} {'Norm_X':<10} {'Norm_Y':<10} {'Norm_Z':<10} {'Norm_Mean':<10} {'Denorm_X':<10} {'Denorm_Y':<10} {'Denorm_Z':<10} {'Denorm_Mean':<10}")
    print("-" * 100)
    
    for result in results:
        name = result['name']
        norm_mse = result['metrics'].get('mse', {'X': 0, 'Y': 0, 'Z': 0})
        denorm_mse = result['denormalized_mse']
        
        print(f"{name:<12} {norm_mse.get('X', 0):<10.4f} {norm_mse.get('Y', 0):<10.4f} {norm_mse.get('Z', 0):<10.4f} "
              f"{(norm_mse.get('X', 0) + norm_mse.get('Y', 0) + norm_mse.get('Z', 0))/3:<10.4f} "
              f"{denorm_mse['X']:<10.4f} {denorm_mse['Y']:<10.4f} {denorm_mse['Z']:<10.4f} {denorm_mse['mean']:<10.4f}")
    
    # Create visualization
    print("\n🔄 Creating MSE comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Normalized MSE Comparison
    ax1 = axes[0, 0]
    arch_names = [r['name'] for r in results]
    norm_x = [r['metrics'].get('mse', {}).get('X', 0) for r in results]
    norm_y = [r['metrics'].get('mse', {}).get('Y', 0) for r in results]
    norm_z = [r['metrics'].get('mse', {}).get('Z', 0) for r in results]
    
    x_pos = np.arange(len(arch_names))
    width = 0.25
    
    ax1.bar(x_pos - width, norm_x, width, label='X', color='blue', alpha=0.7)
    ax1.bar(x_pos, norm_y, width, label='Y', color='green', alpha=0.7)
    ax1.bar(x_pos + width, norm_z, width, label='Z', color='red', alpha=0.7)
    
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Normalized MSE')
    ax1.set_title('Normalized MSE Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(arch_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Denormalized MSE Comparison
    ax2 = axes[0, 1]
    denorm_x = [r['denormalized_mse']['X'] for r in results]
    denorm_y = [r['denormalized_mse']['Y'] for r in results]
    denorm_z = [r['denormalized_mse']['Z'] for r in results]
    
    ax2.bar(x_pos - width, denorm_x, width, label='X', color='blue', alpha=0.7)
    ax2.bar(x_pos, denorm_y, width, label='Y', color='green', alpha=0.7)
    ax2.bar(x_pos + width, denorm_z, width, label='Z', color='red', alpha=0.7)
    
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Denormalized MSE')
    ax2.set_title('Denormalized MSE Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(arch_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MSE Ratio (Denorm/Norm)
    ax3 = axes[0, 2]
    ratio_x = [denorm_x[i] / max(norm_x[i], 1e-8) for i in range(len(results))]
    ratio_y = [denorm_y[i] / max(norm_y[i], 1e-8) for i in range(len(results))]
    ratio_z = [denorm_z[i] / max(norm_z[i], 1e-8) for i in range(len(results))]
    
    ax3.bar(x_pos - width, ratio_x, width, label='X', color='blue', alpha=0.7)
    ax3.bar(x_pos, ratio_y, width, label='Y', color='green', alpha=0.7)
    ax3.bar(x_pos + width, ratio_z, width, label='Z', color='red', alpha=0.7)
    
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('MSE Ratio (Denorm/Norm)')
    ax3.set_title('MSE Amplification Factor')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(arch_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Best Architecture Analysis
    best_result = min(results, key=lambda r: r['denormalized_mse']['mean'])
    ax4 = axes[1, 0]
    
    components = ['X', 'Y', 'Z']
    norm_mse_best = [best_result['metrics'].get('mse', {}).get(c, 0) for c in components]
    denorm_mse_best = [best_result['denormalized_mse'][c] for c in components]
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    ax4.bar(x_pos - width/2, norm_mse_best, width, label='Normalized', color='lightblue', alpha=0.7)
    ax4.bar(x_pos + width/2, denorm_mse_best, width, label='Denormalized', color='darkblue', alpha=0.7)
    
    ax4.set_xlabel('Component')
    ax4.set_ylabel('MSE')
    ax4.set_title(f'Best Architecture: {best_result["name"]}')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(components)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Signal Range vs MSE
    ax5 = axes[1, 1]
    signal_ranges = [traj[:, 0].std(), traj[:, 1].std(), traj[:, 2].std()]
    avg_denorm_mse = [np.mean([r['denormalized_mse']['X'] for r in results]),
                     np.mean([r['denormalized_mse']['Y'] for r in results]),
                     np.mean([r['denormalized_mse']['Z'] for r in results])]
    
    ax5.scatter(signal_ranges, avg_denorm_mse, c=['blue', 'green', 'red'], s=100, alpha=0.7)
    for i, comp in enumerate(components):
        ax5.annotate(comp, (signal_ranges[i], avg_denorm_mse[i]), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    
    ax5.set_xlabel('Signal Standard Deviation')
    ax5.set_ylabel('Average Denormalized MSE')
    ax5.set_title('Signal Range vs Reconstruction Difficulty')
    ax5.grid(True, alpha=0.3)
    
    # 6. Architecture Performance Ranking
    ax6 = axes[1, 2]
    arch_performance = [(r['name'], r['denormalized_mse']['mean']) for r in results]
    arch_performance.sort(key=lambda x: x[1])
    
    names, performances = zip(*arch_performance)
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax6.bar(names, performances, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Architecture')
    ax6.set_ylabel('Average Denormalized MSE')
    ax6.set_title('Architecture Performance Ranking')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add performance values on bars
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('MSE Denormalization Analysis\nNormalized vs Denormalized Reconstruction Quality', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/mse_denormalization_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ MSE analysis saved to: plots/mse_denormalization_analysis.png")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    best_arch = min(results, key=lambda r: r['denormalized_mse']['mean'])
    print(f"🏆 Best Architecture (Denormalized MSE): {best_arch['name']}")
    print(f"   Average Denormalized MSE: {best_arch['denormalized_mse']['mean']:.4f}")
    print(f"   X MSE: {best_arch['denormalized_mse']['X']:.4f}")
    print(f"   Y MSE: {best_arch['denormalized_mse']['Y']:.4f}")
    print(f"   Z MSE: {best_arch['denormalized_mse']['Z']:.4f}")
    
    print(f"\n📊 Key Insights:")
    print(f"   • Z component has highest reconstruction difficulty (largest range)")
    print(f"   • Denormalized MSE shows true reconstruction quality")
    print(f"   • Normalized MSE can be misleading due to signal scaling")
    print(f"   • Signal range correlates with reconstruction difficulty")
    
    plt.show()
    return results

if __name__ == "__main__":
    results = main()
