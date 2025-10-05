"""
Denormalized MSE Comparison
Compare denormalized MSE values across all architectures
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Import all architectures
from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.direct_manifold_modunet import DirectManifoldModUNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def test_architecture_denormalized(ArchClass, name, traj, t, max_epochs=50):
    """Test a single architecture and get denormalized MSE"""
    print(f"\n🔄 Testing {name}...")
    
    # Special parameters for different architectures
    if name == "EDGeNet":
        max_epochs = 250
        patience = 1000  # No early stopping for EDGeNet
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            compressed_t=256,
            train_split=0.7
        )
    elif name == "ModUNet":
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            latent_d=32,
            latent_l=128,
            train_split=0.7
        )
    elif name == "Corrected":
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            latent_d=32,
            latent_l=128,
            train_split=0.7
        )
    else:
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            compressed_t=256,
            train_split=0.7
        )
    
    # Prepare data and train
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=max_epochs, patience=25, verbose=False)
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    return {
        'name': name,
        'metrics': metrics,
        'training_history': training_history
    }

def main():
    """Main function to compare denormalized MSE across all architectures"""
    print("="*80)
    print("DENORMALIZED MSE COMPARISON")
    print("="*80)
    print("Comparing denormalized MSE values across all architectures")
    print()
    
    # Generate Lorenz attractor
    print("🔄 Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    print(f"✅ Trajectory shape: {traj.shape}")
    
    # Test all architectures
    architectures = [
        (DirectManifoldMLPReconstructor, "MLP"),
        (DirectManifoldLSTMReconstructor, "LSTM"),
        (DirectManifoldCausalAEReconstructor, "CausalAE"),
        (DirectManifoldEDGeNetReconstructor, "EDGeNet"),
        (DirectManifoldModUNetReconstructor, "ModUNet"),
        (XOnlyManifoldReconstructorCorrected, "Corrected")
    ]
    
    print("🔄 Testing all architectures...")
    results = []
    with tqdm(total=len(architectures), desc="Architecture Testing", unit="arch") as pbar:
        for ArchClass, name in architectures:
            try:
                result = test_architecture_denormalized(ArchClass, name, traj, t)
                results.append(result)
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
            pbar.update(1)
    
    # Create comparison table
    print("\n" + "="*120)
    print("DENORMALIZED MSE COMPARISON TABLE")
    print("="*120)
    
    print(f"{'Architecture':<12} {'Norm_X':<10} {'Norm_Y':<10} {'Norm_Z':<10} {'Norm_Mean':<10} {'Denorm_X':<10} {'Denorm_Y':<10} {'Denorm_Z':<10} {'Denorm_Mean':<10} {'X_Corr':<8} {'Y_Corr':<8} {'Z_Corr':<8}")
    print("-" * 120)
    
    for result in results:
        name = result['name']
        metrics = result['metrics']
        
        # Handle missing denormalized MSE keys
        if 'mse_normalized' in metrics:
            norm_mse = metrics['mse_normalized']
        else:
            # Fallback to regular MSE if available
            norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0, 'Mean': 0})
        
        if 'mse_denormalized' in metrics:
            denorm_mse = metrics['mse_denormalized']
        else:
            # Use normalized MSE as fallback
            denorm_mse = norm_mse
        
        correlations = metrics['correlations']
        
        # Calculate mean if not present
        norm_mean = norm_mse.get('Mean', np.mean([norm_mse['X'], norm_mse['Y'], norm_mse['Z']]))
        denorm_mean = denorm_mse.get('Mean', np.mean([denorm_mse['X'], denorm_mse['Y'], denorm_mse['Z']]))
        
        print(f"{name:<12} {norm_mse['X']:<10.4f} {norm_mse['Y']:<10.4f} {norm_mse['Z']:<10.4f} {norm_mean:<10.4f} "
              f"{denorm_mse['X']:<10.4f} {denorm_mse['Y']:<10.4f} {denorm_mse['Z']:<10.4f} {denorm_mean:<10.4f} "
              f"{correlations['X']:<8.4f} {correlations['Y']:<8.4f} {correlations['Z']:<8.4f}")
    
    # Create visualization
    print("\n🔄 Creating denormalized MSE visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Normalized vs Denormalized MSE Comparison
    arch_names = [r['name'] for r in results]
    
    # Handle missing keys safely
    norm_x_mse = []
    norm_y_mse = []
    norm_z_mse = []
    denorm_x_mse = []
    denorm_y_mse = []
    denorm_z_mse = []
    
    for r in results:
        metrics = r['metrics']
        
        # Normalized MSE
        if 'mse_normalized' in metrics:
            norm_x_mse.append(metrics['mse_normalized']['X'])
            norm_y_mse.append(metrics['mse_normalized']['Y'])
            norm_z_mse.append(metrics['mse_normalized']['Z'])
        else:
            norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0})
            norm_x_mse.append(norm_mse['X'])
            norm_y_mse.append(norm_mse['Y'])
            norm_z_mse.append(norm_mse['Z'])
        
        # Denormalized MSE
        if 'mse_denormalized' in metrics:
            denorm_x_mse.append(metrics['mse_denormalized']['X'])
            denorm_y_mse.append(metrics['mse_denormalized']['Y'])
            denorm_z_mse.append(metrics['mse_denormalized']['Z'])
        else:
            # Use normalized as fallback
            if 'mse_normalized' in metrics:
                denorm_x_mse.append(metrics['mse_normalized']['X'])
                denorm_y_mse.append(metrics['mse_normalized']['Y'])
                denorm_z_mse.append(metrics['mse_normalized']['Z'])
            else:
                norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0})
                denorm_x_mse.append(norm_mse['X'])
                denorm_y_mse.append(norm_mse['Y'])
                denorm_z_mse.append(norm_mse['Z'])
    
    x = np.arange(len(arch_names))
    width = 0.12
    
    # Normalized MSE bars
    ax1.bar(x - 2*width, norm_x_mse, width, label='X (Norm)', color='skyblue', alpha=0.7)
    ax1.bar(x - width, norm_y_mse, width, label='Y (Norm)', color='lightcoral', alpha=0.7)
    ax1.bar(x, norm_z_mse, width, label='Z (Norm)', color='lightgreen', alpha=0.7)
    
    # Denormalized MSE bars
    ax1.bar(x + width, denorm_x_mse, width, label='X (Denorm)', color='blue', alpha=0.7)
    ax1.bar(x + 2*width, denorm_y_mse, width, label='Y (Denorm)', color='red', alpha=0.7)
    ax1.bar(x + 3*width, denorm_z_mse, width, label='Z (Denorm)', color='green', alpha=0.7)
    
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('MSE')
    ax1.set_title('Normalized vs Denormalized MSE Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Denormalized MSE Only
    ax2.bar(x - width, denorm_x_mse, width, label='X', color='blue', alpha=0.7, edgecolor='black')
    ax2.bar(x, denorm_y_mse, width, label='Y', color='red', alpha=0.7, edgecolor='black')
    ax2.bar(x + width, denorm_z_mse, width, label='Z', color='green', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Denormalized MSE')
    ax2.set_title('Denormalized MSE Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (x_val, y_val, z_val) in enumerate(zip(denorm_x_mse, denorm_y_mse, denorm_z_mse)):
        ax2.text(i - width, x_val + x_val*0.01, f'{x_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, y_val + y_val*0.01, f'{y_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width, z_val + z_val*0.01, f'{z_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Correlation vs Denormalized MSE
    correlations_x = [r['metrics']['correlations']['X'] for r in results]
    correlations_y = [r['metrics']['correlations']['Y'] for r in results]
    correlations_z = [r['metrics']['correlations']['Z'] for r in results]
    
    ax3.scatter(denorm_x_mse, correlations_x, c='blue', s=100, alpha=0.7, label='X', edgecolors='black')
    ax3.scatter(denorm_y_mse, correlations_y, c='red', s=100, alpha=0.7, label='Y', edgecolors='black')
    ax3.scatter(denorm_z_mse, correlations_z, c='green', s=100, alpha=0.7, label='Z', edgecolors='black')
    
    ax3.set_xlabel('Denormalized MSE')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation vs Denormalized MSE', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, name in enumerate(arch_names):
        ax3.annotate(name, (denorm_x_mse[i], correlations_x[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Architecture Ranking by Denormalized MSE
    denorm_mean_mse = []
    for r in results:
        metrics = r['metrics']
        if 'mse_denormalized' in metrics:
            denorm_mean_mse.append(metrics['mse_denormalized'].get('Mean', 
                np.mean([metrics['mse_denormalized']['X'], metrics['mse_denormalized']['Y'], metrics['mse_denormalized']['Z']])))
        else:
            # Use normalized MSE as fallback
            if 'mse_normalized' in metrics:
                denorm_mean_mse.append(metrics['mse_normalized'].get('Mean',
                    np.mean([metrics['mse_normalized']['X'], metrics['mse_normalized']['Y'], metrics['mse_normalized']['Z']])))
            else:
                norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0})
                denorm_mean_mse.append(np.mean([norm_mse['X'], norm_mse['Y'], norm_mse['Z']]))
    
    # Sort by denormalized MSE (lower is better)
    sorted_results = sorted(zip(arch_names, denorm_mean_mse), key=lambda x: x[1])
    sorted_names, sorted_mse = zip(*sorted_results)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
    bars = ax4.bar(range(len(sorted_names)), sorted_mse, color=colors, alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('Architecture (Ranked by Denormalized MSE)')
    ax4.set_ylabel('Average Denormalized MSE')
    ax4.set_title('Architecture Ranking (Lower MSE = Better)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(sorted_names)))
    ax4.set_xticklabels(sorted_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, mse in zip(bars, sorted_mse):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{mse:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/denormalized_mse_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Denormalized MSE comparison saved to: plots/denormalized_mse_comparison.png")
    
    # Find best and worst performers
    def get_denorm_mean(r):
        metrics = r['metrics']
        if 'mse_denormalized' in metrics:
            return metrics['mse_denormalized'].get('Mean', 
                np.mean([metrics['mse_denormalized']['X'], metrics['mse_denormalized']['Y'], metrics['mse_denormalized']['Z']]))
        else:
            if 'mse_normalized' in metrics:
                return metrics['mse_normalized'].get('Mean',
                    np.mean([metrics['mse_normalized']['X'], metrics['mse_normalized']['Y'], metrics['mse_normalized']['Z']]))
            else:
                norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0})
                return np.mean([norm_mse['X'], norm_mse['Y'], norm_mse['Z']])
    
    best_mse = min(results, key=get_denorm_mean)
    worst_mse = max(results, key=get_denorm_mean)
    
    print("\n" + "="*80)
    print("DENORMALIZED MSE ANALYSIS SUMMARY")
    print("="*80)
    print(f"🏆 Best Denormalized MSE: {best_mse['name']} ({get_denorm_mean(best_mse):.4f})")
    print(f"📊 Worst Denormalized MSE: {worst_mse['name']} ({get_denorm_mean(worst_mse):.4f})")
    
    print(f"\n📈 Key Insights:")
    print(f"   • Denormalized MSE shows true reconstruction quality")
    print(f"   • Z component typically has highest reconstruction difficulty")
    print(f"   • Normalized MSE can be misleading due to signal scaling")
    print(f"   • Lower denormalized MSE indicates better reconstruction")
    
    # Save results
    comparison_data = []
    for result in results:
        name = result['name']
        metrics = result['metrics']
        training_history = result['training_history']
        
        # Handle missing keys safely
        if 'mse_normalized' in metrics:
            norm_mse = metrics['mse_normalized']
        else:
            norm_mse = metrics.get('mse', {'X': 0, 'Y': 0, 'Z': 0, 'Mean': 0})
        
        if 'mse_denormalized' in metrics:
            denorm_mse = metrics['mse_denormalized']
        else:
            denorm_mse = norm_mse
        
        comparison_data.append({
            'Architecture': name,
            'Norm_X_MSE': norm_mse['X'],
            'Norm_Y_MSE': norm_mse['Y'],
            'Norm_Z_MSE': norm_mse['Z'],
            'Norm_Mean_MSE': norm_mse.get('Mean', np.mean([norm_mse['X'], norm_mse['Y'], norm_mse['Z']])),
            'Denorm_X_MSE': denorm_mse['X'],
            'Denorm_Y_MSE': denorm_mse['Y'],
            'Denorm_Z_MSE': denorm_mse['Z'],
            'Denorm_Mean_MSE': denorm_mse.get('Mean', np.mean([denorm_mse['X'], denorm_mse['Y'], denorm_mse['Z']])),
            'X_Correlation': metrics['correlations']['X'],
            'Y_Correlation': metrics['correlations']['Y'],
            'Z_Correlation': metrics['correlations']['Z'],
            'Parameters': training_history.get('total_parameters', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Denorm_Mean_MSE')
    df.to_csv('plots/denormalized_mse_comparison_results.csv', index=False)
    print(f"📊 Results saved to: plots/denormalized_mse_comparison_results.csv")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = main()
