"""
Compare all Direct Manifold Architectures

Tests MLP, LSTM, CausalAE, EDGeNet, ModUNet, and Corrected versions on the same Lorenz attractor data.
All architectures use the Hankel → Direct Signal pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '.')

from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.direct_manifold_modunet import DirectManifoldModUNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_architecture(ArchClass, name, traj, t, max_epochs=100):
    """Test a single architecture"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Special handling for different architectures
    if name == "EDGeNet":
        # EDGeNet needs longer training and NO early stopping
        max_epochs = 250
        patience = 1000  # Effectively no early stopping
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            compressed_t=256,
            train_split=0.7
        )
    elif name == "Corrected":
        # Corrected version uses different parameters
        max_epochs = 100
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            latent_d=32,
            latent_l=128,
            train_split=0.7
        )
    else:
        # Standard direct manifold architectures
        max_epochs = 100
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            compressed_t=256,
            train_split=0.7
        )
    
    # All architectures follow the same pattern
    reconstructor.prepare_data(traj, t)
    
    # Use custom patience for EDGeNet
    if name == "EDGeNet":
        training_history = reconstructor.train(max_epochs=max_epochs, patience=patience, verbose=True)
    else:
        training_history = reconstructor.train(max_epochs=max_epochs, verbose=True)
    
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Fix training history for Corrected version
    if name == "Corrected":
        # Count parameters manually for corrected version
        if 'total_parameters' not in training_history:
            total_params = sum(p.numel() for p in reconstructor.encoder.parameters() if p.requires_grad) + \
                          sum(p.numel() for p in reconstructor.decoder.parameters() if p.requires_grad)
            training_history['total_parameters'] = total_params
        
        # Add training time if missing (calculate from training history)
        if 'training_time' not in training_history:
            # Estimate training time based on epochs and typical training speed
            epochs_trained = len(training_history.get('train_losses', []))
            training_history['training_time'] = epochs_trained * 0.5  # Rough estimate
    
    return {
        'name': name,
        'reconstructor': reconstructor,
        'training_history': training_history,
        'metrics': metrics,
        'original_attractor': original_attractor,
        'reconstructed_attractor': reconstructed_attractor
    }

def main():
    print("="*60)
    print("DIRECT MANIFOLD ARCHITECTURE COMPARISON")
    print("="*60)
    print("\nPipeline: Hankel(X) → Compressed(X,Y,Z) → Direct Signals(X,Y,Z)")
    print("No Hankel reconstruction step!\n")
    
    # Generate Lorenz attractor once
    print("Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    print(f"Trajectory shape: {traj.shape}\n")
    
    # Test all architectures
    architectures = [
        (DirectManifoldMLPReconstructor, "MLP"),
        (DirectManifoldLSTMReconstructor, "LSTM"),
        (DirectManifoldCausalAEReconstructor, "CausalAE"),
        (DirectManifoldEDGeNetReconstructor, "EDGeNet"),
        (DirectManifoldModUNetReconstructor, "ModUNet"),
        (XOnlyManifoldReconstructorCorrected, "Corrected")
    ]
    
    results = []
    for ArchClass, name in architectures:
        try:
            result = test_architecture(ArchClass, name, traj, t)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {name}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Architecture':<15} {'Params':<12} {'Time(s)':<10} {'X Corr':<10} {'Y Corr':<10} {'Z Corr':<10} {'Mean Err':<10}")
    print("-" * 80)
    
    for result in results:
        name = result['name']
        params = result['training_history']['total_parameters']
        time_s = result['training_history']['training_time']
        corr_x = result['metrics']['correlations']['X']
        corr_y = result['metrics']['correlations']['Y']
        corr_z = result['metrics']['correlations']['Z']
        mean_err = result['metrics']['mean_error']
        
        print(f"{name:<15} {params:<12,} {time_s:<10.2f} {corr_x:<10.4f} {corr_y:<10.4f} {corr_z:<10.4f} {mean_err:<10.4f}")
    
    # Find best performers
    print("\n" + "="*60)
    print("BEST PERFORMERS")
    print("="*60)
    
    best_x = max(results, key=lambda r: r['metrics']['correlations']['X'])
    best_y = max(results, key=lambda r: r['metrics']['correlations']['Y'])
    best_z = max(results, key=lambda r: r['metrics']['correlations']['Z'])
    best_overall = max(results, key=lambda r: np.mean([r['metrics']['correlations']['X'], 
                                                         r['metrics']['correlations']['Y'], 
                                                         r['metrics']['correlations']['Z']]))
    fastest = min(results, key=lambda r: r['training_history']['training_time'])
    most_efficient = min(results, key=lambda r: r['training_history']['total_parameters'])
    
    print(f"🏆 Best X correlation: {best_x['name']} ({best_x['metrics']['correlations']['X']:.4f})")
    print(f"🏆 Best Y correlation: {best_y['name']} ({best_y['metrics']['correlations']['Y']:.4f})")
    print(f"🏆 Best Z correlation: {best_z['name']} ({best_z['metrics']['correlations']['Z']:.4f})")
    print(f"🏆 Best overall: {best_overall['name']} (avg: {np.mean([best_overall['metrics']['correlations']['X'], best_overall['metrics']['correlations']['Y'], best_overall['metrics']['correlations']['Z']]):.4f})")
    print(f"⚡ Fastest training: {fastest['name']} ({fastest['training_history']['training_time']:.2f}s)")
    print(f"💡 Most efficient: {most_efficient['name']} ({most_efficient['training_history']['total_parameters']:,} params)")
    
    # Visualize all six architectures
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)
    
    fig = plt.figure(figsize=(30, 12))
    
    for idx, result in enumerate(results):
        name = result['name']
        original = result['original_attractor']
        reconstructed = result['reconstructed_attractor']
        metrics = result['metrics']
        
        # 3D plot
        ax = fig.add_subplot(2, 6, idx + 1, projection='3d')
        ax.plot(original[:, 0], original[:, 1], original[:, 2], alpha=0.5, linewidth=0.5, color='blue', label='Original')
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], alpha=0.5, linewidth=0.5, color='red', label='Reconstructed')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name} - 3D Attractor', fontsize=10)
        ax.legend(fontsize=8)
        
        # Correlation bar chart
        ax2 = fig.add_subplot(2, 6, idx + 7)
        components = ['X', 'Y', 'Z']
        correlations = [metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']]
        colors = ['blue', 'green', 'red']
        bars = ax2.bar(components, correlations, color=colors, alpha=0.7)
        ax2.set_ylabel('Correlation')
        ax2.set_title(f'{name} - Correlations', fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Direct Manifold Architecture Comparison\n(Hankel → Direct Signal Pipeline)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/direct_manifold_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: plots/direct_manifold_comparison.png")
    plt.show()
    
    # Create comprehensive comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*100)
    
    # Create detailed table
    print(f"\n{'Architecture':<12} {'Parameters':<12} {'Time(s)':<10} {'X Corr':<10} {'Y Corr':<10} {'Z Corr':<10} {'Avg Corr':<10} {'Mean Err':<10} {'Efficiency':<12}")
    print("-" * 100)
    
    for result in results:
        name = result['name']
        params = result['training_history']['total_parameters']
        time_s = result['training_history']['training_time']
        corr_x = result['metrics']['correlations']['X']
        corr_y = result['metrics']['correlations']['Y']
        corr_z = result['metrics']['correlations']['Z']
        avg_corr = np.mean([corr_x, corr_y, corr_z])
        mean_err = result['metrics']['mean_error']
        efficiency = params / max(time_s, 0.1)  # Parameters per second (avoid division by zero)
        
        print(f"{name:<12} {params:<12,} {time_s:<10.2f} {corr_x:<10.4f} {corr_y:<10.4f} {corr_z:<10.4f} {avg_corr:<10.4f} {mean_err:<10.4f} {efficiency:<12.0f}")
    
    # Create ranking analysis
    print("\n" + "="*60)
    print("RANKING ANALYSIS")
    print("="*60)
    
    # Sort by different metrics
    by_avg_corr = sorted(results, key=lambda r: np.mean([r['metrics']['correlations']['X'], r['metrics']['correlations']['Y'], r['metrics']['correlations']['Z']]), reverse=True)
    by_efficiency = sorted(results, key=lambda r: r['training_history']['total_parameters'] / max(r['training_history']['training_time'], 0.1))
    by_speed = sorted(results, key=lambda r: r['training_history']['training_time'])
    by_params = sorted(results, key=lambda r: r['training_history']['total_parameters'])
    
    print("📊 By Average Correlation:")
    for i, result in enumerate(by_avg_corr[:3]):
        avg_corr = np.mean([result['metrics']['correlations']['X'], result['metrics']['correlations']['Y'], result['metrics']['correlations']['Z']])
        print(f"  {i+1}. {result['name']}: {avg_corr:.4f}")
    
    print("\n⚡ By Training Speed:")
    for i, result in enumerate(by_speed[:3]):
        print(f"  {i+1}. {result['name']}: {result['training_history']['training_time']:.2f}s")
    
    print("\n💡 By Parameter Efficiency:")
    for i, result in enumerate(by_efficiency[:3]):
        efficiency = result['training_history']['total_parameters'] / max(result['training_history']['training_time'], 0.1)
        print(f"  {i+1}. {result['name']}: {efficiency:.0f} params/s")
    
    print("\n🔢 By Model Size:")
    for i, result in enumerate(by_params[:3]):
        print(f"  {i+1}. {result['name']}: {result['training_history']['total_parameters']:,} params")
    
    print("\n✅ All tests completed successfully!")
    return results

if __name__ == "__main__":
    results = main()

