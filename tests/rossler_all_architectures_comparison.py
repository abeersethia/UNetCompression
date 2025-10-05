"""
Rössler Attractor X-Only Reconstruction - All Architectures Comparison
Compares MLP, LSTM, CausalAE, EDGeNet, and Corrected architectures on Rössler attractor
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
from tqdm import tqdm

# Import all architectures and Rössler generator
from src.core.rossler import generate_rossler_full, visualize_rossler_attractor
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructionCorrected

def test_all_rossler_architectures():
    """Test all architectures on Rössler attractor reconstruction"""
    print("="*100)
    print("RÖSSLER ATTRACTOR X-ONLY RECONSTRUCTION - ALL ARCHITECTURES COMPARISON")
    print("="*100)
    print("Comparing MLP, LSTM, CausalAE, EDGeNet, and Corrected architectures")
    print("on Rössler attractor X-only reconstruction")
    print()
    
    # Generate Rössler attractor
    print("🔄 Generating Rössler attractor...")
    traj, t = generate_rossler_full(T=100.0, dt=0.01)  # Longer duration for full state space
    print(f"✅ Rössler trajectory shape: {traj.shape}")
    print(f"   X range: [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}]")
    print(f"   Y range: [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}]")
    print(f"   Z range: [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}]")
    print()
    
    # Define architectures to test
    architectures = {
        'MLP': {
            'class': DirectManifoldMLPReconstructor,
            'epochs': 150,
            'color': 'blue',
            'description': 'Multi-Layer Perceptron'
        },
        'LSTM': {
            'class': DirectManifoldLSTMReconstructor,
            'epochs': 150,
            'color': 'green',
            'description': 'Long Short-Term Memory'
        },
        'CausalAE': {
            'class': DirectManifoldCausalAEReconstructor,
            'epochs': 150,
            'color': 'red',
            'description': 'Causal Autoencoder'
        },
        'EDGeNet': {
            'class': DirectManifoldEDGeNetReconstructor,
            'epochs': 250,
            'color': 'orange',
            'description': 'Enhanced Dynamic Graph Edge Network'
        },
        'Corrected': {
            'class': XOnlyManifoldReconstructionCorrected,
            'epochs': 150,
            'color': 'purple',
            'description': 'Corrected Architecture'
        }
    }
    
    # Store results
    results = {}
    reconstructors = {}
    original_attractors = {}
    reconstructed_attractors = {}
    
    # Test each architecture
    for arch_name, arch_config in tqdm(architectures.items(), desc="Testing architectures"):
        print(f"\n{'='*60}")
        print(f"TESTING {arch_name.upper()} ARCHITECTURE")
        print(f"{'='*60}")
        print(f"Description: {arch_config['description']}")
        print(f"Epochs: {arch_config['epochs']}")
        
        try:
            # Create reconstructor
            print(f"🔄 Creating {arch_name} reconstructor...")
            reconstructor = arch_config['class'](
                window_len=512,
                delay_embedding_dim=10,
                stride=5,
                compressed_t=256,
                train_split=0.7
            )
            
            # Prepare data and train
            print(f"🔄 Training {arch_name}...")
            reconstructor.prepare_data(traj, t)
            training_history = reconstructor.train(max_epochs=arch_config['epochs'], verbose=False)
            
            # Reconstruct manifold
            print(f"🔄 Reconstructing with {arch_name}...")
            original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
            
            # Store results
            results[arch_name] = {
                'metrics': metrics,
                'training_history': training_history,
                'color': arch_config['color'],
                'description': arch_config['description']
            }
            reconstructors[arch_name] = reconstructor
            original_attractors[arch_name] = original_attractor
            reconstructed_attractors[arch_name] = reconstructed_attractor
            
            print(f"✅ {arch_name} completed successfully!")
            print(f"   X Correlation: {metrics['correlations']['X']:.4f}")
            print(f"   Y Correlation: {metrics['correlations']['Y']:.4f}")
            print(f"   Z Correlation: {metrics['correlations']['Z']:.4f}")
            print(f"   Mean Error: {metrics['mean_error']:.4f}")
            
        except Exception as e:
            print(f"❌ {arch_name} failed: {str(e)}")
            continue
    
    # Create comprehensive comparison visualization
    print(f"\n🔄 Creating comprehensive Rössler comparison visualization...")
    n_archs = len(results)
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Original Rössler Attractor (3D)
    ax1 = fig.add_subplot(3, n_archs + 1, 1, projection='3d')
    ax1.plot(original_attractors[list(results.keys())[0]][:, 0], 
             original_attractors[list(results.keys())[0]][:, 1], 
             original_attractors[list(results.keys())[0]][:, 2], 
             alpha=0.8, linewidth=1, color='black', label='Original')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Rössler\nAttractor', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2-6. Reconstructed Attractors (3D)
    for i, (arch_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(3, n_archs + 1, i + 2, projection='3d')
        ax.plot(reconstructed_attractors[arch_name][:, 0], 
                reconstructed_attractors[arch_name][:, 1], 
                reconstructed_attractors[arch_name][:, 2], 
                alpha=0.8, linewidth=1, color=result['color'], label=f'{arch_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{arch_name}\nReconstructed', fontsize=12, fontweight='bold')
        ax.legend()
    
    # 7-11. X-Y Projections
    for i, (arch_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(3, n_archs + 1, n_archs + 2 + i)
        ax.plot(original_attractors[arch_name][:, 0], original_attractors[arch_name][:, 1], 
                alpha=0.6, linewidth=1, color='black', label='Original')
        ax.plot(reconstructed_attractors[arch_name][:, 0], reconstructed_attractors[arch_name][:, 1], 
                alpha=0.8, linewidth=1, color=result['color'], label=f'{arch_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{arch_name} X-Y\nProjection', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 12-16. Time Series - Y Component (Reconstructed)
    for i, (arch_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(3, n_archs + 1, 2 * n_archs + 2 + i)
        ax.plot(t, original_attractors[arch_name][:, 1], 
                alpha=0.6, linewidth=1, color='black', label='Original Y')
        ax.plot(t, reconstructed_attractors[arch_name][:, 1], 
                alpha=0.8, linewidth=1, color=result['color'], label=f'{arch_name} Y')
        ax.set_xlabel('Time')
        ax.set_ylabel('Y Value')
        ax.set_title(f'{arch_name} Y\nComponent', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Rössler Attractor X-Only Reconstruction - All Architectures Comparison\nFrom X Component to Full Attractor', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('plots/rossler_all_architectures_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Rössler all architectures comparison saved to: plots/rossler_all_architectures_comparison.png")
    
    # Create performance comparison table
    print(f"\n🔄 Creating performance comparison table...")
    performance_data = []
    
    for arch_name, result in results.items():
        metrics = result['metrics']
        training_history = result['training_history']
        
        # Handle different MSE formats
        mse_info = "N/A"
        if 'mse_denormalized' in metrics:
            mse_info = f"{metrics['mse_denormalized']['Mean']:.4f}"
        elif 'mse' in metrics:
            mse_info = f"{metrics['mse']['Mean']:.4f}"
        
        performance_data.append({
            'Architecture': arch_name,
            'Description': result['description'],
            'Parameters': training_history['total_parameters'],
            'Training Time (s)': training_history['training_time'],
            'X Correlation': metrics['correlations']['X'],
            'Y Correlation': metrics['correlations']['Y'],
            'Z Correlation': metrics['correlations']['Z'],
            'Mean Correlation': np.mean([metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']]),
            'Mean Error': metrics['mean_error'],
            'MSE': mse_info
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(performance_data)
    df = df.sort_values('Mean Correlation', ascending=False)
    df.to_csv('plots/rossler_all_architectures_comparison_results.csv', index=False)
    
    # Print results table
    print("\n" + "="*100)
    print("RÖSSLER ATTRACTOR X-ONLY RECONSTRUCTION - PERFORMANCE COMPARISON")
    print("="*100)
    print(df.to_string(index=False, float_format='%.4f'))
    print()
    
    # Find best performers
    best_correlation = df.loc[df['Mean Correlation'].idxmax()]
    best_error = df.loc[df['Mean Error'].idxmin()]
    fastest = df.loc[df['Training Time (s)'].idxmin()]
    
    print("🏆 BEST PERFORMERS:")
    print(f"   🎯 Best Correlation: {best_correlation['Architecture']} ({best_correlation['Mean Correlation']:.4f})")
    print(f"   📉 Lowest Error: {best_error['Architecture']} ({best_error['Mean Error']:.4f})")
    print(f"   ⚡ Fastest Training: {fastest['Architecture']} ({fastest['Training Time (s)']:.2f}s)")
    print()
    
    print("✅ SUCCESS: All architectures tested on Rössler attractor!")
    print("🎯 Rössler attractor (simpler than Lorenz) provides a good testbed for X-only reconstruction.")
    print("📊 Results saved to: plots/rossler_all_architectures_comparison_results.csv")
    
    plt.show()
    return results, reconstructors, original_attractors, reconstructed_attractors, df

if __name__ == "__main__":
    results, reconstructors, original_attractors, reconstructed_attractors, df = test_all_rossler_architectures()
