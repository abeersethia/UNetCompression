"""
All Architecture Comparison
Comprehensive comparison of all direct manifold architectures with proper training epochs
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import pandas as pd

# Import all architectures
from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_mlp import DirectManifoldMLPReconstructor
from src.architectures.direct_manifold_lstm import DirectManifoldLSTMReconstructor
from src.architectures.direct_manifold_causalae import DirectManifoldCausalAEReconstructor
from src.architectures.direct_manifold_edgenet import DirectManifoldEDGeNetReconstructor
from src.architectures.direct_manifold_modunet import DirectManifoldModUNetReconstructor
from src.architectures.x_only_manifold_reconstruction_corrected import XOnlyManifoldReconstructorCorrected

def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_shape):
    """Estimate FLOPs for a model with given input shape"""
    try:
        # Simple FLOP estimation based on operations
        total_flops = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                # Conv FLOPs = output_elements * (kernel_size * in_channels + bias)
                if hasattr(module, 'kernel_size'):
                    kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                else:
                    kernel_size = 1
                
                # Estimate output size (simplified)
                if len(input_shape) == 3:  # (B, C, L)
                    output_elements = input_shape[0] * module.out_channels * (input_shape[2] // module.stride[0])
                else:
                    output_elements = input_shape[0] * module.out_channels * input_shape[2]
                
                flops = output_elements * (kernel_size * module.in_channels + (1 if module.bias is not None else 0))
                total_flops += flops
                
            elif isinstance(module, torch.nn.Linear):
                # Linear FLOPs = output_features * (input_features + bias)
                flops = module.out_features * (module.in_features + (1 if module.bias is not None else 0))
                total_flops += flops
                
        return total_flops
    except:
        return 0

def test_architecture(ArchClass, name, traj, t, max_epochs=150, patience=25):
    """Test a single architecture with proper parameters"""
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
    elif name == "Corrected":
        # Use different latent parameters for corrected version
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            latent_d=32,
            latent_l=128,
            train_split=0.7
        )
    else:
        # Standard parameters for other architectures
        reconstructor = ArchClass(
            window_len=512,
            delay_embedding_dim=10,
            stride=5,
            compressed_t=256,
            train_split=0.7
        )
    
    # Prepare data
    reconstructor.prepare_data(traj, t)
    
    # Train with timing
    start_time = time.time()
    training_history = reconstructor.train(max_epochs=max_epochs, patience=patience, verbose=False)
    training_time = time.time() - start_time
    
    # Reconstruct manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Count parameters and estimate FLOPs
    total_params = 0
    total_flops = 0
    
    try:
        if hasattr(reconstructor, 'encoder') and hasattr(reconstructor, 'decoder'):
            total_params = count_parameters(reconstructor.encoder) + count_parameters(reconstructor.decoder)
            # Estimate FLOPs with sample input
            sample_input = torch.randn(1, 10, 512)
            encoder_flops = estimate_flops(reconstructor.encoder, sample_input.shape)
            decoder_flops = estimate_flops(reconstructor.decoder, (1, 32, 256))  # Approximate latent shape
            total_flops = encoder_flops + decoder_flops
    except:
        pass
    
    # Add training time to history
    training_history['training_time'] = training_time
    training_history['total_parameters'] = total_params
    training_history['total_flops'] = total_flops
    
    return {
        'name': name,
        'reconstructor': reconstructor,
        'original': original_attractor,
        'reconstructed': reconstructed_attractor,
        'metrics': metrics,
        'training_history': training_history
    }

def create_comprehensive_comparison():
    """Create comprehensive comparison of all architectures"""
    print("="*80)
    print("ALL ARCHITECTURE COMPREHENSIVE COMPARISON")
    print("="*80)
    print("Comparing MLP, LSTM, CausalAE, EDGeNet, ModUNet, and Corrected")
    print("EDGeNet: 250 epochs, Others: 150 epochs")
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
                result = test_architecture(ArchClass, name, traj, t)
                results.append(result)
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
            pbar.update(1)
    
    # Create comprehensive visualization
    print("🔄 Creating comprehensive comparison visualization...")
    fig = plt.figure(figsize=(30, 20))
    
    # Create time vector
    time_axis = np.linspace(0, 20.0, len(traj))
    
    # Plot original attractor
    ax_orig = fig.add_subplot(4, len(results) + 1, 1, projection='3d')
    ax_orig.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.8, linewidth=1, color='blue')
    ax_orig.set_title('Original Lorenz', fontsize=12, fontweight='bold')
    ax_orig.set_xlabel('X'); ax_orig.set_ylabel('Y'); ax_orig.set_zlabel('Z')
    
    # Plot each architecture
    for i, result in enumerate(results):
        name = result['name']
        reconstructed = result['reconstructed']
        metrics = result['metrics']
        training_history = result['training_history']
        
        # 3D Reconstruction
        ax_3d = fig.add_subplot(4, len(results) + 1, i + 2, projection='3d')
        ax_3d.plot(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
                   alpha=0.8, linewidth=1, color='red')
        ax_3d.set_title(f'{name} (3D)', fontsize=12, fontweight='bold')
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
        
        # X-Y Projection
        ax_xy = fig.add_subplot(4, len(results) + 1, len(results) + 1 + i + 2)
        ax_xy.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=0.5, color='blue', label='Original')
        ax_xy.plot(reconstructed[:, 0], reconstructed[:, 1], alpha=0.8, linewidth=1, color='red', label=name)
        ax_xy.set_title(f'{name} (X-Y)', fontsize=10)
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True, alpha=0.3)
        
        # Y-Z Projection
        ax_yz = fig.add_subplot(4, len(results) + 1, 2 * (len(results) + 1) + i + 2)
        ax_yz.plot(traj[:, 1], traj[:, 2], alpha=0.5, linewidth=0.5, color='blue', label='Original')
        ax_yz.plot(reconstructed[:, 1], reconstructed[:, 2], alpha=0.8, linewidth=1, color='red', label=name)
        ax_yz.set_title(f'{name} (Y-Z)', fontsize=10)
        ax_yz.set_xlabel('Y'); ax_yz.set_ylabel('Z')
        ax_yz.legend(fontsize=8)
        ax_yz.grid(True, alpha=0.3)
        
        # Time Series Comparison
        ax_ts = fig.add_subplot(4, len(results) + 1, 3 * (len(results) + 1) + i + 2)
        ax_ts.plot(time_axis, traj[:, 0], alpha=0.5, linewidth=0.5, color='blue', label='Original X')
        ax_ts.plot(time_axis, reconstructed[:, 0], alpha=0.8, linewidth=1, color='red', label=f'{name} X')
        ax_ts.set_title(f'{name} (Time Series)', fontsize=10)
        ax_ts.set_xlabel('Time'); ax_ts.set_ylabel('X Value')
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)
    
    plt.suptitle('All Architecture Comparison\nLorenz Attractor Reconstruction (EDGeNet: 250 epochs, Others: 150 epochs)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/all_architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ All architecture comparison saved to: plots/all_architecture_comparison.png")
    
    # Create performance comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*100)
    
    performance_data = []
    for result in results:
        name = result['name']
        metrics = result['metrics']
        training_history = result['training_history']
        
        # Handle NaN correlations
        corr_x = metrics['correlations']['X']
        corr_y = metrics['correlations']['Y']
        corr_z = metrics['correlations']['Z']
        
        if np.isnan(corr_x): corr_x = 0.0
        if np.isnan(corr_y): corr_y = 0.0
        if np.isnan(corr_z): corr_z = 0.0
        
        avg_corr = np.mean([corr_x, corr_y, corr_z])
        
        performance_data.append({
            'Architecture': name,
            'X_Correlation': corr_x,
            'Y_Correlation': corr_y,
            'Z_Correlation': corr_z,
            'Avg_Correlation': avg_corr,
            'Mean_Error': metrics['mean_error'],
            'Parameters': training_history.get('total_parameters', 0),
            'FLOPs': training_history.get('total_flops', 0),
            'Training_Time': training_history.get('training_time', 0),
            'Epochs': 250 if name == 'EDGeNet' else 150
        })
    
    df = pd.DataFrame(performance_data)
    df = df.sort_values('Avg_Correlation', ascending=False)
    
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Create performance visualization
    print("\n🔄 Creating performance comparison visualization...")
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Correlation Comparison
    arch_names = df['Architecture'].tolist()
    avg_corrs = df['Avg_Correlation'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(arch_names)))
    
    bars1 = ax1.bar(arch_names, avg_corrs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Correlation')
    ax1.set_title('Average Correlation Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for bar, corr in zip(bars1, avg_corrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training Time Comparison
    training_times = df['Training_Time'].tolist()
    bars2 = ax2.bar(arch_names, training_times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add time values on bars
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Parameters Comparison
    parameters = df['Parameters'].tolist()
    bars3 = ax3.bar(arch_names, parameters, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Parameters')
    ax3.set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add parameter values on bars
    for bar, params in zip(bars3, parameters):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 4. FLOPs Comparison
    flops = df['FLOPs'].tolist()
    bars4 = ax4.bar(arch_names, flops, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('FLOPs (estimated)')
    ax4.set_title('Computational Complexity (FLOPs)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add FLOP values on bars
    for bar, flop_val in zip(bars4, flops):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{flop_val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Performance comparison saved to: plots/performance_comparison.png")
    
    # Find and highlight best performers
    best_correlation = df.loc[df['Avg_Correlation'].idxmax()]
    fastest_training = df.loc[df['Training_Time'].idxmin()]
    most_efficient = df.loc[df['Parameters'].idxmin()]
    
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    print(f"🏆 Best Correlation: {best_correlation['Architecture']} (Avg: {best_correlation['Avg_Correlation']:.4f})")
    print(f"⚡ Fastest Training: {fastest_training['Architecture']} ({fastest_training['Training_Time']:.1f}s)")
    print(f"🎯 Most Efficient: {most_efficient['Architecture']} ({most_efficient['Parameters']:,} params)")
    
    # Efficiency analysis
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80)
    df['Efficiency_Score'] = df['Avg_Correlation'] / (df['Parameters'] / 1e6)  # Correlation per million parameters
    df['Speed_Score'] = df['Avg_Correlation'] / (df['Training_Time'] / 60)  # Correlation per minute of training
    
    best_efficiency = df.loc[df['Efficiency_Score'].idxmax()]
    best_speed = df.loc[df['Speed_Score'].idxmax()]
    
    print(f"🎯 Most Parameter-Efficient: {best_efficiency['Architecture']} ({best_efficiency['Efficiency_Score']:.2f} corr/M params)")
    print(f"⚡ Best Training Speed: {best_speed['Architecture']} ({best_speed['Speed_Score']:.2f} corr/min)")
    
    plt.show()
    return results, df

if __name__ == "__main__":
    results, df = create_comprehensive_comparison()
