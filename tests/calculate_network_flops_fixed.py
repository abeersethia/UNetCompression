"""
Network FLOPs Calculation - Fixed Version
Calculate FLOPs (Floating Point Operations) for all direct manifold architectures
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def calculate_conv1d_flops(module, input_shape):
    """Calculate FLOPs for Conv1d layer"""
    # Input shape: (B, C_in, L)
    batch_size, in_channels, input_length = input_shape
    
    # Output shape calculation
    output_length = (input_length + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
    out_channels = module.out_channels
    
    # FLOPs = output_elements * (kernel_ops + bias)
    kernel_ops = module.kernel_size[0] * in_channels
    bias_ops = 1 if module.bias is not None else 0
    flops_per_output = kernel_ops + bias_ops
    
    total_flops = batch_size * out_channels * output_length * flops_per_output
    
    return total_flops, (batch_size, out_channels, output_length)

def calculate_linear_flops(module, input_shape):
    """Calculate FLOPs for Linear layer"""
    # Input shape: (B, ..., features)
    batch_size = input_shape[0]
    in_features = module.in_features
    out_features = module.out_features
    
    # FLOPs = batch_size * out_features * (in_features + bias)
    bias_ops = 1 if module.bias is not None else 0
    flops_per_output = in_features + bias_ops
    
    total_flops = batch_size * out_features * flops_per_output
    
    return total_flops, (batch_size, out_features)

def calculate_lstm_flops(module, input_shape):
    """Calculate FLOPs for LSTM layer"""
    # Input shape: (B, L, features) or (B, features, L)
    if len(input_shape) == 3:
        if input_shape[1] > input_shape[2]:  # (B, L, features)
            batch_size, seq_len, input_size = input_shape
        else:  # (B, features, L)
            batch_size, input_size, seq_len = input_shape
    else:
        batch_size, input_size = input_shape
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    
    # LSTM FLOPs calculation (simplified)
    # Each LSTM cell has 4 gates, each gate has matrix multiplication
    # Input to hidden: 4 * (input_size * hidden_size)
    # Hidden to hidden: 4 * (hidden_size * hidden_size)
    # Bias operations: 4 * hidden_size
    
    flops_per_timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    flops_per_layer = batch_size * seq_len * flops_per_timestep
    total_flops = flops_per_layer * num_layers
    
    return total_flops, (batch_size, hidden_size, seq_len)

def calculate_batch_norm_flops(module, input_shape):
    """Calculate FLOPs for BatchNorm1d layer"""
    # BatchNorm operations: normalization + scaling + shifting
    if len(input_shape) == 3:
        batch_size, channels, length = input_shape
    else:
        batch_size, channels = input_shape
        length = 1
    
    # Normalization: subtract mean, divide by std
    # Scaling: multiply by weight
    # Shifting: add bias
    flops_per_element = 3  # subtract, divide, multiply, add (simplified)
    
    total_flops = batch_size * channels * length * flops_per_element
    
    return total_flops, input_shape

def calculate_model_flops(model, input_shape, verbose=False):
    """Calculate total FLOPs for a model"""
    model.eval()
    total_flops = 0
    current_shape = input_shape
    
    if verbose:
        print(f"Input shape: {input_shape}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            flops, output_shape = calculate_conv1d_flops(module, current_shape)
            total_flops += flops
            current_shape = output_shape
            if verbose:
                print(f"  {name}: Conv1d - {flops:,} FLOPs, output: {output_shape}")
                
        elif isinstance(module, nn.Linear):
            # Flatten input for linear layer
            if len(current_shape) > 2:
                batch_size = current_shape[0]
                flattened_size = np.prod(current_shape[1:])
                current_shape = (batch_size, flattened_size)
            
            flops, output_shape = calculate_linear_flops(module, current_shape)
            total_flops += flops
            current_shape = output_shape
            if verbose:
                print(f"  {name}: Linear - {flops:,} FLOPs, output: {output_shape}")
                
        elif isinstance(module, nn.LSTM):
            flops, output_shape = calculate_lstm_flops(module, current_shape)
            total_flops += flops
            current_shape = output_shape
            if verbose:
                print(f"  {name}: LSTM - {flops:,} FLOPs, output: {output_shape}")
                
        elif isinstance(module, nn.BatchNorm1d):
            flops, output_shape = calculate_batch_norm_flops(module, current_shape)
            total_flops += flops
            current_shape = output_shape
            if verbose:
                print(f"  {name}: BatchNorm1d - {flops:,} FLOPs")
    
    return total_flops

def analyze_architecture_flops(ArchClass, name, input_shape=(1, 10, 512), verbose=False):
    """Analyze FLOPs for a specific architecture"""
    print(f"\n🔍 Analyzing {name} FLOPs...")
    
    try:
        # Create reconstructor with appropriate parameters
        if name == "EDGeNet":
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
        
        # Generate sample data to initialize models
        traj, t = generate_lorenz_full(T=20.0, dt=0.01)
        reconstructor.prepare_data(traj, t)
        
        # Create models
        if hasattr(reconstructor, 'create_model'):
            encoder, decoder = reconstructor.create_model()
        elif hasattr(reconstructor, 'create_3d_autoencoder'):
            encoder, decoder = reconstructor.create_3d_autoencoder()
        else:
            print(f"❌ Cannot create model for {name}")
            return None
        
        # Calculate FLOPs for encoder and decoder
        encoder_flops = calculate_model_flops(encoder, input_shape, verbose)
        
        # Estimate decoder input shape based on architecture
        if name == "EDGeNet":
            decoder_input_shape = (1, 3, 256)  # EDGeNet compressed manifold
        elif name == "ModUNet" or name == "Corrected":
            decoder_input_shape = (1, 32, 128)  # ModUNet/Corrected latent shape
        else:
            decoder_input_shape = (1, 32, 256)  # Default latent shape
        
        decoder_flops = calculate_model_flops(decoder, decoder_input_shape, verbose)
        
        total_flops = encoder_flops + decoder_flops
        total_params = count_parameters(encoder) + count_parameters(decoder)
        
        result = {
            'name': name,
            'encoder_flops': encoder_flops,
            'decoder_flops': decoder_flops,
            'total_flops': total_flops,
            'total_params': total_params,
            'flops_per_param': total_flops / total_params if total_params > 0 else 0
        }
        
        print(f"✅ {name}:")
        print(f"   Encoder FLOPs: {encoder_flops:,}")
        print(f"   Decoder FLOPs: {decoder_flops:,}")
        print(f"   Total FLOPs: {total_flops:,}")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   FLOPs/Param: {result['flops_per_param']:.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to calculate FLOPs for all architectures"""
    print("="*80)
    print("NETWORK FLOPs CALCULATION - FIXED VERSION")
    print("="*80)
    print("Calculating FLOPs for all direct manifold architectures")
    print()
    
    # Define architectures to analyze
    architectures = [
        (DirectManifoldMLPReconstructor, "MLP"),
        (DirectManifoldLSTMReconstructor, "LSTM"),
        (DirectManifoldCausalAEReconstructor, "CausalAE"),
        (DirectManifoldEDGeNetReconstructor, "EDGeNet"),
        (DirectManifoldModUNetReconstructor, "ModUNet"),
        (XOnlyManifoldReconstructorCorrected, "Corrected")
    ]
    
    # Analyze each architecture
    results = []
    for ArchClass, name in architectures:
        result = analyze_architecture_flops(ArchClass, name, verbose=False)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No architectures could be analyzed successfully")
        return None
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('total_flops')
    
    # Print comprehensive results table
    print("\n" + "="*100)
    print("COMPREHENSIVE FLOPs ANALYSIS")
    print("="*100)
    
    print(f"{'Architecture':<12} {'Encoder_FLOPs':<15} {'Decoder_FLOPs':<15} {'Total_FLOPs':<15} {'Parameters':<12} {'FLOPs/Param':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['name']:<12} {row['encoder_flops']:<15,} {row['decoder_flops']:<15,} {row['total_flops']:<15,} {row['total_params']:<12,} {row['flops_per_param']:<12.2f}")
    
    # Create visualization
    print("\n🔄 Creating FLOPs visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total FLOPs comparison
    names = df['name'].tolist()
    total_flops = df['total_flops'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars1 = ax1.bar(names, total_flops, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total FLOPs')
    ax1.set_title('Total FLOPs Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add FLOP values on bars
    for bar, flops in zip(bars1, total_flops):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                 f'{flops/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 2. Encoder vs Decoder FLOPs
    encoder_flops = df['encoder_flops'].tolist()
    decoder_flops = df['decoder_flops'].tolist()
    
    x = np.arange(len(names))
    width = 0.35
    
    bars2_enc = ax2.bar(x - width/2, encoder_flops, width, label='Encoder', color='skyblue', alpha=0.7)
    bars2_dec = ax2.bar(x + width/2, decoder_flops, width, label='Decoder', color='lightcoral', alpha=0.7)
    
    ax2.set_ylabel('FLOPs')
    ax2.set_title('Encoder vs Decoder FLOPs', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameters vs FLOPs
    parameters = df['total_params'].tolist()
    
    scatter = ax3.scatter(parameters, total_flops, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('Total FLOPs')
    ax3.set_title('Parameters vs FLOPs', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, name in enumerate(names):
        ax3.annotate(name, (parameters[i], total_flops[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. FLOPs per Parameter efficiency
    flops_per_param = df['flops_per_param'].tolist()
    
    bars4 = ax4.bar(names, flops_per_param, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('FLOPs per Parameter')
    ax4.set_title('Computational Efficiency (FLOPs/Param)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add efficiency values on bars
    for bar, efficiency in zip(bars4, flops_per_param):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{efficiency:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/network_flops_analysis_fixed.png', dpi=300, bbox_inches='tight')
    print("✅ FLOPs analysis saved to: plots/network_flops_analysis_fixed.png")
    
    # Find most and least efficient architectures
    most_efficient = df.loc[df['flops_per_param'].idxmin()]
    least_efficient = df.loc[df['flops_per_param'].idxmax()]
    lowest_flops = df.loc[df['total_flops'].idxmin()]
    highest_flops = df.loc[df['total_flops'].idxmax()]
    
    print("\n" + "="*80)
    print("FLOPs ANALYSIS SUMMARY")
    print("="*80)
    print(f"🎯 Most Efficient (lowest FLOPs/Param): {most_efficient['name']} ({most_efficient['flops_per_param']:.2f})")
    print(f"⚡ Least Efficient (highest FLOPs/Param): {least_efficient['name']} ({least_efficient['flops_per_param']:.2f})")
    print(f"🚀 Lowest Total FLOPs: {lowest_flops['name']} ({lowest_flops['total_flops']:,})")
    print(f"🔥 Highest Total FLOPs: {highest_flops['name']} ({highest_flops['total_flops']:,})")
    
    # Save results to CSV
    df.to_csv('plots/flops_analysis_results_fixed.csv', index=False)
    print(f"📊 Results saved to: plots/flops_analysis_results_fixed.csv")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = main()
