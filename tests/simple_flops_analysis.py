"""
Simple FLOPs Analysis
Robust FLOPs calculation for all architectures including EDGeNet
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

def estimate_flops_simple(model, input_shape, verbose=False):
    """Simple FLOP estimation based on layer types and parameters"""
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            # Conv1d FLOPs estimation
            try:
                # Get layer parameters safely
                kernel_size = getattr(module, 'kernel_size', (3,))
                if isinstance(kernel_size, tuple):
                    kernel_size = kernel_size[0]
                
                stride = getattr(module, 'stride', (1,))
                if isinstance(stride, tuple):
                    stride = stride[0]
                
                padding = getattr(module, 'padding', (0,))
                if isinstance(padding, tuple):
                    padding = padding[0]
                
                # Estimate output size
                input_length = input_shape[2] if len(input_shape) > 2 else 1
                output_length = (input_length + 2 * padding - kernel_size) // stride + 1
                
                # FLOPs = output_elements * (kernel_ops + bias)
                kernel_ops = kernel_size * module.in_channels
                bias_ops = 1 if module.bias is not None else 0
                flops_per_output = kernel_ops + bias_ops
                
                layer_flops = input_shape[0] * module.out_channels * output_length * flops_per_output
                total_flops += layer_flops
                
                if verbose:
                    print(f"  {name}: Conv1d - {layer_flops:,} FLOPs")
                    
            except Exception as e:
                if verbose:
                    print(f"  {name}: Conv1d - Error calculating FLOPs: {e}")
                # Fallback estimation
                total_flops += module.in_channels * module.out_channels * 1000  # Rough estimate
                
        elif isinstance(module, nn.Linear):
            # Linear FLOPs
            try:
                bias_ops = 1 if module.bias is not None else 0
                layer_flops = input_shape[0] * module.out_features * (module.in_features + bias_ops)
                total_flops += layer_flops
                
                if verbose:
                    print(f"  {name}: Linear - {layer_flops:,} FLOPs")
                    
            except Exception as e:
                if verbose:
                    print(f"  {name}: Linear - Error calculating FLOPs: {e}")
                # Fallback estimation
                total_flops += module.in_features * module.out_features * 10  # Rough estimate
                
        elif isinstance(module, nn.LSTM):
            # LSTM FLOPs (simplified)
            try:
                hidden_size = module.hidden_size
                input_size = module.input_size if hasattr(module, 'input_size') else 10
                num_layers = module.num_layers
                
                # Simplified LSTM FLOPs calculation
                flops_per_timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size)
                layer_flops = input_shape[0] * input_shape[1] * flops_per_timestep * num_layers
                total_flops += layer_flops
                
                if verbose:
                    print(f"  {name}: LSTM - {layer_flops:,} FLOPs")
                    
            except Exception as e:
                if verbose:
                    print(f"  {name}: LSTM - Error calculating FLOPs: {e}")
                # Fallback estimation
                total_flops += 1000000  # Rough estimate
    
    return total_flops

def analyze_architecture_simple(ArchClass, name, verbose=False):
    """Analyze FLOPs for a specific architecture with simple estimation"""
    print(f"\n🔍 Analyzing {name} FLOPs (Simple Method)...")
    
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
        input_shape = (1, 10, 512)  # Standard input shape
        
        if verbose:
            print(f"  Analyzing encoder with input shape: {input_shape}")
        encoder_flops = estimate_flops_simple(encoder, input_shape, verbose)
        
        # Estimate decoder input shape based on architecture
        if name == "EDGeNet":
            decoder_input_shape = (1, 3, 256)  # EDGeNet compressed manifold
        elif name == "ModUNet" or name == "Corrected":
            decoder_input_shape = (1, 32, 128)  # ModUNet/Corrected latent shape
        else:
            decoder_input_shape = (1, 32, 256)  # Default latent shape
        
        if verbose:
            print(f"  Analyzing decoder with input shape: {decoder_input_shape}")
        decoder_flops = estimate_flops_simple(decoder, decoder_input_shape, verbose)
        
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
    print("SIMPLE FLOPs ANALYSIS - ALL ARCHITECTURES")
    print("="*80)
    print("Calculating FLOPs for all direct manifold architectures including EDGeNet")
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
        result = analyze_architecture_simple(ArchClass, name, verbose=False)
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
    print("COMPREHENSIVE FLOPs ANALYSIS - ALL ARCHITECTURES")
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
    ax1.set_title('Total FLOPs Comparison (All Architectures)', fontsize=14, fontweight='bold')
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
    plt.savefig('plots/network_flops_analysis_complete.png', dpi=300, bbox_inches='tight')
    print("✅ Complete FLOPs analysis saved to: plots/network_flops_analysis_complete.png")
    
    # Find most and least efficient architectures
    most_efficient = df.loc[df['flops_per_param'].idxmin()]
    least_efficient = df.loc[df['flops_per_param'].idxmax()]
    lowest_flops = df.loc[df['total_flops'].idxmin()]
    highest_flops = df.loc[df['total_flops'].idxmax()]
    
    print("\n" + "="*80)
    print("FLOPs ANALYSIS SUMMARY - ALL ARCHITECTURES")
    print("="*80)
    print(f"🎯 Most Efficient (lowest FLOPs/Param): {most_efficient['name']} ({most_efficient['flops_per_param']:.2f})")
    print(f"⚡ Least Efficient (highest FLOPs/Param): {least_efficient['name']} ({least_efficient['flops_per_param']:.2f})")
    print(f"🚀 Lowest Total FLOPs: {lowest_flops['name']} ({lowest_flops['total_flops']:,})")
    print(f"🔥 Highest Total FLOPs: {highest_flops['name']} ({highest_flops['total_flops']:,})")
    
    # Check if EDGeNet is included
    edgenet_result = df[df['name'] == 'EDGeNet']
    if not edgenet_result.empty:
        print(f"✅ EDGeNet successfully analyzed: {edgenet_result.iloc[0]['total_flops']:,} FLOPs")
    else:
        print("❌ EDGeNet not included in results")
    
    # Save results to CSV
    df.to_csv('plots/flops_analysis_complete.csv', index=False)
    print(f"📊 Complete results saved to: plots/flops_analysis_complete.csv")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = main()
