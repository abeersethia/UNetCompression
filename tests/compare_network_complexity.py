"""
Compare Network Complexity: FLOPs and Parameters

Tests all four architectures and reports computational complexity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.core.lorenz import generate_lorenz_full

# Import all architectures
from src.architectures.direct_manifold_mlp import DirectMLPEncoder, DirectMLPDecoder
from src.architectures.direct_manifold_lstm import DirectLSTMEncoder, DirectLSTMDecoder
from src.architectures.direct_manifold_causalae import DirectCausalEncoder, DirectCausalDecoder
from src.architectures.direct_manifold_edgenet import EDGeNet

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_shape):
    """Estimate FLOPs for a model"""
    total_flops = 0
    
    def conv1d_flops(layer, input_size):
        kernel_size = layer.kernel_size[0]
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        # FLOPs = 2 * input_size * kernel_size * in_channels * out_channels
        return 2 * input_size * kernel_size * in_channels * out_channels
    
    def linear_flops(layer):
        # FLOPs = 2 * in_features * out_features
        return 2 * layer.in_features * layer.out_features
    
    # Estimate based on layer types
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            total_flops += conv1d_flops(module, input_shape[-1])
        elif isinstance(module, torch.nn.ConvTranspose1d):
            total_flops += conv1d_flops(module, input_shape[-1])
        elif isinstance(module, torch.nn.Linear):
            total_flops += linear_flops(module)
        elif isinstance(module, torch.nn.LSTM):
            # LSTM FLOPs: 4 gates * (2 * input_size * hidden_size + 2 * hidden_size * hidden_size) * seq_len
            input_size = module.input_size
            hidden_size = module.hidden_size
            seq_len = input_shape[-1]
            total_flops += 4 * (2 * input_size * hidden_size + 2 * hidden_size * hidden_size) * seq_len
        elif isinstance(module, torch.nn.MultiheadAttention):
            # Attention FLOPs: 4 * d_model * seq_len^2 (Q, K, V projections + output)
            d_model = module.embed_dim
            seq_len = input_shape[-1]
            total_flops += 4 * d_model * seq_len * seq_len
    
    return total_flops

def test_mlp():
    """Test MLP architecture"""
    print("\n" + "="*60)
    print("MLP Architecture (Hankel → Direct Signal)")
    print("="*60)
    
    # Hankel input
    input_d = 10
    input_l = 512
    compressed_t = 256
    
    encoder = DirectMLPEncoder(input_d, input_l, compressed_t)
    decoder = DirectMLPDecoder(compressed_t, input_l)
    
    # Set to eval mode to avoid batch norm issues with batch size 1
    encoder.eval()
    decoder.eval()
    
    # Test forward pass
    x = torch.randn(1, input_d, input_l)
    compressed = encoder(x)
    output = decoder(compressed)
    
    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params
    
    # Estimate FLOPs
    encoder_flops = estimate_flops(encoder, (1, input_d, input_l))
    decoder_flops = estimate_flops(decoder, (1, 3, compressed_t))
    total_flops = encoder_flops + decoder_flops
    
    print(f"Input shape: {x.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nParameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nFLOPs (estimated):")
    print(f"  Encoder: {encoder_flops:,}")
    print(f"  Decoder: {decoder_flops:,}")
    print(f"  Total: {total_flops:,}")
    
    return {
        'name': 'MLP',
        'input_type': 'Hankel',
        'params': total_params,
        'flops': total_flops
    }

def test_lstm():
    """Test LSTM architecture"""
    print("\n" + "="*60)
    print("LSTM Architecture (Hankel → Direct Signal)")
    print("="*60)
    
    input_d = 10
    input_l = 512
    compressed_t = 256
    
    encoder = DirectLSTMEncoder(input_d, input_l, compressed_t)
    decoder = DirectLSTMDecoder(compressed_t, input_l)
    
    # Test forward pass
    x = torch.randn(1, input_d, input_l)
    compressed = encoder(x)
    output = decoder(compressed)
    
    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params
    
    # Estimate FLOPs
    encoder_flops = estimate_flops(encoder, (1, input_d, input_l))
    decoder_flops = estimate_flops(decoder, (1, 3, compressed_t))
    total_flops = encoder_flops + decoder_flops
    
    print(f"Input shape: {x.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nParameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nFLOPs (estimated):")
    print(f"  Encoder: {encoder_flops:,}")
    print(f"  Decoder: {decoder_flops:,}")
    print(f"  Total: {total_flops:,}")
    
    return {
        'name': 'LSTM',
        'input_type': 'Hankel',
        'params': total_params,
        'flops': total_flops
    }

def test_causalae():
    """Test CausalAE architecture"""
    print("\n" + "="*60)
    print("CausalAE Architecture (Hankel → Direct Signal)")
    print("="*60)
    
    input_d = 10
    input_l = 512
    compressed_t = 256
    
    encoder = DirectCausalEncoder(input_d, input_l, compressed_t)
    decoder = DirectCausalDecoder(compressed_t, input_l)
    
    # Test forward pass
    x = torch.randn(1, input_d, input_l)
    compressed = encoder(x)
    output = decoder(compressed)
    
    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params
    
    # Estimate FLOPs
    encoder_flops = estimate_flops(encoder, (1, input_d, input_l))
    decoder_flops = estimate_flops(decoder, (1, 3, compressed_t))
    total_flops = encoder_flops + decoder_flops
    
    print(f"Input shape: {x.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nParameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nFLOPs (estimated):")
    print(f"  Encoder: {encoder_flops:,}")
    print(f"  Decoder: {decoder_flops:,}")
    print(f"  Total: {total_flops:,}")
    
    return {
        'name': 'CausalAE',
        'input_type': 'Hankel',
        'params': total_params,
        'flops': total_flops
    }

def test_edgenet():
    """Test EDGeNet architecture"""
    print("\n" + "="*60)
    print("EDGeNet Architecture (Raw Signal → Direct Signal)")
    print("="*60)
    
    signal_length = 2000
    
    model = EDGeNet(
        in_c=1,
        n_classes=3,
        layers=[8, 10, 12, 14, 16],
        downsample=[True, True, True, True],
        k_sz=3,
        up_mode='pixelshuffle',
        pool='pool',
        conv_bridge=True,
        shortcut=True,
        skip_conn=True,
        residual_enc=True,
        residual_dec=True,
        causal_enc=False,
        causal_dec=False,
        conv_encoder='normal'
    )
    
    # Test forward pass
    x = torch.randn(1, 1, signal_length)
    output = model(x)
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Estimate FLOPs
    total_flops = estimate_flops(model, (1, 1, signal_length))
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"\nFLOPs (estimated):")
    print(f"  Total: {total_flops:,}")
    
    return {
        'name': 'EDGeNet',
        'input_type': 'Raw Signal',
        'params': total_params,
        'flops': total_flops
    }

def main():
    print("="*60)
    print("NETWORK COMPLEXITY COMPARISON")
    print("="*60)
    
    results = []
    
    # Test all architectures
    try:
        results.append(test_mlp())
    except Exception as e:
        print(f"\n❌ Error testing MLP: {e}")
    
    try:
        results.append(test_lstm())
    except Exception as e:
        print(f"\n❌ Error testing LSTM: {e}")
    
    try:
        results.append(test_causalae())
    except Exception as e:
        print(f"\n❌ Error testing CausalAE: {e}")
    
    try:
        results.append(test_edgenet())
    except Exception as e:
        print(f"\n❌ Error testing EDGeNet: {e}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPLEXITY COMPARISON TABLE")
    print("="*60)
    print(f"\n{'Architecture':<15} {'Input Type':<15} {'Parameters':<15} {'FLOPs':<20} {'Params/MFLOP':<15}")
    print("-" * 85)
    
    for result in results:
        name = result['name']
        input_type = result['input_type']
        params = result['params']
        flops = result['flops']
        efficiency = params / (flops / 1e6) if flops > 0 else 0
        
        print(f"{name:<15} {input_type:<15} {params:<15,} {flops/1e6:<20.2f} {efficiency:<15.2f}")
    
    # Print insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    if results:
        lightest = min(results, key=lambda r: r['params'])
        heaviest = max(results, key=lambda r: r['params'])
        most_compute = max(results, key=lambda r: r['flops'])
        least_compute = min(results, key=lambda r: r['flops'])
        
        print(f"💡 Lightest model: {lightest['name']} ({lightest['params']:,} parameters)")
        print(f"🏋️  Heaviest model: {heaviest['name']} ({heaviest['params']:,} parameters)")
        print(f"⚡ Least compute: {least_compute['name']} ({least_compute['flops']/1e6:.2f} MFLOPs)")
        print(f"🔥 Most compute: {most_compute['name']} ({most_compute['flops']/1e6:.2f} MFLOPs)")
        
        print(f"\n📊 Parameter range: {lightest['params']:,} → {heaviest['params']:,} ({heaviest['params']/lightest['params']:.1f}x)")
        print(f"📊 FLOPs range: {least_compute['flops']/1e6:.2f} → {most_compute['flops']/1e6:.2f} MFLOPs ({most_compute['flops']/least_compute['flops']:.1f}x)")
    
    print("\n✅ Complexity analysis completed!")
    return results

if __name__ == "__main__":
    results = main()

