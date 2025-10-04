"""
Direct Manifold Reconstruction - CausalAE Architecture

Pipeline:
1. Input: Hankel matrix of X component (B, delay_dim, window_len)
2. Encoder: Hankel → Compressed Manifold (B, 3, T) using Causal Convolutions
3. Decoder: Compressed Manifold → Direct Time-Domain Signals (B, 3, T) using Deconvolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_base import DirectManifoldBaseReconstructor, count_parameters

class CausalConv1d(nn.Module):
    """Causal 1D Convolution that respects temporal causality"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        # Remove future information by trimming the end
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class DirectCausalEncoder(nn.Module):
    """Causal Encoder: Hankel → Compressed Manifold"""
    def __init__(self, input_d, input_l, output_t):
        super().__init__()
        self.input_d = input_d
        self.input_l = input_l
        self.output_t = output_t
        
        # Causal convolutional layers with increasing dilation
        self.causal_conv1 = CausalConv1d(input_d, 32, kernel_size=3, dilation=1)
        self.causal_conv2 = CausalConv1d(32, 64, kernel_size=3, dilation=2)
        self.causal_conv3 = CausalConv1d(64, 128, kernel_size=3, dilation=4)
        self.causal_conv4 = CausalConv1d(128, 3, kernel_size=3, dilation=8)  # Output 3 channels
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(3)
        
        self.dropout = nn.Dropout(0.3)
        
        # Temporal compression
        self.temporal_compress = nn.AdaptiveAvgPool1d(output_t)
        
    def forward(self, x):
        # x: (B, input_d, input_l) Hankel matrix
        
        # Causal convolution layers
        x = F.relu(self.bn1(self.causal_conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.causal_conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.causal_conv3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.causal_conv4(x)))  # (B, 3, L)
        x = self.dropout(x)
        
        # Temporal compression
        x = self.temporal_compress(x)  # (B, 3, output_t)
        
        return x

class DirectCausalDecoder(nn.Module):
    """Causal Decoder: Compressed Manifold → Direct Signals"""
    def __init__(self, input_t, output_t):
        super().__init__()
        self.input_t = input_t
        self.output_t = output_t
        
        # Temporal expansion
        self.temporal_expand = nn.Upsample(size=output_t, mode='linear', align_corners=False)
        
        # Deconvolution layers (reverse of encoder)
        self.deconv1 = nn.ConvTranspose1d(3, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose1d(32, 3, kernel_size=3, padding=1)  # Output 3 channels
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (B, 3, input_t) compressed manifold
        
        # Expand temporally
        x = self.temporal_expand(x)  # (B, 3, output_t)
        
        # Deconvolution layers
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.dropout(x)
        
        x = self.deconv4(x)  # (B, 3, output_t)
        
        return x

class DirectManifoldCausalAEReconstructor(DirectManifoldBaseReconstructor):
    """Direct Manifold Reconstructor using CausalAE"""
    
    def create_model(self):
        """Create encoder-decoder model"""
        self.encoder = DirectCausalEncoder(
            input_d=self.delay_embedding_dim,
            input_l=self.window_len,
            output_t=self.compressed_t
        )
        
        self.decoder = DirectCausalDecoder(
            input_t=self.compressed_t,
            output_t=self.window_len
        )
        
        return self.encoder, self.decoder
    
    def train(self, max_epochs=150, base_noise_std=0.1, patience=25, verbose=True):
        """Train the model"""
        print(f"\n=== TRAINING DIRECT MANIFOLD CAUSALAE ===")
        print(f"Architecture: CausalAE (Hankel → Direct Signal)")
        print(f"Compressed manifold size: (B, 3, {self.compressed_t})")
        
        self.create_model()
        return super().train(max_epochs, base_noise_std, patience, verbose)

def main():
    """Test Direct Manifold CausalAE"""
    print("=== DIRECT MANIFOLD RECONSTRUCTION (CAUSALAE) ===")
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor
    reconstructor = DirectManifoldCausalAEReconstructor(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        compressed_t=256,
        train_split=0.7
    )
    
    # Prepare data and train
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=100, verbose=True)
    
    # Reconstruct manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    print(f"\n=== DIRECT CAUSALAE SUMMARY ===")
    print(f"✅ Architecture: Direct Manifold CausalAE")
    print(f"📊 Parameters: {training_history['total_parameters']:,}")
    print(f"⏱️  Training Time: {training_history['training_time']:.2f}s")
    print(f"🔗 Correlations: X={metrics['correlations']['X']:.4f}, Y={metrics['correlations']['Y']:.4f}, Z={metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics

if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()

