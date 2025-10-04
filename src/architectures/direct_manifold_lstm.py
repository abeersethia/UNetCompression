"""
Direct Manifold Reconstruction - LSTM Architecture

Pipeline:
1. Input: Hankel matrix of X component (B, delay_dim, window_len)
2. Encoder: Hankel → Compressed Manifold (B, 3, T) using LSTM
3. Decoder: Compressed Manifold → Direct Time-Domain Signals (B, 3, T) using LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.lorenz import generate_lorenz_full
from src.architectures.direct_manifold_base import DirectManifoldBaseReconstructor, count_parameters

class DirectLSTMEncoder(nn.Module):
    """LSTM Encoder: Hankel → Compressed Manifold"""
    def __init__(self, input_d, input_l, output_t):
        super().__init__()
        self.input_d = input_d
        self.input_l = input_l
        self.output_t = output_t
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_d, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True, dropout=0.2)
        
        # Project to 3 channels (X, Y, Z)
        self.projection = nn.Linear(32, 3)
        
        # Temporal compression
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_t)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (B, input_d, input_l) Hankel matrix
        # Transpose for LSTM: (B, L, D)
        x = x.transpose(1, 2)  # (B, input_l, input_d)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        
        # Project to 3 channels
        x = self.projection(x)  # (B, input_l, 3)
        
        # Transpose and compress temporally
        x = x.transpose(1, 2)  # (B, 3, input_l)
        x = self.temporal_pool(x)  # (B, 3, output_t)
        
        return x

class DirectLSTMDecoder(nn.Module):
    """LSTM Decoder: Compressed Manifold → Direct Signals"""
    def __init__(self, input_t, output_t):
        super().__init__()
        self.input_t = input_t
        self.output_t = output_t
        
        # Temporal expansion
        self.temporal_expand = nn.Upsample(size=output_t, mode='linear', align_corners=False)
        
        # Project from 3 channels
        self.input_proj = nn.Linear(3, 32)
        
        # LSTM layers (reverse of encoder)
        self.lstm1 = nn.LSTM(32, 64, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(128, 3, batch_first=True, dropout=0.3)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (B, 3, input_t) compressed manifold
        
        # Expand temporally
        x = self.temporal_expand(x)  # (B, 3, output_t)
        
        # Transpose for LSTM
        x = x.transpose(1, 2)  # (B, output_t, 3)
        
        # Project
        x = self.input_proj(x)  # (B, output_t, 32)
        x = self.dropout(x)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)  # (B, output_t, 3)
        
        # Transpose back
        x = x.transpose(1, 2)  # (B, 3, output_t)
        
        return x

class DirectManifoldLSTMReconstructor(DirectManifoldBaseReconstructor):
    """Direct Manifold Reconstructor using LSTM"""
    
    def create_model(self):
        """Create encoder-decoder model"""
        self.encoder = DirectLSTMEncoder(
            input_d=self.delay_embedding_dim,
            input_l=self.window_len,
            output_t=self.compressed_t
        )
        
        self.decoder = DirectLSTMDecoder(
            input_t=self.compressed_t,
            output_t=self.window_len
        )
        
        return self.encoder, self.decoder
    
    def train(self, max_epochs=150, base_noise_std=0.1, patience=25, verbose=True):
        """Train the model"""
        print(f"\n=== TRAINING DIRECT MANIFOLD LSTM ===")
        print(f"Architecture: LSTM (Hankel → Direct Signal)")
        print(f"Compressed manifold size: (B, 3, {self.compressed_t})")
        
        self.create_model()
        return super().train(max_epochs, base_noise_std, patience, verbose)

def main():
    """Test Direct Manifold LSTM"""
    print("=== DIRECT MANIFOLD RECONSTRUCTION (LSTM) ===")
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor
    reconstructor = DirectManifoldLSTMReconstructor(
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
    
    print(f"\n=== DIRECT LSTM SUMMARY ===")
    print(f"✅ Architecture: Direct Manifold LSTM")
    print(f"📊 Parameters: {training_history['total_parameters']:,}")
    print(f"⏱️  Training Time: {training_history['training_time']:.2f}s")
    print(f"🔗 Correlations: X={metrics['correlations']['X']:.4f}, Y={metrics['correlations']['Y']:.4f}, Z={metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics

if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()

