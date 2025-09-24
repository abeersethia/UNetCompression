"""
Direct Manifold-to-Signal Autoencoder for Attractor Reconstruction
Encoder: Hankel matrix → Compressed manifold
Decoder: Compressed manifold → Direct time-domain signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectManifoldAutoencoder(nn.Module):
    """Direct manifold-to-signal autoencoder"""
    
    def __init__(self, hankel_size, signal_length, latent_dim=32):
        super().__init__()
        
        self.hankel_size = hankel_size  # Total elements in Hankel matrix
        self.signal_length = signal_length  # Length of output time signal
        self.latent_dim = latent_dim
        
        # Encoder: Hankel matrix → Compressed manifold
        self.encoder = nn.Sequential(
            nn.Linear(hankel_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: Compressed manifold → Direct time-domain signal
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, signal_length)  # Direct to time-domain signal
        )
        
    def forward(self, hankel_matrix):
        # Flatten Hankel matrix
        hankel_flat = hankel_matrix.view(hankel_matrix.size(0), -1)
        
        # Encode: Hankel → Compressed manifold
        latent = self.encoder(hankel_flat)
        
        # Decode: Compressed manifold → Direct time signal
        time_signal = self.decoder(latent)
        
        return time_signal, latent
    
    def get_latent_representation(self, hankel_matrix):
        """Extract only the latent representation"""
        hankel_flat = hankel_matrix.view(hankel_matrix.size(0), -1)
        return self.encoder(hankel_flat)
    
    def generate_from_latent(self, latent):
        """Generate time signal directly from latent manifold"""
        return self.decoder(latent)


class AdvancedManifoldAutoencoder(nn.Module):
    """Advanced version with convolutional encoder and 1D decoder"""
    
    def __init__(self, hankel_shape, signal_length, latent_dim=32):
        super().__init__()
        
        self.hankel_shape = hankel_shape  # (n_windows, 1, window_len)
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        
        n_windows, channels, window_len = hankel_shape
        
        # Encoder: Process Hankel matrix structure
        self.conv_encoder = nn.Sequential(
            # Process each window
            nn.Conv1d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        
        # Calculate size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, window_len)
            dummy_conv = self.conv_encoder(dummy_input)
            conv_size = dummy_conv.numel() // dummy_conv.size(0)
        
        # Flatten and compress to manifold
        self.manifold_encoder = nn.Sequential(
            nn.Linear(conv_size * n_windows, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder: Manifold → Direct time signal
        self.signal_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, signal_length)
        )
        
    def forward(self, hankel_matrix):
        batch_size = hankel_matrix.size(0)
        n_windows = hankel_matrix.size(1)
        
        # Process each window through conv encoder
        processed_windows = []
        for i in range(n_windows):
            window = hankel_matrix[:, i, :].unsqueeze(1)  # (batch, 1, window_len)
            conv_out = self.conv_encoder(window)  # (batch, 64, reduced_len)
            processed_windows.append(conv_out)
        
        # Concatenate all processed windows
        all_windows = torch.cat(processed_windows, dim=2)  # (batch, 64, total_len)
        
        # Flatten and encode to manifold
        flattened = all_windows.view(batch_size, -1)
        latent = self.manifold_encoder(flattened)
        
        # Decode manifold to direct time signal
        time_signal = self.signal_decoder(latent)
        
        return time_signal, latent
    
    def get_latent_representation(self, hankel_matrix):
        """Extract only the latent representation"""
        batch_size = hankel_matrix.size(0)
        n_windows = hankel_matrix.size(1)
        
        # Process windows
        processed_windows = []
        for i in range(n_windows):
            window = hankel_matrix[:, i, :].unsqueeze(1)
            conv_out = self.conv_encoder(window)
            processed_windows.append(conv_out)
        
        all_windows = torch.cat(processed_windows, dim=2)
        flattened = all_windows.view(batch_size, -1)
        return self.manifold_encoder(flattened)
    
    def generate_from_latent(self, latent):
        """Generate time signal directly from latent manifold"""
        return self.signal_decoder(latent)
