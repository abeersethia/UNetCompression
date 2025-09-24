"""
X-Only Manifold Reconstruction using LSTM Autoencoder

This file implements X-only manifold reconstruction with an 
LSTM-based autoencoder architecture for temporal sequence modeling.

Based on: x_only_manifold_reconstruction_corrected.py
Architecture: LSTM Autoencoder
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_lstm_flops(model, sequence_length):
    """Estimate FLOPs for LSTM model"""
    total_flops = 0
    for layer in model.modules():
        if isinstance(layer, nn.LSTM):
            input_size = layer.input_size
            hidden_size = layer.hidden_size
            num_layers = layer.num_layers
            # LSTM has 4 gates, each with input and hidden connections
            flops_per_timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size) * 2
            total_flops += flops_per_timestep * sequence_length * num_layers
        elif isinstance(layer, nn.Linear):
            total_flops += layer.in_features * layer.out_features * 2
    return total_flops

class LSTMEncoder(nn.Module):
    """LSTM Encoder for sequence modeling"""
    def __init__(self, input_d, input_l, latent_d, latent_l):
        super().__init__()
        self.input_d = input_d
        self.input_l = input_l
        self.latent_d = latent_d
        self.latent_l = latent_l
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_d, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True, dropout=0.2)
        
        # Projection layer to latent space
        self.projection = nn.Linear(32, latent_d)
        
        # Temporal compression
        self.temporal_pool = nn.AdaptiveAvgPool1d(latent_l)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (B, D, L) -> (B, L, D) for LSTM
        x = x.transpose(1, 2)  # (B, L, D)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        
        # Project to latent dimensions
        x = self.projection(x)  # (B, L, latent_d)
        
        # Transpose for temporal pooling
        x = x.transpose(1, 2)  # (B, latent_d, L)
        
        # Temporal compression
        x = self.temporal_pool(x)  # (B, latent_d, latent_l)
        
        return x

class LSTMDecoder(nn.Module):
    """LSTM Decoder for sequence reconstruction"""
    def __init__(self, latent_d, latent_l, output_d, output_l):
        super().__init__()
        self.latent_d = latent_d
        self.latent_l = latent_l
        self.output_d = output_d
        self.output_l = output_l
        
        # Temporal expansion
        self.temporal_expand = nn.Upsample(size=output_l, mode='linear', align_corners=False)
        
        # Projection layer from latent space
        self.projection = nn.Linear(latent_d, 32)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(32, 64, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(128, output_d, batch_first=True, dropout=0.3)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (B, latent_d, latent_l)
        
        # Temporal expansion
        x = self.temporal_expand(x)  # (B, latent_d, output_l)
        
        # Transpose for LSTM
        x = x.transpose(1, 2)  # (B, output_l, latent_d)
        
        # Project from latent space
        x = self.projection(x)  # (B, output_l, 32)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)  # (B, output_l, output_d)
        
        # Transpose back to (B, D, L) format
        x = x.transpose(1, 2)  # (B, output_d, output_l)
        
        return x

class XOnlyManifoldReconstructorLSTM:
    """X-Only Manifold Reconstructor using LSTM architecture"""
    
    def __init__(self, window_len=512, delay_embedding_dim=10, stride=5, 
                 latent_d=32, latent_l=128, train_split=0.7):
        self.window_len = window_len
        self.delay_embedding_dim = delay_embedding_dim
        self.stride = stride
        self.latent_d = latent_d
        self.latent_l = latent_l
        self.train_split = train_split
        
        self.encoder = None
        self.decoder = None
        self.dataset_x = None
        self.dataset_y = None
        self.dataset_z = None
        
    def create_lstm_autoencoder(self):
        """Create LSTM autoencoder architecture"""
        encoder = LSTMEncoder(
            input_d=self.delay_embedding_dim,
            input_l=self.window_len,
            latent_d=self.latent_d,
            latent_l=self.latent_l
        )
        
        decoder = LSTMDecoder(
            latent_d=self.latent_d,
            latent_l=self.latent_l,
            output_d=3 * self.delay_embedding_dim,
            output_l=self.window_len
        )
        
        return encoder, decoder
    
    def prepare_data(self, traj, t):
        """Prepare data for training"""
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        
        print(f"Preparing data...")
        print(f"  Original signal length: {len(x)}")
        print(f"  Original attractor shape: {traj.shape}")
        
        # Create datasets for all components
        self.dataset_x = Hankel3DDataset(
            signal=x, window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride, normalize=True, shuffle=False
        )
        
        self.dataset_y = Hankel3DDataset(
            signal=y, window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride, normalize=True, shuffle=False
        )
        
        self.dataset_z = Hankel3DDataset(
            signal=z, window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride, normalize=True, shuffle=False
        )
        
        # Get Hankel matrices
        hankel_x = self.dataset_x.hankel_matrix
        hankel_y = self.dataset_y.hankel_matrix
        hankel_z = self.dataset_z.hankel_matrix
        
        n_batches, delay_dim, window_len = hankel_x.shape
        
        print(f"  Hankel matrix shape: {hankel_x.shape}")
        print(f"  Latent shape will be: ({n_batches}, {self.latent_d}, {self.latent_l})")
        
        # Create target data (X + Y + Z)
        self.target_data = np.concatenate([hankel_x, hankel_y, hankel_z], axis=1)
        self.input_data = hankel_x
        
        print(f"  Input data shape: {self.input_data.shape}")
        print(f"  Target data shape: {self.target_data.shape}")
        
        # Train/test split
        n_train_batches = int(n_batches * self.train_split)
        self.train_indices = np.arange(n_train_batches)
        self.test_indices = np.arange(n_train_batches, n_batches)
        
        self.train_input = self.input_data[self.train_indices]
        self.test_input = self.input_data[self.test_indices]
        self.train_target = self.target_data[self.train_indices]
        self.test_target = self.target_data[self.test_indices]
        
        print(f"  Train batches: {len(self.train_indices)}")
        print(f"  Test batches: {len(self.test_indices)}")
        
        return n_batches
    
    def get_reversed_adaptive_noise_std(self, epoch, max_epochs, base_std=0.1):
        """Get reversed adaptive noise standard deviation"""
        return base_std * (epoch / max_epochs) + 0.01
    
    def train(self, max_epochs=150, base_noise_std=0.1, patience=25, verbose=True):
        """Train the LSTM autoencoder"""
        if self.input_data is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        
        print(f"\n=== TRAINING LSTM AUTOENCODER ===")
        print(f"Architecture: Long Short-Term Memory")
        print(f"Input: X component only")
        print(f"Output: Full attractor (X, Y, Z)")
        print(f"Latent shape: (B, {self.latent_d}, {self.latent_l})")
        
        # Create autoencoder
        self.encoder, self.decoder = self.create_lstm_autoencoder()
        
        # Print model statistics
        total_params = count_parameters(self.encoder) + count_parameters(self.decoder)
        total_flops = estimate_lstm_flops(self.encoder, self.window_len) + estimate_lstm_flops(self.decoder, self.window_len)
        
        print(f"Model Statistics:")
        print(f"  Encoder parameters: {count_parameters(self.encoder):,}")
        print(f"  Decoder parameters: {count_parameters(self.decoder):,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Estimated FLOPs: {total_flops:,}")
        
        # Convert to tensors
        train_tensor_input = torch.from_numpy(self.train_input).float()
        test_tensor_input = torch.from_numpy(self.test_input).float()
        train_tensor_target = torch.from_numpy(self.train_target).float()
        test_tensor_target = torch.from_numpy(self.test_target).float()
        
        # Setup training
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=1e-3, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = torch.nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        noise_levels = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            noise_std = self.get_reversed_adaptive_noise_std(epoch, max_epochs, base_noise_std)
            noise_levels.append(noise_std)
            
            # Training
            self.encoder.train()
            self.decoder.train()
            optimizer.zero_grad()
            
            noisy_input = train_tensor_input + torch.randn_like(train_tensor_input) * noise_std
            train_latent = self.encoder(noisy_input)
            train_reconstructed = self.decoder(train_latent)
            train_loss = criterion(train_reconstructed, train_tensor_target)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            
            # Validation
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                val_latent = self.encoder(test_tensor_input)
                val_reconstructed = self.decoder(val_latent)
                val_loss = criterion(val_reconstructed, test_tensor_target)
            
            scheduler.step(val_loss)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_encoder_state = self.encoder.state_dict().copy()
                best_decoder_state = self.decoder.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
            
            if verbose and (epoch + 1) % 30 == 0:
                print(f'Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss.item():.6f}, '
                      f'Val Loss: {val_loss.item():.6f}, Noise: {noise_std:.4f}')
        
        # Load best model
        self.encoder.load_state_dict(best_encoder_state)
        self.decoder.load_state_dict(best_decoder_state)
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'noise_levels': noise_levels,
            'best_val_loss': best_val_loss,
            'total_parameters': total_params,
            'estimated_flops': total_flops
        }
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def reconstruct_manifold(self):
        """Reconstruct the full manifold from X component only"""
        if self.encoder is None or self.decoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        print(f"\n=== RECONSTRUCTING MANIFOLD (LSTM) ===")
        
        # Get reconstructed data for all data
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            
            all_input = torch.from_numpy(self.input_data).float()
            all_reconstructed = self.decoder(self.encoder(all_input)).numpy()
        
        print(f"Reconstructed data shape: {all_reconstructed.shape}")
        
        # Split reconstructed data back into X, Y, Z components
        n_batches = all_reconstructed.shape[0]
        recon_x = all_reconstructed[:, :self.delay_embedding_dim, :]
        recon_y = all_reconstructed[:, self.delay_embedding_dim:2*self.delay_embedding_dim, :]
        recon_z = all_reconstructed[:, 2*self.delay_embedding_dim:3*self.delay_embedding_dim, :]
        
        # Reconstruct signals from Hankel matrices
        recon_signal_x = reconstruct_from_3d_hankel(
            recon_x, self.dataset_x.batch_starts, self.dataset_x.stride, 
            len(self.dataset_x.signal), mean=self.dataset_x.mean, std=self.dataset_x.std
        )
        
        recon_signal_y = reconstruct_from_3d_hankel(
            recon_y, self.dataset_y.batch_starts, self.dataset_y.stride, 
            len(self.dataset_y.signal), mean=self.dataset_y.mean, std=self.dataset_y.std
        )
        
        recon_signal_z = reconstruct_from_3d_hankel(
            recon_z, self.dataset_z.batch_starts, self.dataset_z.stride, 
            len(self.dataset_z.signal), mean=self.dataset_z.mean, std=self.dataset_z.std
        )
        
        # Ensure all signals have the same length
        min_length = min(len(recon_signal_x), len(recon_signal_y), len(recon_signal_z))
        
        # Get original signals
        x_orig = self.dataset_x.signal[:min_length]
        y_orig = self.dataset_y.signal[:min_length]
        z_orig = self.dataset_z.signal[:min_length]
        
        # Truncate reconstructed signals
        recon_x_trunc = recon_signal_x[:min_length]
        recon_y_trunc = recon_signal_y[:min_length]
        recon_z_trunc = recon_signal_z[:min_length]
        
        # Create attractors
        original_attractor = np.column_stack([x_orig, y_orig, z_orig])
        reconstructed_attractor = np.column_stack([recon_x_trunc, recon_y_trunc, recon_z_trunc])
        
        # Calculate metrics
        corr_x = np.corrcoef(x_orig, recon_x_trunc)[0, 1]
        corr_y = np.corrcoef(y_orig, recon_y_trunc)[0, 1]
        corr_z = np.corrcoef(z_orig, recon_z_trunc)[0, 1]
        
        mse_x = mean_squared_error(x_orig, recon_x_trunc)
        mse_y = mean_squared_error(y_orig, recon_y_trunc)
        mse_z = mean_squared_error(z_orig, recon_z_trunc)
        
        error_3d = np.linalg.norm(original_attractor - reconstructed_attractor, axis=1)
        
        metrics = {
            'correlations': {'X': corr_x, 'Y': corr_y, 'Z': corr_z},
            'mse': {'X': mse_x, 'Y': mse_y, 'Z': mse_z},
            'error_3d': error_3d,
            'mean_error': np.mean(error_3d),
            'std_error': np.std(error_3d),
            'max_error': np.max(error_3d)
        }
        
        print(f"X correlation (input): {corr_x:.4f}")
        print(f"Y correlation (reconstructed): {corr_y:.4f}")
        print(f"Z correlation (reconstructed): {corr_z:.4f}")
        print(f"Mean reconstruction error: {np.mean(error_3d):.4f}")
        
        return original_attractor, reconstructed_attractor, metrics
    
    def visualize_results(self, original_attractor, reconstructed_attractor, metrics, save_path=None):
        """Create visualization of LSTM results"""
        print(f"\n=== CREATING LSTM VISUALIZATION ===")
        
        # Create time vector
        t = np.linspace(0, 20.0, len(original_attractor))
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original Lorenz Attractor (3D)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='blue', label='Original')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original Lorenz Attractor\n(3D)', fontsize=12)
        
        # 2. LSTM Reconstructed Manifold (3D)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='red', label='LSTM Reconstructed')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('LSTM Reconstructed Manifold\n(3D)', fontsize=12)
        
        # 3. Overlay Comparison
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
                 alpha=0.6, linewidth=1, color='blue', label='Original')
        ax3.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='red', label='LSTM Reconstructed')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('LSTM Overlay Comparison', fontsize=12)
        ax3.legend()
        
        # 4. Correlation Analysis
        ax4 = fig.add_subplot(2, 3, 4)
        components = ['X', 'Y', 'Z']
        correlations = [metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']]
        colors = ['blue', 'green', 'red']
        
        bars = ax4.bar(components, correlations, color=colors, alpha=0.7)
        ax4.set_ylabel('Correlation')
        ax4.set_title('LSTM Component Correlations', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Reconstruction Error
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, metrics['error_3d'], alpha=0.8, linewidth=1, color='red')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Reconstruction Error')
        ax5.set_title('LSTM Error Evolution', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f'''LSTM AUTOENCODER RESULTS:

ARCHITECTURE: Long Short-Term Memory
PARAMETERS: {self.training_history['total_parameters']:,}
FLOPS: {self.training_history['estimated_flops']:,}

CORRELATIONS:
X (input): {metrics['correlations']['X']:.4f}
Y (reconstructed): {metrics['correlations']['Y']:.4f}
Z (reconstructed): {metrics['correlations']['Z']:.4f}

ERRORS:
Mean Error: {metrics['mean_error']:.4f}
Std Error: {metrics['std_error']:.4f}
Max Error: {metrics['max_error']:.4f}

COMPRESSION:
Temporal: {self.window_len / self.latent_l:.1f}:1
Features: {self.delay_embedding_dim} → {self.latent_d}

LATENT SHAPE: ({self.input_data.shape[0]}, {self.latent_d}, {self.latent_l})

LSTM LAYERS: 3 layers (128→64→32)
TEMPORAL MODELING: Sequence processing'''
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LSTM visualization saved to: {save_path}")
        
        plt.show()
        
        return fig

def main():
    """Main function for LSTM autoencoder testing"""
    print("=== X-ONLY MANIFOLD RECONSTRUCTION (LSTM AUTOENCODER) ===")
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create LSTM reconstructor
    reconstructor = XOnlyManifoldReconstructorLSTM(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        latent_d=32,
        latent_l=128,
        train_split=0.7
    )
    
    # Prepare data and train
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=100, verbose=True)
    
    # Reconstruct manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Visualize results
    fig = reconstructor.visualize_results(
        original_attractor, reconstructed_attractor, metrics,
        save_path='lstm_manifold_reconstruction.png'
    )
    
    # Print final results
    print(f"\n=== LSTM AUTOENCODER SUMMARY ===")
    print(f"✅ Architecture: Long Short-Term Memory")
    print(f"📊 Parameters: {training_history['total_parameters']:,}")
    print(f"⚡ FLOPs: {training_history['estimated_flops']:,}")
    print(f"🔗 X correlation: {metrics['correlations']['X']:.4f}")
    print(f"🔗 Y correlation: {metrics['correlations']['Y']:.4f}")
    print(f"🔗 Z correlation: {metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics

if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()
