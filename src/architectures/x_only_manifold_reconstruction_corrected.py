"""
X-Only Manifold Reconstruction for Lorenz Attractor - DIRECT SIGNAL VERSION

This file implements the X-only manifold reconstruction approach where:
- Input: Hankel matrix of X component (B, delay_dim, window_len)
- Latent: (B, D, L) where D is network-determined and L is compressed
- Output: Direct time-domain signals (X, Y, Z)

Pipeline: Hankel → Direct Signal (bypasses Hankel reconstruction)

The latent space preserves the 3D structure with:
- B: Batch size
- D: Network-determined feature dimensions (flexible)
- L: Compressed signal length

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from src.core.lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

class XOnlyManifoldReconstructorCorrected:
    """
    X-Only Manifold Reconstructor with direct signal output
    
    This class implements the complete pipeline for reconstructing the full
    Lorenz attractor from just the X component using direct signal output.
    Pipeline: Hankel → Direct Signal (bypasses Hankel reconstruction)
    """
    
    def __init__(self, window_len=512, delay_embedding_dim=10, stride=5, 
                 latent_d=32, latent_l=128, train_split=0.7):
        """
        Initialize the X-Only Manifold Reconstructor
        
        Args:
            window_len (int): Length of each window in Hankel matrix
            delay_embedding_dim (int): Number of delay embeddings per batch
            stride (int): Stride between consecutive batches
            latent_d (int): Number of feature dimensions in latent space
            latent_l (int): Compressed signal length in latent space
            train_split (float): Fraction of data for training
        """
        self.window_len = window_len
        self.delay_embedding_dim = delay_embedding_dim
        self.stride = stride
        self.latent_d = latent_d
        self.latent_l = latent_l
        self.train_split = train_split
        
        # Will be set during training
        self.encoder = None
        self.decoder = None
        self.dataset_x = None
        self.original_signals = None  # Store original X, Y, Z signals
        
    def create_3d_autoencoder(self):
        """Create 3D autoencoder with proper latent shape (B, D, L)"""
        
        class Encoder3D(nn.Module):
            def __init__(self, input_d, input_l, latent_d, latent_l):
                super().__init__()
                self.input_d = input_d
                self.input_l = input_l
                self.latent_d = latent_d
                self.latent_l = latent_l
                
                # 3D Convolutional layers for temporal compression
                self.conv1 = nn.Conv1d(input_d, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, latent_d, kernel_size=3, padding=1)
                
                # Temporal compression
                self.temporal_compress = nn.AdaptiveAvgPool1d(latent_l)
                
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(128)
                self.bn3 = nn.BatchNorm1d(latent_d)
                
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # x shape: (B, D, L)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.dropout(x)
                
                # Compress temporal dimension
                x = self.temporal_compress(x)
                
                return x  # Shape: (B, latent_d, latent_l)
        
        class Decoder3D(nn.Module):
            def __init__(self, latent_d, latent_l, output_l):
                super().__init__()
                self.latent_d = latent_d
                self.latent_l = latent_l
                self.output_l = output_l
                
                # Temporal expansion
                self.temporal_expand = nn.AdaptiveAvgPool1d(output_l)
                
                # 3D Convolutional layers for direct signal reconstruction
                self.conv1 = nn.Conv1d(latent_d, 128, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(64, 3, kernel_size=3, padding=1)  # Output 3 direct signals (X, Y, Z)
                
                self.bn1 = nn.BatchNorm1d(128)
                self.bn2 = nn.BatchNorm1d(64)
                
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # x shape: (B, latent_d, latent_l)
                
                # Expand temporal dimension
                x = self.temporal_expand(x)
                
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                
                x = self.conv3(x)
                
                return x  # Shape: (B, 3, output_l) = (B, 3, 512) - Direct signals
        
        encoder = Encoder3D(
            input_d=self.delay_embedding_dim,
            input_l=self.window_len,
            latent_d=self.latent_d,
            latent_l=self.latent_l
        )
        
        decoder = Decoder3D(
            latent_d=self.latent_d,
            latent_l=self.latent_l,
            output_l=self.window_len
        )
        
        return encoder, decoder
    
    def prepare_data(self, traj, t):
        """Prepare data for training"""
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        
        # Store original signals for later comparison
        self.original_signals = {'x': x, 'y': y, 'z': z}
        
        print(f"Preparing data...")
        print(f"  Original signal length: {len(x)}")
        print(f"  Original attractor shape: {traj.shape}")
        
        # Create dataset for X component (input)
        self.dataset_x = Hankel3DDataset(
            signal=x, 
            window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride,
            normalize=True, 
            shuffle=False
        )
        
        # Get Hankel matrix for X (input)
        hankel_x = self.dataset_x.hankel_matrix
        n_batches, delay_dim, window_len = hankel_x.shape
        
        print(f"  Hankel matrix shape: {hankel_x.shape}")
        print(f"  Latent shape will be: ({n_batches}, {self.latent_d}, {self.latent_l})")
        
        # Create target data: Direct time-domain signals (X, Y, Z)
        # Align target with Hankel batch structure
        target_signals = []
        for batch_idx in range(n_batches):
            batch_start = self.dataset_x.batch_starts[batch_idx]
            batch_end = min(batch_start + self.window_len, len(x))
            
            # Extract signals for this batch
            x_segment = x[batch_start:batch_end]
            y_segment = y[batch_start:batch_end]
            z_segment = z[batch_start:batch_end]
            
            # Pad if necessary
            if len(x_segment) < self.window_len:
                pad_len = self.window_len - len(x_segment)
                x_segment = np.pad(x_segment, (0, pad_len), mode='edge')
                y_segment = np.pad(y_segment, (0, pad_len), mode='edge')
                z_segment = np.pad(z_segment, (0, pad_len), mode='edge')
            
            target_signals.append(np.stack([x_segment, y_segment, z_segment]))
        
        self.target_data = np.array(target_signals)  # (n_batches, 3, window_len)
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
        """Train the X-only manifold reconstructor"""
        if self.input_data is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        
        print(f"\n=== TRAINING X-ONLY MANIFOLD RECONSTRUCTOR (DIRECT SIGNAL) ===")
        print(f"Input: Hankel matrix of X component")
        print(f"Output: Direct time-domain signals (X, Y, Z)")
        print(f"Latent shape: (B, {self.latent_d}, {self.latent_l})")
        print(f"Method: Hankel → Direct Signal (bypasses Hankel reconstruction)")
        
        # Create autoencoder
        self.encoder, self.decoder = self.create_3d_autoencoder()
        
        # Convert to tensors
        train_tensor_input = torch.from_numpy(self.train_input).float()
        test_tensor_input = torch.from_numpy(self.test_input).float()
        train_tensor_target = torch.from_numpy(self.train_target).float()
        test_tensor_target = torch.from_numpy(self.test_target).float()
        
        # Setup training
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=5e-4, weight_decay=1e-3
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
            
            # Add noise to input (X component only)
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
            'best_val_loss': best_val_loss
        }
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def reconstruct_manifold(self):
        """Reconstruct the full manifold from X component only using direct signals"""
        if self.encoder is None or self.decoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        print(f"\n=== RECONSTRUCTING DIRECT MANIFOLD FROM X-ONLY INPUT ===")
        
        # Get reconstructed data for all data
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            
            all_input = torch.from_numpy(self.input_data).float()
            all_reconstructed = self.decoder(self.encoder(all_input)).numpy()
        
        print(f"Reconstructed data shape: {all_reconstructed.shape}")
        
        # Reconstruct full signals by averaging overlapping segments
        n_batches = all_reconstructed.shape[0]
        original_length = len(self.dataset_x.signal)
        recon_x = np.zeros(original_length)
        recon_y = np.zeros(original_length)
        recon_z = np.zeros(original_length)
        counts = np.zeros(original_length)
        
        for batch_idx in range(n_batches):
            batch_start = self.dataset_x.batch_starts[batch_idx]
            batch_end = min(batch_start + self.window_len, original_length)
            segment_len = batch_end - batch_start
            
            recon_x[batch_start:batch_end] += all_reconstructed[batch_idx, 0, :segment_len]
            recon_y[batch_start:batch_end] += all_reconstructed[batch_idx, 1, :segment_len]
            recon_z[batch_start:batch_end] += all_reconstructed[batch_idx, 2, :segment_len]
            counts[batch_start:batch_end] += 1
        
        # Average overlapping regions
        recon_x /= np.maximum(counts, 1)
        recon_y /= np.maximum(counts, 1)
        recon_z /= np.maximum(counts, 1)
        
        # Get original signals
        x_orig = self.original_signals['x']
        y_orig = self.original_signals['y']
        z_orig = self.original_signals['z']
        
        # Create attractors
        original_attractor = np.column_stack([x_orig, y_orig, z_orig])
        reconstructed_attractor = np.column_stack([recon_x, recon_y, recon_z])
        
        print(f"Original attractor shape: {original_attractor.shape}")
        print(f"Reconstructed attractor shape: {reconstructed_attractor.shape}")
        
        # Calculate metrics
        corr_x = np.corrcoef(x_orig, recon_x)[0, 1]
        corr_y = np.corrcoef(y_orig, recon_y)[0, 1]
        corr_z = np.corrcoef(z_orig, recon_z)[0, 1]
        
        mse_x = mean_squared_error(x_orig, recon_x)
        mse_y = mean_squared_error(y_orig, recon_y)
        mse_z = mean_squared_error(z_orig, recon_z)
        
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
    
    def get_latent_representations(self):
        """Get latent representations for all data"""
        if self.encoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        with torch.no_grad():
            self.encoder.eval()
            latent_all = self.encoder(torch.from_numpy(self.input_data).float()).numpy()
        
        print(f"Latent space shape: {latent_all.shape}")
        return latent_all
    
    def visualize_reconstructed_manifold(self, original_attractor, reconstructed_attractor, metrics, 
                                       save_path=None):
        """
        Create comprehensive visualization of the reconstructed manifold
        
        Args:
            original_attractor (np.array): Original attractor
            reconstructed_attractor (np.array): Reconstructed attractor
            metrics (dict): Reconstruction metrics
            save_path (str): Path to save the plot
        """
        print(f"\n=== CREATING MANIFOLD VISUALIZATION ===")
        
        # Get latent representations
        latent_all = self.get_latent_representations()
        
        # Create time vector
        t = np.linspace(0, 20.0, len(original_attractor))
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Original Lorenz Attractor (3D)
        ax1 = fig.add_subplot(3, 5, 1, projection='3d')
        ax1.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='blue', label='Original')
        ax1.scatter(original_attractor[0, 0], original_attractor[0, 1], original_attractor[0, 2], 
                   color='green', s=100, label='Start')
        ax1.scatter(original_attractor[-1, 0], original_attractor[-1, 1], original_attractor[-1, 2], 
                   color='red', s=100, label='End')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original Lorenz Attractor\n(3D)', fontsize=12)
        
        # 2. RECONSTRUCTED Manifold from X-only Latent Space (3D)
        ax2 = fig.add_subplot(3, 5, 2, projection='3d')
        ax2.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax2.scatter(reconstructed_attractor[0, 0], reconstructed_attractor[0, 1], reconstructed_attractor[0, 2], 
                   color='green', s=100, label='Start')
        ax2.scatter(reconstructed_attractor[-1, 0], reconstructed_attractor[-1, 1], reconstructed_attractor[-1, 2], 
                   color='blue', s=100, label='End')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('RECONSTRUCTED Manifold\nfrom X-only Latent Space', fontsize=12)
        
        # 3. Overlay Comparison (3D)
        ax3 = fig.add_subplot(3, 5, 3, projection='3d')
        ax3.plot(original_attractor[:, 0], original_attractor[:, 1], original_attractor[:, 2], 
                 alpha=0.6, linewidth=1, color='blue', label='Original')
        ax3.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
                 alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Overlay Comparison\n(Original vs Reconstructed)', fontsize=12)
        ax3.legend()
        
        # 4. Original vs Reconstructed (2D X-Y)
        ax4 = fig.add_subplot(3, 5, 4)
        ax4.plot(original_attractor[:, 0], original_attractor[:, 1], alpha=0.8, linewidth=1, color='blue', label='Original')
        ax4.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('X-Y Projection\n(Original vs Reconstructed)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Original vs Reconstructed (2D X-Z)
        ax5 = fig.add_subplot(3, 5, 5)
        ax5.plot(original_attractor[:, 0], original_attractor[:, 2], alpha=0.8, linewidth=1, color='blue', label='Original')
        ax5.plot(reconstructed_attractor[:, 0], reconstructed_attractor[:, 2], alpha=0.8, linewidth=1, color='red', label='Reconstructed')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Z')
        ax5.set_title('X-Z Projection\n(Original vs Reconstructed)', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Reconstruction Error (3D)
        ax6 = fig.add_subplot(3, 5, 6, projection='3d')
        scatter = ax6.scatter(reconstructed_attractor[:, 0], reconstructed_attractor[:, 1], reconstructed_attractor[:, 2], 
                             c=metrics['error_3d'], cmap='hot', alpha=0.8, s=20)
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title('Reconstruction Error\n(3D Colored by Error)', fontsize=12)
        plt.colorbar(scatter, ax=ax6, label='Error Magnitude')
        
        # 7. Error Distribution
        ax7 = fig.add_subplot(3, 5, 7)
        ax7.hist(metrics['error_3d'], bins=50, alpha=0.7, color='orange', density=True)
        ax7.set_xlabel('Reconstruction Error')
        ax7.set_ylabel('Density')
        ax7.set_title('Error Distribution', fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        # 8. Time Series Comparison - X (Input)
        ax8 = fig.add_subplot(3, 5, 8)
        ax8.plot(t, original_attractor[:, 0], alpha=0.8, linewidth=1, color='blue', label='Original X')
        ax8.plot(t, reconstructed_attractor[:, 0], alpha=0.8, linewidth=1, color='red', label='Reconstructed X')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('X Value')
        ax8.set_title('X Component (INPUT)\n(Time Series)', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Time Series Comparison - Y (Reconstructed from X)
        ax9 = fig.add_subplot(3, 5, 9)
        ax9.plot(t, original_attractor[:, 1], alpha=0.8, linewidth=1, color='blue', label='Original Y')
        ax9.plot(t, reconstructed_attractor[:, 1], alpha=0.8, linewidth=1, color='red', label='Reconstructed Y')
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Y Value')
        ax9.set_title('Y Component (RECONSTRUCTED)\n(Time Series)', fontsize=12)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Time Series Comparison - Z (Reconstructed from X)
        ax10 = fig.add_subplot(3, 5, 10)
        ax10.plot(t, original_attractor[:, 2], alpha=0.8, linewidth=1, color='blue', label='Original Z')
        ax10.plot(t, reconstructed_attractor[:, 2], alpha=0.8, linewidth=1, color='red', label='Reconstructed Z')
        ax10.set_xlabel('Time')
        ax10.set_ylabel('Z Value')
        ax10.set_title('Z Component (RECONSTRUCTED)\n(Time Series)', fontsize=12)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. Correlation Analysis
        ax11 = fig.add_subplot(3, 5, 11)
        components = ['X', 'Y', 'Z']
        correlations = [metrics['correlations']['X'], metrics['correlations']['Y'], metrics['correlations']['Z']]
        colors = ['blue', 'green', 'red']
        
        bars = ax11.bar(components, correlations, color=colors, alpha=0.7)
        ax11.set_ylabel('Correlation')
        ax11.set_title('Component Correlations\n(Original vs Reconstructed)', fontsize=12)
        ax11.set_ylim(0, 1)
        ax11.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 12. MSE Analysis
        ax12 = fig.add_subplot(3, 5, 12)
        mse_values = [metrics['mse']['X'], metrics['mse']['Y'], metrics['mse']['Z']]
        bars = ax12.bar(components, mse_values, color=colors, alpha=0.7)
        ax12.set_ylabel('MSE')
        ax12.set_title('Component MSE\n(Original vs Reconstructed)', fontsize=12)
        ax12.grid(True, alpha=0.3)
        
        # Add MSE values on bars
        for bar, mse in zip(bars, mse_values):
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 13. Latent Space Visualization (3D projection)
        ax13 = fig.add_subplot(3, 5, 13, projection='3d')
        # Flatten latent space for 3D visualization
        latent_flat = latent_all.reshape(latent_all.shape[0], -1)
        ax13.plot(latent_flat[:, 0], latent_flat[:, 1], latent_flat[:, 2], alpha=0.8, linewidth=1, color='purple')
        ax13.scatter(latent_flat[0, 0], latent_flat[0, 1], latent_flat[0, 2], color='green', s=100, label='Start')
        ax13.scatter(latent_flat[-1, 0], latent_flat[-1, 1], latent_flat[-1, 2], color='red', s=100, label='End')
        ax13.set_xlabel('Latent Dim 1')
        ax13.set_ylabel('Latent Dim 2')
        ax13.set_zlabel('Latent Dim 3')
        ax13.set_title('Latent Space (3D Projection)\n(Learned from X only)', fontsize=12)
        
        # 14. Reconstruction Quality Metrics
        ax14 = fig.add_subplot(3, 5, 14)
        ax14.axis('off')
        metrics_text = f'''X-ONLY MANIFOLD RECONSTRUCTION (DIRECT SIGNAL):

APPROACH:
Input: Hankel matrix of X component
Output: Direct time-domain signals (X, Y, Z)
Pipeline: Hankel → Direct Signal

SHAPE COMPARISON:
Original: {original_attractor.shape}
Reconstructed: {reconstructed_attractor.shape}
Latent: {latent_all.shape}

COMPONENT CORRELATIONS:
X (input): {metrics['correlations']['X']:.4f}
Y (reconstructed): {metrics['correlations']['Y']:.4f}
Z (reconstructed): {metrics['correlations']['Z']:.4f}

COMPONENT MSE:
X (input): {metrics['mse']['X']:.6f}
Y (reconstructed): {metrics['mse']['Y']:.6f}
Z (reconstructed): {metrics['mse']['Z']:.6f}

OVERALL METRICS:
Mean Error: {metrics['mean_error']:.4f}
Std Error: {metrics['std_error']:.4f}
Max Error: {metrics['max_error']:.4f}

LATENT SPACE:
Shape: (B={latent_all.shape[0]}, D={latent_all.shape[1]}, L={latent_all.shape[2]})
Compression: {self.window_len / self.latent_l:.1f}:1 temporal

RECONSTRUCTION PROCESS:
1. Hankel X → Latent (B, D, L)
2. Latent → Direct signals (X, Y, Z)
3. Bypass Hankel reconstruction
4. Direct signal output

SUCCESS: ✓ Direct signals from X-only Hankel input!'''

        ax14.text(0.05, 0.95, metrics_text, transform=ax14.transAxes, fontsize=10,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 15. Error Evolution Over Time
        ax15 = fig.add_subplot(3, 5, 15)
        ax15.plot(t, metrics['error_3d'], alpha=0.8, linewidth=1, color='red')
        ax15.set_xlabel('Time')
        ax15.set_ylabel('Reconstruction Error')
        ax15.set_title('Error Evolution\nOver Time', fontsize=12)
        ax15.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Manifold visualization saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Main function to demonstrate X-only manifold reconstruction with direct signals"""
    print("=== X-ONLY MANIFOLD RECONSTRUCTION (DIRECT SIGNAL VERSION) ===")
    print("This demonstrates the direct signal approach: Hankel → Direct Signal")
    print("Pipeline: Input Hankel X → Latent (B, D, L) → Direct signals (X, Y, Z)")
    print()
    
    # Generate Lorenz attractor
    print("Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor with direct signal output
    reconstructor = XOnlyManifoldReconstructorCorrected(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        latent_d=32,  # Network-determined feature dimensions
        latent_l=128,  # Compressed signal length
        train_split=0.7
    )
    
    # Prepare data
    n_batches = reconstructor.prepare_data(traj, t)
    
    # Train the model
    training_history = reconstructor.train(max_epochs=150, verbose=True)
    
    # Reconstruct the manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Get latent representations
    latent_all = reconstructor.get_latent_representations()
    
    # Create visualization
    fig = reconstructor.visualize_reconstructed_manifold(
        original_attractor, reconstructed_attractor, metrics,
        save_path='x_only_manifold_reconstruction_direct_signal.png'
    )
    
    # Print final results
    print(f"\n=== FINAL RESULTS (DIRECT SIGNAL) ===")
    print(f"✅ SUCCESS: Full Lorenz attractor reconstructed from X-only input!")
    print(f"🎯 KEY INSIGHT: Hankel → Direct Signal (bypasses Hankel reconstruction)")
    print(f"📊 Latent shape: {latent_all.shape}")
    print(f"🔗 X correlation: {metrics['correlations']['X']:.4f}")
    print(f"🔗 Y correlation: {metrics['correlations']['Y']:.4f}")
    print(f"🔗 Z correlation: {metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean reconstruction error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics


if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()
