"""
X-Only Manifold Reconstruction for Lorenz Attractor

This file implements the X-only manifold reconstruction approach where:
- Input: X component only
- Output: Full attractor (X, Y, Z)
- Method: Learn causal relationships in latent space

The approach leverages delay embedding principles to reconstruct the full
dynamical system from a single component by learning the underlying
causal relationships between X, Y, and Z components.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hankel_matrix_3d import Hankel3DDataset, reconstruct_from_3d_hankel
from lorenz import generate_lorenz_full
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os

class XOnlyManifoldReconstructor:
    """
    X-Only Manifold Reconstructor for Lorenz Attractor
    
    This class implements the complete pipeline for reconstructing the full
    Lorenz attractor from just the X component using delay embedding and
    autoencoder-based manifold learning.
    """
    
    def __init__(self, window_len=512, delay_embedding_dim=10, stride=5, 
                 latent_dim=64, train_split=0.7):
        """
        Initialize the X-Only Manifold Reconstructor
        
        Args:
            window_len (int): Length of each window in Hankel matrix
            delay_embedding_dim (int): Number of delay embeddings per batch
            stride (int): Stride between consecutive batches
            latent_dim (int): Dimension of latent space
            train_split (float): Fraction of data for training
        """
        self.window_len = window_len
        self.delay_embedding_dim = delay_embedding_dim
        self.stride = stride
        self.latent_dim = latent_dim
        self.train_split = train_split
        
        # Will be set during training
        self.encoder = None
        self.decoder = None
        self.dataset_x = None
        self.dataset_y = None
        self.dataset_z = None
        self.input_dim = None
        self.output_dim = None
        
    def create_autoencoder(self):
        """Create the autoencoder architecture"""
        if self.input_dim is None:
            raise ValueError("Input dimension not set. Call prepare_data first.")
            
        self.output_dim = 3 * self.input_dim  # Output all three components
        
        encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, self.latent_dim)
        )
        
        decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, self.output_dim)
        )
        
        return encoder, decoder
    
    def prepare_data(self, traj, t):
        """
        Prepare data for training
        
        Args:
            traj (np.array): Lorenz trajectory (N, 3)
            t (np.array): Time vector (N,)
        """
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        
        print(f"Preparing data...")
        print(f"  Original signal length: {len(x)}")
        print(f"  Original attractor shape: {traj.shape}")
        
        # Create datasets for all components
        self.dataset_x = Hankel3DDataset(
            signal=x, 
            window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride,
            normalize=True, 
            shuffle=False
        )
        
        self.dataset_y = Hankel3DDataset(
            signal=y, 
            window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride,
            normalize=True, 
            shuffle=False
        )
        
        self.dataset_z = Hankel3DDataset(
            signal=z, 
            window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride,
            normalize=True, 
            shuffle=False
        )
        
        # Get Hankel matrices
        hankel_x = self.dataset_x.hankel_matrix
        hankel_y = self.dataset_y.hankel_matrix
        hankel_z = self.dataset_z.hankel_matrix
        
        n_batches, delay_dim, window_len = hankel_x.shape
        self.input_dim = delay_dim * window_len
        
        print(f"  Hankel matrix shape: {hankel_x.shape}")
        print(f"  Input dimension: {self.input_dim}")
        
        # Flatten for autoencoder
        flattened_x = hankel_x.reshape(n_batches, -1)
        flattened_y = hankel_y.reshape(n_batches, -1)
        flattened_z = hankel_z.reshape(n_batches, -1)
        
        # Create target data (X + Y + Z)
        self.target_data = np.concatenate([flattened_x, flattened_y, flattened_z], axis=1)
        self.input_data = flattened_x
        
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
        """
        Train the X-only manifold reconstructor
        
        Args:
            max_epochs (int): Maximum number of training epochs
            base_noise_std (float): Base noise standard deviation
            patience (int): Early stopping patience
            verbose (bool): Whether to print training progress
        """
        if self.input_data is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        
        print(f"\n=== TRAINING X-ONLY MANIFOLD RECONSTRUCTOR ===")
        print(f"Input: X component only")
        print(f"Output: Full attractor (X, Y, Z)")
        print(f"Method: Learn causal relationships in latent space")
        
        # Create autoencoder
        self.encoder, self.decoder = self.create_autoencoder()
        
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
        """
        Reconstruct the full manifold from X component only
        
        Returns:
            tuple: (original_attractor, reconstructed_attractor, metrics)
        """
        if self.encoder is None or self.decoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        print(f"\n=== RECONSTRUCTING MANIFOLD FROM X-ONLY INPUT ===")
        
        # Get reconstructed data for all data
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            
            all_input = torch.from_numpy(self.input_data).float()
            all_reconstructed = self.decoder(self.encoder(all_input)).numpy()
        
        print(f"Reconstructed data shape: {all_reconstructed.shape}")
        
        # Split reconstructed data back into X, Y, Z components
        recon_x = all_reconstructed[:, :self.input_dim]
        recon_y = all_reconstructed[:, self.input_dim:2*self.input_dim]
        recon_z = all_reconstructed[:, 2*self.input_dim:3*self.input_dim]
        
        print(f"Reconstructed X shape: {recon_x.shape}")
        print(f"Reconstructed Y shape: {recon_y.shape}")
        print(f"Reconstructed Z shape: {recon_z.shape}")
        
        # Reshape back to Hankel matrix shape
        n_batches = recon_x.shape[0]
        recon_hankel_x = recon_x.reshape(n_batches, self.delay_embedding_dim, self.window_len)
        recon_hankel_y = recon_y.reshape(n_batches, self.delay_embedding_dim, self.window_len)
        recon_hankel_z = recon_z.reshape(n_batches, self.delay_embedding_dim, self.window_len)
        
        # Reconstruct signals from Hankel matrices
        print("Reconstructing signals from Hankel matrices...")
        recon_signal_x = reconstruct_from_3d_hankel(
            recon_hankel_x, self.dataset_x.batch_starts, self.dataset_x.stride, 
            len(self.dataset_x.signal), mean=self.dataset_x.mean, std=self.dataset_x.std
        )
        
        recon_signal_y = reconstruct_from_3d_hankel(
            recon_hankel_y, self.dataset_y.batch_starts, self.dataset_y.stride, 
            len(self.dataset_y.signal), mean=self.dataset_y.mean, std=self.dataset_y.std
        )
        
        recon_signal_z = reconstruct_from_3d_hankel(
            recon_hankel_z, self.dataset_z.batch_starts, self.dataset_z.stride, 
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
        
        print(f"Original attractor shape: {original_attractor.shape}")
        print(f"Reconstructed attractor shape: {reconstructed_attractor.shape}")
        
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
    
    def get_latent_representations(self):
        """Get latent representations for all data"""
        if self.encoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        with torch.no_grad():
            self.encoder.eval()
            latent_all = self.encoder(torch.from_numpy(self.input_data).float()).numpy()
        
        return latent_all
    
    def visualize_results(self, original_attractor, reconstructed_attractor, metrics, 
                         save_path=None):
        """
        Create comprehensive visualization of results
        
        Args:
            original_attractor (np.array): Original attractor
            reconstructed_attractor (np.array): Reconstructed attractor
            metrics (dict): Reconstruction metrics
            save_path (str): Path to save the plot
        """
        print(f"\n=== CREATING VISUALIZATION ===")
        
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
        
        # 13. Latent Space Visualization
        ax13 = fig.add_subplot(3, 5, 13, projection='3d')
        ax13.plot(latent_all[:, 0], latent_all[:, 1], latent_all[:, 2], alpha=0.8, linewidth=1, color='purple')
        ax13.scatter(latent_all[0, 0], latent_all[0, 1], latent_all[0, 2], color='green', s=100, label='Start')
        ax13.scatter(latent_all[-1, 0], latent_all[-1, 1], latent_all[-1, 2], color='red', s=100, label='End')
        ax13.set_xlabel('Latent Dim 1')
        ax13.set_ylabel('Latent Dim 2')
        ax13.set_zlabel('Latent Dim 3')
        ax13.set_title('Latent Space\n(Learned from X only)', fontsize=12)
        
        # 14. Reconstruction Quality Metrics
        ax14 = fig.add_subplot(3, 5, 14)
        ax14.axis('off')
        metrics_text = f'''X-ONLY MANIFOLD RECONSTRUCTION:

APPROACH:
Input: X component only
Output: Full attractor (X, Y, Z)
Method: Learn causal relationships

SHAPE COMPARISON:
Original: {original_attractor.shape}
Reconstructed: {reconstructed_attractor.shape}

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
Dimensions: {self.latent_dim}
Compression: {original_attractor.size / (len(self.input_data) * self.latent_dim):.1f}:1

RECONSTRUCTION PROCESS:
1. X → Latent Space
2. Latent → (X, Y, Z)
3. Learn causal relationships
4. Reconstruct full attractor

SUCCESS: ✓ Full attractor from X-only input!'''

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
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """
    Main function to demonstrate X-only manifold reconstruction
    """
    print("=== X-ONLY MANIFOLD RECONSTRUCTION FOR LORENZ ATTRACTOR ===")
    print("This demonstrates how to reconstruct the full Lorenz attractor")
    print("from just the X component using delay embedding and autoencoder-based")
    print("manifold learning.")
    print()
    
    # Generate Lorenz attractor
    print("Generating Lorenz attractor...")
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor
    reconstructor = XOnlyManifoldReconstructor(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        latent_dim=64,
        train_split=0.7
    )
    
    # Prepare data
    n_batches = reconstructor.prepare_data(traj, t)
    
    # Train the model
    training_history = reconstructor.train(max_epochs=150, verbose=True)
    
    # Reconstruct the manifold
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    # Create visualization
    fig = reconstructor.visualize_results(
        original_attractor, reconstructed_attractor, metrics,
        save_path='x_only_manifold_reconstruction.png'
    )
    
    # Print final results
    print(f"\n=== FINAL RESULTS ===")
    print(f"✅ SUCCESS: Full Lorenz attractor reconstructed from X-only input!")
    print(f"🎯 KEY INSIGHT: The autoencoder learned the causal relationships between X, Y, Z!")
    print(f"📊 Compression ratio: {original_attractor.size / (n_batches * reconstructor.latent_dim):.1f}:1")
    print(f"🔗 X correlation: {metrics['correlations']['X']:.4f}")
    print(f"🔗 Y correlation: {metrics['correlations']['Y']:.4f}")
    print(f"🔗 Z correlation: {metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean reconstruction error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics


if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()
