"""
Base class for Direct Manifold Reconstruction

Shared functionality for all direct manifold architectures.
"""

import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.hankel_matrix_3d import Hankel3DDataset

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DirectManifoldBaseReconstructor:
    """Base class for direct manifold reconstruction"""
    
    def __init__(self, window_len=512, delay_embedding_dim=10, stride=5, 
                 compressed_t=256, train_split=0.7):
        self.window_len = window_len
        self.delay_embedding_dim = delay_embedding_dim
        self.stride = stride
        self.compressed_t = compressed_t
        self.train_split = train_split
        
        self.encoder = None
        self.decoder = None
        self.dataset_x = None
        self.original_length = None
        self.original_signals = None
        
    def prepare_data(self, traj, t):
        """Prepare data for training"""
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        
        self.original_length = len(x)
        self.original_signals = {'x': x, 'y': y, 'z': z}
        
        print(f"Preparing data...")
        print(f"  Original signal length: {len(x)}")
        print(f"  Original attractor shape: {traj.shape}")
        
        # Create Hankel matrix for X component (input)
        self.dataset_x = Hankel3DDataset(
            signal=x, window_len=self.window_len, 
            delay_embedding_dim=self.delay_embedding_dim, 
            stride=self.stride, normalize=True, shuffle=False
        )
        
        hankel_x = self.dataset_x.hankel_matrix
        n_batches = hankel_x.shape[0]
        
        # Target: Direct time-domain signals (X, Y, Z)
        target_signals = []
        for batch_idx in range(n_batches):
            batch_start = self.dataset_x.batch_starts[batch_idx]
            batch_end = min(batch_start + self.window_len, len(x))
            
            x_segment = x[batch_start:batch_end]
            y_segment = y[batch_start:batch_end]
            z_segment = z[batch_start:batch_end]
            
            if len(x_segment) < self.window_len:
                pad_len = self.window_len - len(x_segment)
                x_segment = np.pad(x_segment, (0, pad_len), mode='edge')
                y_segment = np.pad(y_segment, (0, pad_len), mode='edge')
                z_segment = np.pad(z_segment, (0, pad_len), mode='edge')
            
            target_signals.append(np.stack([x_segment, y_segment, z_segment]))
        
        self.target_data = np.array(target_signals)
        self.input_data = hankel_x
        
        print(f"  Hankel matrix shape: {hankel_x.shape}")
        print(f"  Input data shape: {self.input_data.shape}")
        print(f"  Target data shape: {self.target_data.shape}")
        print(f"  Compressed manifold size: {self.compressed_t}")
        
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
    
    def train(self, max_epochs=150, base_noise_std=0.1, patience=25, verbose=True):
        """Train the model"""
        if self.input_data is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        
        total_params = count_parameters(self.encoder) + count_parameters(self.decoder)
        print(f"Model Statistics:")
        print(f"  Encoder parameters: {count_parameters(self.encoder):,}")
        print(f"  Decoder parameters: {count_parameters(self.decoder):,}")
        print(f"  Total parameters: {total_params:,}")
        
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
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            noise_std = base_noise_std * (epoch / max_epochs) + 0.01
            
            # Training
            self.encoder.train()
            self.decoder.train()
            optimizer.zero_grad()
            
            noisy_input = train_tensor_input + torch.randn_like(train_tensor_input) * noise_std
            compressed_manifold = self.encoder(noisy_input)
            train_output = self.decoder(compressed_manifold)
            train_loss = criterion(train_output, train_tensor_target)
            
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
                val_compressed = self.encoder(test_tensor_input)
                val_output = self.decoder(val_compressed)
                val_loss = criterion(val_output, test_tensor_target)
            
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
        
        training_time = time.time() - start_time
        
        # Load best model
        self.encoder.load_state_dict(best_encoder_state)
        self.decoder.load_state_dict(best_decoder_state)
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_parameters': total_params,
            'training_time': training_time
        }
        
        print(f"Training completed in {training_time:.2f}s. Best validation loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def reconstruct_manifold(self):
        """Reconstruct the full manifold directly"""
        if self.encoder is None or self.decoder is None:
            raise ValueError("Model not trained. Call train first.")
        
        print(f"\n=== RECONSTRUCTING DIRECT MANIFOLD ===")
        
        # Get reconstructed data
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            
            all_input = torch.from_numpy(self.input_data).float()
            compressed_manifold = self.encoder(all_input)
            all_reconstructed = self.decoder(compressed_manifold).numpy()
        
        print(f"Compressed manifold shape: {compressed_manifold.shape}")
        print(f"Reconstructed data shape: {all_reconstructed.shape}")
        
        # Reconstruct full signals by averaging overlapping segments
        n_batches = all_reconstructed.shape[0]
        recon_x = np.zeros(self.original_length)
        recon_y = np.zeros(self.original_length)
        recon_z = np.zeros(self.original_length)
        counts = np.zeros(self.original_length)
        
        for batch_idx in range(n_batches):
            batch_start = self.dataset_x.batch_starts[batch_idx]
            batch_end = min(batch_start + self.window_len, self.original_length)
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
        
        # Calculate metrics
        corr_x = np.corrcoef(x_orig, recon_x)[0, 1]
        corr_y = np.corrcoef(y_orig, recon_y)[0, 1]
        corr_z = np.corrcoef(z_orig, recon_z)[0, 1]
        
        error_3d = np.linalg.norm(original_attractor - reconstructed_attractor, axis=1)
        
        # Calculate normalized MSE
        mse_x_norm = np.mean((x_orig - recon_x) ** 2)
        mse_y_norm = np.mean((y_orig - recon_y) ** 2)
        mse_z_norm = np.mean((z_orig - recon_z) ** 2)
        
        # Denormalize for true MSE calculation
        # Get original signal statistics for denormalization
        x_orig_raw = self.original_signals['x']
        y_orig_raw = self.original_signals['y']
        z_orig_raw = self.original_signals['z']
        
        # Denormalize reconstructed signals
        recon_x_denorm = recon_x * np.std(x_orig_raw) + np.mean(x_orig_raw)
        recon_y_denorm = recon_y * np.std(y_orig_raw) + np.mean(y_orig_raw)
        recon_z_denorm = recon_z * np.std(z_orig_raw) + np.mean(z_orig_raw)
        
        # Calculate denormalized MSE
        mse_x_denorm = np.mean((x_orig_raw - recon_x_denorm) ** 2)
        mse_y_denorm = np.mean((y_orig_raw - recon_y_denorm) ** 2)
        mse_z_denorm = np.mean((z_orig_raw - recon_z_denorm) ** 2)
        
        metrics = {
            'correlations': {'X': corr_x, 'Y': corr_y, 'Z': corr_z},
            'error_3d': error_3d,
            'mean_error': np.mean(error_3d),
            'std_error': np.std(error_3d),
            'max_error': np.max(error_3d),
            'mse_normalized': {'X': mse_x_norm, 'Y': mse_y_norm, 'Z': mse_z_norm, 'Mean': np.mean([mse_x_norm, mse_y_norm, mse_z_norm])},
            'mse_denormalized': {'X': mse_x_denorm, 'Y': mse_y_denorm, 'Z': mse_z_denorm, 'Mean': np.mean([mse_x_denorm, mse_y_denorm, mse_z_denorm])}
        }
        
        print(f"X correlation: {corr_x:.4f}")
        print(f"Y correlation: {corr_y:.4f}")
        print(f"Z correlation: {corr_z:.4f}")
        print(f"Mean reconstruction error: {np.mean(error_3d):.4f}")
        print(f"Normalized MSE - X: {mse_x_norm:.6f}, Y: {mse_y_norm:.6f}, Z: {mse_z_norm:.6f}")
        print(f"Denormalized MSE - X: {mse_x_denorm:.6f}, Y: {mse_y_denorm:.6f}, Z: {mse_z_denorm:.6f}")
        
        return original_attractor, reconstructed_attractor, metrics

