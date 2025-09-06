"""
Training Functions for Direct Manifold-to-Signal Autoencoder
"""

import torch
import torch.nn as nn
import numpy as np


def train_direct_manifold_model(model, dataloader, target_signal, n_epochs=30, lr=1e-3, device='cpu'):
    """Train the direct manifold-to-signal autoencoder"""
    model.to(device)
    
    # Use AdamW optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use L1 loss for better edge preservation
    loss_fn = nn.L1Loss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )

    # Convert target signal to tensor
    target_tensor = torch.from_numpy(target_signal).float().to(device)
    if len(target_tensor.shape) == 1:
        target_tensor = target_tensor.unsqueeze(0)  # Add batch dimension to match model output

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for hankel_matrix in dataloader:
            hankel_matrix = hankel_matrix.to(device)
            opt.zero_grad()
            
            # Forward pass: Hankel → Latent → Direct time signal
            reconstructed_signal, latent = model(hankel_matrix)
            
            # Loss: Compare reconstructed signal with original time signal
            loss = loss_fn(reconstructed_signal, target_tensor)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Print every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == n_epochs:
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {epoch:03d}/{n_epochs} - L1 Loss: {avg_loss:.6e} - LR: {current_lr:.2e}")
    
    return model


def reconstruct_direct_signal(model, dataloader, device='cpu'):
    """Reconstruct signal directly from manifold"""
    model.eval()
    
    with torch.no_grad():
        for hankel_matrix in dataloader:
            hankel_matrix = hankel_matrix.to(device)
            reconstructed_signal, latent = model(hankel_matrix)
            reconstructed_signal = reconstructed_signal.cpu().numpy()
            latent = latent.cpu().numpy()
            break  # Only one sample
    
    return reconstructed_signal, latent


def generate_from_latent(model, latent_vector, device='cpu'):
    """Generate time signal from a specific latent vector"""
    model.eval()
    
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latent_vector).float().to(device)
        if len(latent_tensor.shape) == 1:
            latent_tensor = latent_tensor.unsqueeze(0)  # Add batch dimension
        
        if hasattr(model, 'generate_from_latent'):
            generated_signal = model.generate_from_latent(latent_tensor)
        else:
            # Fallback: use decoder directly
            generated_signal = model.decoder(latent_tensor)
        
        generated_signal = generated_signal.cpu().numpy()
    
    return generated_signal
