"""
Hankel Matrix Dataset for Attractor Reconstruction
Works with full Hankel matrices instead of individual windows
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def build_hankel_matrix(signal, window_len, stride=1):
    """Build full Hankel matrix from signal"""
    T = len(signal)
    starts = list(range(0, T - window_len + 1, stride))
    rows = np.stack([signal[s:s+window_len] for s in starts], axis=0)
    return rows, starts


class HankelMatrixDataset(Dataset):
    """Dataset for Hankel matrix-based attractor reconstruction"""
    
    def __init__(self, signal, window_len=1024, stride=1, normalize=True):
        self.signal = signal.astype(np.float32)
        self.window_len = window_len
        self.stride = stride
        
        # Build Hankel matrix
        self.hankel_matrix, self.starts = build_hankel_matrix(self.signal, window_len, stride)
        
        # Reshape for model input (batch, channels, height, width)
        # Height = number of windows, Width = window length
        self.inputs = self.hankel_matrix[:, None, :]  # (N_windows, 1, window_len)
        
        if normalize:
            self.mean = self.inputs.mean()
            self.std = self.inputs.std() + 1e-8
            self.inputs = (self.inputs - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return 1  # Single Hankel matrix

    def __getitem__(self, idx):
        # Return the entire Hankel matrix as a single sample
        return torch.from_numpy(self.inputs).float()


def reconstruct_from_hankel_matrix(pred_matrix, starts, T, window_fn=None):
    """Reconstruct full signal from predicted Hankel matrix"""
    recon_agg = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    
    N_windows, _, L = pred_matrix.shape

    if window_fn is None:
        window = np.ones(L, dtype=np.float32)
    else:
        window = window_fn(L).astype(np.float32)

    for i in range(N_windows):
        s = starts[i]
        win = pred_matrix[i, 0] * window
        recon_agg[s:s+L] += win
        counts[s:s+L] += window

    mask = counts > 0
    recon = np.zeros_like(recon_agg)
    recon[mask] = recon_agg[mask] / counts[mask]
    return recon
