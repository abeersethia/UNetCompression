"""
Signal-to-Signal Reconstruction using EXACT EDGeNet Architecture

Pipeline:
1. Input: Raw X signal (B, 1, T)
2. EDGeNet: Signal → Signal (B, 3, T) 
3. Output: Direct X, Y, Z signals

Uses the EDGeNet architecture from: https://github.com/dipayandewan94/EDGeNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.lorenz import generate_lorenz_full

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# EXACT EDGENET ARCHITECTURE FROM GITHUB
# ============================================================================

class PixelShuffle1D(torch.nn.Module):
    """1D pixel shuffler from original EDGeNet"""
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm1d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: self.pool = nn.MaxPool1d(kernel_size=2)
        else: self.pool = False

        block.append(nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm1d(out_c))

        block.append(nn.Conv1d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm1d(out_c))

        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_c)
        self.mish = nn.GELU()
        self.conv1 = nn.Conv1d(in_c, out_c // 4, 1)
        self.bn2 = nn.BatchNorm1d(out_c // 4)
        self.conv2 = nn.Conv1d(out_c // 4, out_c // 4, kernel_size=k_sz, padding='same')
        self.dropout = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(out_c // 4)
        self.conv5 = nn.Conv1d(out_c // 4, out_c, 1, 1, bias=False)
        self.conv6 = nn.Conv1d(in_c, out_c, 1, 1, padding='same', bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.mish(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.mish(out)
        out = self.conv5(out)
        
        residual = self.conv6(x)
        out += residual
        return out

class Trunk(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, residual=True, causal=False):
        super(Trunk, self).__init__()
        if residual:
            self.conv = nn.Sequential(
                ResBlock(in_c, out_c, k_sz=k_sz),
                ResBlock(out_c, out_c, k_sz=k_sz),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same'),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(out_c, out_c, kernel_size=k_sz, padding='same'),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
    
    def forward(self, x):
        out = self.conv(x)
        return out

class AttConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool='pool', attention=False, residual=True, causal=False, conv_type='normal'):
        super(AttConvBlock, self).__init__()
        self.conv_type = conv_type
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm1d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        if pool=='pool':
            self.pool = nn.MaxPool1d(kernel_size=2)
        elif pool=='conv':
            self.pool = nn.Conv1d(in_c, in_c, kernel_size=2, stride=2)
        else:
            self.pool = False

        self.conv = Trunk(in_c, out_c, k_sz=k_sz, residual=residual, causal=causal)
        
        if attention==True:
            self.mpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.softmax1_blocks = self._get_conv_layer(in_c, out_c, k_sz, dilation=6)
            self.skip1_connection_residual_block = self._get_conv_layer(out_c, out_c, k_sz)
            self.mpool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.softmax2_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=4)
            self.skip2_connection_residual_block = self._get_conv_layer(out_c, out_c, k_sz)
            self.mpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            self.softmax3_blocks = nn.Sequential(
                self._get_conv_layer(out_c, out_c, k_sz, dilation=2),
                self._get_conv_layer(out_c, out_c, k_sz, dilation=2)
            )
            self.interpolation3 = nn.Upsample(scale_factor=2)
            self.softmax4_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=4)
            self.interpolation2 = nn.Upsample(scale_factor=2)
            self.softmax5_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=6)
            self.interpolation1 = nn.Upsample(scale_factor=2)
            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                self._get_conv_layer(out_c, out_c, k_sz=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                self._get_conv_layer(out_c, out_c, k_sz=1),
                nn.Sigmoid()
            )
            self.last_blocks = self._get_conv_layer(out_c, out_c, k_sz)

    def _get_conv_layer(self, in_c, out_c, k_sz, dilation=1):
        if self.conv_type == 'depthwise':
            return nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same', dilation=dilation, groups=in_c)
        else:
            return nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same', dilation=dilation)

    def forward(self, x, attention=False):
        if self.pool: x = self.pool(x)
        out_trunk = self.conv(x)
        if attention==True:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            out_interp3 = self.interpolation3(out_softmax3)
            
            # Match sizes for addition
            if out_interp3.size(-1) != out_skip2_connection.size(-1):
                out_interp3 = F.interpolate(out_interp3, size=out_skip2_connection.size(-1), mode='linear', align_corners=False)
            out = torch.add(out_interp3, out_skip2_connection)
            
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4)
            
            # Match sizes for addition
            if out_interp2.size(-1) != out_skip1_connection.size(-1):
                out_interp2 = F.interpolate(out_interp2, size=out_skip1_connection.size(-1), mode='linear', align_corners=False)
            out = torch.add(out_interp2, out_skip1_connection)
            
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5)
            
            # Match sizes for multiplication
            if out_interp1.size(-1) != out_trunk.size(-1):
                out_interp1 = F.interpolate(out_interp1, size=out_trunk.size(-1), mode='linear', align_corners=False)
            out_softmax6 = self.softmax6_blocks(out_interp1)
            
            out = torch.multiply((1 + out_softmax6), out_trunk)
            out = self.last_blocks(out)
        else:
            out = out_trunk
        if self.shortcut: return out + self.shortcut(x)
        else: return out

class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv', upsample=True):
        super(UpsampleBlock, self).__init__()
        block = []
        if upsample:
            if up_mode == 'transp_conv':
                block.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2))
            elif up_mode == 'up_conv':
                block.append(nn.Upsample(scale_factor=2))
                block.append(nn.Conv1d(in_c, out_c, kernel_size=1))
            elif up_mode == 'pixelshuffle':
                block.append(nn.Conv1d(in_c, 2*out_c, kernel_size=1))
                block.append(PixelShuffle1D(2))
            else:
                raise Exception('Upsampling mode not supported')
        else:
            block.append(nn.Conv1d(in_c, out_c, kernel_size=1))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class DoubleAttBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=True, pool='pool', attention=True, residual=True, causal=False, conv_encoder='normal'):
        super(DoubleAttBlock, self).__init__()
        
        self.attention = attention
        self.block1 = AttConvBlock(in_c, in_c, k_sz=k_sz,
                              shortcut=shortcut, pool=False, attention=False, residual=residual, causal=causal, conv_type=conv_encoder)
        self.block2 = AttConvBlock(in_c, out_c, k_sz=k_sz,
                              shortcut=shortcut, pool=pool, attention=self.attention, residual=residual, causal=causal, conv_type=conv_encoder)

    def forward(self, x):
        out = self.block1(x, attention=False)
        out = self.block2(out, attention=self.attention)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        self.block = ResBlock(channels, channels)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False, 
                 skip_conn=True, residual=True, causal=False, upsample=True):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge
        self.skip_conn = skip_conn
        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode, upsample=upsample)
        self.conv_layer1 = AttConvBlock(out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal)
        self.conv_layer2 = AttConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip=None):
        up = self.up_layer(x)
        up = self.conv_layer1(up, attention=True)
        if skip is not None and self.skip_conn:
            # Match sizes if needed
            if up.size(-1) != skip.size(-1):
                up = F.interpolate(up, size=skip.size(-1), mode='linear', align_corners=False)
            
            if self.conv_bridge:
                skip = self.conv_bridge_layer(skip)
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1) 
            else:
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1)
            out = self.conv_layer2(out, attention=True)
        else:
            out = up
        return out

class EDGeNetEncoder(nn.Module):
    def __init__(
        self,
        in_c: int,
        k_sz: int,
        layers: List[int],
        shortcut: bool = True,
        pool='pool',
        residual=True,
        causal=False,
        conv_encoder='normal',
        downsample: List[bool] = None
    ):
        super().__init__()
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False)
        
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            if i <= 7:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool if downsample[i] else False, attention=True, residual=residual, causal=causal, conv_encoder=conv_encoder)
            else:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool if downsample[i] else False, attention=False, residual=residual, causal=causal, conv_encoder=conv_encoder)
            self.down_path.append(block)
        
    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        return x, down_activations

class EDGeNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        k_sz: int,
        layers: List[int],
        upsample: List[bool],
        up_mode='up_conv',
        conv_bridge: bool = True,
        shortcut: bool = True,
        skip_conn: bool = True,
        residual=True, 
        causal=False,
    ):
        super().__init__()
        
        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut, skip_conn=skip_conn, residual=residual, causal=causal, upsample=upsample[i])
            self.up_path.append(block)
            
        self.final = nn.Conv1d(layers[0], n_classes, kernel_size=1)

    def forward(self, x, down_activations):
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)

class EDGeNet(nn.Module):
    def __init__(self, in_c, n_classes, layers, downsample, k_sz=3, up_mode='pixelshuffle', pool='pool', 
                 conv_bridge=True, shortcut=True, skip_conn=True, residual_enc=False, residual_dec=True, 
                 causal_enc=False, causal_dec=False, conv_encoder='normal'):
        super(EDGeNet, self).__init__()
        upsample = list(reversed(downsample))
        self.encoder = EDGeNetEncoder(in_c, k_sz, layers, pool=pool, residual=residual_enc,
                                      causal=causal_enc, conv_encoder=conv_encoder, downsample=downsample)
        self.decoder = EDGeNetDecoder(n_classes, k_sz, layers, upsample, up_mode, conv_bridge, shortcut, skip_conn, 
                                      residual=residual_dec, causal=causal_dec)

    def forward(self, x):
        x, down_activations = self.encoder(x)
        x = self.decoder(x, down_activations)
        return x

# ============================================================================
# HANKEL-TO-SIGNAL RECONSTRUCTOR (Using Base Class)
# ============================================================================

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.architectures.direct_manifold_base import DirectManifoldBaseReconstructor

class DirectManifoldEDGeNetReconstructor(DirectManifoldBaseReconstructor):
    """Hankel-to-Direct Signal Reconstructor using exact EDGeNet"""
    
    def __init__(self, window_len=512, delay_embedding_dim=10, stride=5, 
                 compressed_t=256, train_split=0.7):
        super().__init__(window_len, delay_embedding_dim, stride, compressed_t, train_split)
    
    def create_model(self):
       
        # Input: Hankel (B, 10, 512)
        # Output: Direct signals (B, 3, 512)
        
        model = EDGeNet(
            in_c=self.delay_embedding_dim,  # Input: 10 channels (Hankel)
            n_classes=3,  
            layers=[8, 10, 12],  # Optimal configuration based on D analysis
            downsample=[True, True],  # Two downsampling stages
            k_sz=3,
            up_mode='pixelshuffle',  
            pool='pool',
            conv_bridge=False,  # Use bridge blocks for skip connections
            shortcut=True,  # Use shortcut connections
            skip_conn=False,  # Use skip connections
            residual_enc=True,  # Use ResBlocks in encoder
            residual_dec=True,  # Use residual connections in decoder
            causal_enc=False,
            causal_dec=False,
            conv_encoder='normal'  # Use normal convolutions (not depthwise)
        )
        
        # Store the full model
        self.model = model
        
        # For compatibility with base class, wrap model as encoder/decoder
        class EDGeNetAsEncoder(nn.Module):
            def __init__(self, edgenet_model):
                super().__init__()
                self.edgenet = edgenet_model
            def forward(self, x):
                # Just return the full output (we'll use it as "latent")
                return self.edgenet(x)
        
        class EDGeNetAsDecoder(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                # Already processed by "encoder", just pass through
                return x
        
        self.encoder = EDGeNetAsEncoder(model)
        self.decoder = EDGeNetAsDecoder()
        
        return self.encoder, self.decoder
    
    def train(self, max_epochs=150, base_noise_std=0.1, patience=25, verbose=True):
        """Train the model"""
        print(f"\n=== TRAINING ACTUAL EDGENET (HANKEL→DIRECT) ===")
        print(f"Architecture: EXACT EDGeNet from github.com/dipayandewan94/EDGeNet")
        print(f"Input: Hankel (B, {self.delay_embedding_dim}, {self.window_len})")
        print(f"Output: Direct Signals (B, 3, {self.window_len})")
        
        self.create_model()
        return super().train(max_epochs, base_noise_std, patience, verbose)

def main():
    print("="*60)
    print("EDGENET: HANKEL→DIRECT RECONSTRUCTION")
    print("="*60)
    
    # Generate Lorenz attractor
    traj, t = generate_lorenz_full(T=20.0, dt=0.01)
    
    # Create reconstructor
    reconstructor = DirectManifoldEDGeNetReconstructor(
        window_len=512,
        delay_embedding_dim=10,
        stride=5,
        compressed_t=256,
        train_split=0.7
    )
    
    # Prepare data and train
    reconstructor.prepare_data(traj, t)
    training_history = reconstructor.train(max_epochs=250, verbose=True)
    
    # Reconstruct
    original_attractor, reconstructed_attractor, metrics = reconstructor.reconstruct_manifold()
    
    print(f"\n=== ACTUAL EDGENET SUMMARY ===")
    print(f"✅ Architecture: ACTUAL EDGeNet (Hankel→Direct)")
    print(f"📚 Source: github.com/dipayandewan94/EDGeNet")
    print(f"📊 Parameters: {training_history['total_parameters']:,}")
    print(f"⏱️  Training Time: {training_history['training_time']:.2f}s")
    print(f"🔗 Correlations: X={metrics['correlations']['X']:.4f}, Y={metrics['correlations']['Y']:.4f}, Z={metrics['correlations']['Z']:.4f}")
    print(f"📈 Mean error: {metrics['mean_error']:.4f}")
    
    return reconstructor, original_attractor, reconstructed_attractor, metrics

if __name__ == "__main__":
    reconstructor, original_attractor, reconstructed_attractor, metrics = main()
