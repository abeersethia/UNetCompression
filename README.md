# Direct Manifold-to-Signal Attractor Reconstruction

This project implements a **Direct Manifold-to-Signal Autoencoder** for reconstructing Lorenz attractor dynamics from compressed latent representations.

## Architecture

- **Encoder**: Hankel matrix (762,368 elements) → Compressed manifold (32 dimensions)
- **Decoder**: Compressed manifold (32 dimensions) → Direct time-domain signal (2000 points)
- **Compression Ratio**: 381.2:1

## Key Features

✅ **Direct Learning**: Decoder maps from compressed latent space directly to time-domain signal  
✅ **Massive Compression**: 762,368 Hankel elements → 32 latent dimensions  
✅ **Attractor Reconstruction**: Learns meaningful latent manifold for dynamical system reconstruction  
✅ **No Skip Connections**: Forces network to learn compressed representation  

## Project Structure

```
UNetCompression/
├── main_direct_manifold.py      # Main execution script
├── direct_manifold_model.py     # DirectManifoldAutoencoder model
├── direct_manifold_training.py # Training functions
├── hankel_dataset.py           # HankelMatrixDataset class
├── lorenz.py                   # Lorenz system generation
├── generate_plots.py           # Plot generation script
├── analyze_artifacts.py        # Artifact analysis script
├── reconstruction_plots/       # Generated plots and visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the direct manifold reconstruction**:
   ```bash
   python main_direct_manifold.py
   ```

3. **Generate all plots and visualizations**:
   ```bash
   python generate_plots.py
   ```

4. **Analyze reconstruction artifacts**:
   ```bash
   python analyze_artifacts.py
   ```

## Results

- **Final L1 Loss**: 0.741
- **MSE**: 1.98e+00  
- **MAE**: 1.22e+00
- **SNR**: 15.36 dB
- **Correlation**: 99.68%
- **Training**: 100 epochs with adaptive learning rate

The model successfully learns to compress the Hankel representation into a meaningful 32D latent manifold and reconstruct the original time-domain signal directly from this compressed manifold, preserving the essential dynamics of the Lorenz attractor.