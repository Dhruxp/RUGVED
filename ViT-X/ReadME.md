# Vision Transformer from Scratch

A complete implementation of a Vision Transformer (ViT) built from scratch in PyTorch without using any pretrained models or built-in transformer modules. This project focuses on understanding and implementing every component of the Vision Transformer architecture while incorporating several modern improvements over the original design.

## Features

* Custom patch embedding using Conv2D
* Learnable CLS token
* Learnable positional embeddings
* Multi-Head Self-Attention implemented from scratch
* Feed Forward Network (MLP) with GELU activation
* Pre-LayerNorm Transformer Encoder
* LayerScale residual scaling
* Stochastic Depth (DropPath)
* Attention map extraction for visualization
* AdamW optimizer with Cosine Annealing learning rate scheduling
* Mixed Precision (AMP) training
* Gradient clipping
* Model checkpointing and training history tracking

## Architecture

```text
Input Image
    │
    ▼
Patch Embedding
    │
    ▼
CLS Token + Position Embedding
    │
    ▼
Transformer Encoder × N
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── LayerScale
    ├── DropPath
    ├── Residual Connection
    ├── LayerNorm
    ├── MLP
    ├── LayerScale
    ├── DropPath
    └── Residual Connection
    │
    ▼
Final LayerNorm
    │
    ▼
Classification Head
```

## Dataset

* Intel Image Classification Dataset
* Classes:

  * Buildings
  * Forest
  * Glacier
  * Mountain
  * Sea
  * Street

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* tqdm

## Future Improvements

* Relative Positional Bias
* Attention Rollout
* Attention Heatmap Visualization
* MixUp and CutMix
* Label Smoothing
* t-SNE / UMAP Feature Visualization
* Ablation Studies

## Learning Objectives

This project was developed to gain a deep understanding of Vision Transformers by implementing every core component from first principles instead of relying on existing model implementations.
