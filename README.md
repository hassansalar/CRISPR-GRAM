# CRISPR-GRAM: Graph Attention Network with Geometry-aware Manifold Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

> **Note**: This repository contains the official implementation of CRISPR-GRAM, a deep learning framework for predicting CRISPR-Cas9 off-target activities using Graph Attention Networks and Geometry-aware Manifold Learning.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 📖 Overview

CRISPR-GRAM is a novel deep learning framework that combines **Graph Attention Networks (GAT)** with **Geometry-aware Manifold Learning** for accurate prediction of CRISPR-Cas9 off-target activities. Unlike conventional methods that rely solely on sequence features, CRISPR-GRAM explicitly models the interactions between guide RNA and target DNA sequences using graph-based reasoning, while simultaneously learning a biologically meaningful latent manifold that separates active and inactive off-targets.

### Key Innovations

- **Graph Attention Network**: Models the guide-target interaction as a graph with 40 nodes, enabling the model to capture complex spatial relationships
- **Geometry-aware Manifold Learning**: Learns a low-dimensional manifold that preserves the geometric structure of the data, improving feature separation and interpretability
- **Multi-head Attention Pooling**: Aggregates node-level features using attention mechanisms for better representation
- **Supervised Contrastive Learning**: Enhances class separability in the latent space, leading to better generalization

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **GAT-based Architecture** | 4-layer Graph Attention Network with 8 attention heads |
| **Manifold Learning** | Geometry-aware manifold encoder for interpretable representations |
| **Contrastive Learning** | Supervised contrastive loss with triplet margin for improved feature separation |
| **Attention Pooling** | Multi-head attention-based graph pooling for feature aggregation |
| **Mixed Precision Training** | Optimized training with automatic mixed precision (AMP) |
| **Ablation Study** | Comprehensive ablation analysis to evaluate component contributions |
| **Interpretability** | Attention visualization for understanding model predictions |

---

## 🏆 Performance

CRISPR-GRAM achieves state-of-the-art performance on the DeepCRISPR benchmark dataset:

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.75%** |
| **ROC-AUC** | **96.33%** |
| **PR-AUC** | **79.24%** |
| **F1-score** | **78.09%** |
| **Precision** | **79.82%** |
| **Recall** | **76.43%** |

### Comparison with State-of-the-Art

| Method | ROC-AUC (%) | PR-AUC (%) | Indel Support |
|--------|-------------|------------|---------------|
| CRISPR-Net | 99.1 | 32.3 | ✅ |
| DeepCRISPR | 98.1 | 49.7 | ❌ |
| Elevation-score | 97.9 | 16.3 | ❌ |
| **CRISPR-GRAM (Ours)** | **96.33** | **79.24** | ✅ |

---

## 📋 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CRISPR-GRAM.git
cd CRISPR-GRAM

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
