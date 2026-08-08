
"""
=============================================================
CRISPR-GRAM Configuration File - با Ablation Study پس‌آموزشی
=============================================================
Author : Hassan Salari
=============================================================
"""

import torch


class Config:

    # =====================================================
    # Dataset
    # =====================================================
    CLS_DIR = "/kaggle/input/datasets/hassansalari588/crisper/paper_data-classification"
    REG_DIR = "/kaggle/input/datasets/hassansalari588/crisper/paper_data-regression"

    DATA_TYPE = "cls"

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    NUM_WORKERS = 4
    PIN_MEMORY = True
    SEQUENCE_LENGTH = 23
    NUM_SGRNA_NODES = 20
    NUM_TARGET_NODES = 20
    NUM_GRAPH_NODES = 40
    INPUT_DIM = 126

    # =====================================================
    # Feature Projection
    # =====================================================
    EMBEDDING_DIM = 128
    DROPOUT = 0.12

    # =====================================================
    # Graph Attention Network
    # =====================================================
    NUM_GAT_LAYERS = 4
    NUM_HEADS = 8
    FFN_EXPANSION = 4
    GAT_DROPOUT = 0.12

    # =====================================================
    # Geometry-aware Manifold Learning
    # =====================================================
    MANIFOLD_DIM = 96
    PROJECTION_DIM = 64
    CONTRASTIVE_TEMPERATURE = 0.08
    MANIFOLD_WEIGHT = 0.15

    # =====================================================
    # Classification Head
    # =====================================================
    CLASSIFIER_HIDDEN = 128
    NUM_CLASSES = 1
    CLASSIFIER_DROPOUT = 0.3

    # =====================================================
    # Training
    # =====================================================
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 5e-5
    GRAD_CLIP = 1.0
    EARLY_STOPPING = 15

    # =====================================================
    # Learning Rate Scheduling
    # =====================================================
    LR_SCHEDULER = "cosine"
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 3

    # =====================================================
    # Hardware
    # =====================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True

    # =====================================================
    # Output
    # =====================================================
    OUTPUT_DIR = "results"
    ABLATION_OUTPUT_DIR = "ablation_results"
    CHECKPOINT_PATH = "results/best_model.pth"

    # =====================================================
    # Random Seed
    # =====================================================
    SEED = 42

    # =====================================================
    # Utility
    # =====================================================
    @classmethod
    def to_dict(cls):
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }
