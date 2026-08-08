"""
==========================================================
CRISPR-GRAM - Training Script
==========================================================
"""

import os
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import random_split, DataLoader

from config import Config
from dataset.crispr_dataset import CRISPRDataset
from models.crispr_gram import CRISPR_GRAM
from losses.manifold_loss import CRISPRLoss
from training.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders():

    assert os.path.exists(Config.CLS_DIR), \
        f"Classification dataset not found:\n{Config.CLS_DIR}"

    print("\n" + "="*60)
    print("Loading CRISPR Dataset ...")
    print("="*60)

    dataset = CRISPRDataset(
        data_dir=Config.CLS_DIR,
        data_type=Config.DATA_TYPE
    )

    total_size = len(dataset)
    train_size = int(total_size * Config.TRAIN_RATIO)
    val_size = int(total_size * Config.VAL_RATIO)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Total Samples : {len(dataset):,}")
    print(f"Train Samples : {len(train_set):,}")
    print(f"Validation    : {len(val_set):,}")
    print(f"Test          : {len(test_set):,}")
    print(f"Feature Dim   : {dataset.combined_features.shape[1]}")
    print("="*60 + "\n")

    return dataset, train_loader, val_loader, test_loader


def build_model():
    """Build model"""
    model = CRISPR_GRAM(
        input_dim=Config.INPUT_DIM,
        embed_dim=Config.EMBEDDING_DIM,
        num_nodes=Config.NUM_GRAPH_NODES,
        manifold_dim=Config.MANIFOLD_DIM,
        num_layers=Config.NUM_GAT_LAYERS,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    print(f"Embedding Dimension: {Config.EMBEDDING_DIM}")
    print(f"Number of Heads    : {Config.NUM_HEADS}")
    print(f"Head Dimension     : {Config.EMBEDDING_DIM // Config.NUM_HEADS}")
    print(f"Total Parameters   : {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*60 + "\n")

    return model.to(Config.DEVICE)


def build_optimizer(model):
    """Build optimizer with different learning rates"""
    
    graph_params = []
    manifold_params = []
    classifier_params = []
    embedding_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embedding_params.append(param)
        elif 'graph_encoder' in name:
            graph_params.append(param)
        elif 'manifold' in name or 'projection' in name:
            manifold_params.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
        else:
            manifold_params.append(param)
    
    param_groups = [
        {'params': embedding_params, 'lr': Config.LEARNING_RATE * 0.5},
        {'params': graph_params, 'lr': Config.LEARNING_RATE},
        {'params': manifold_params, 'lr': Config.LEARNING_RATE * 0.8},
        {'params': classifier_params, 'lr': Config.LEARNING_RATE * 0.5},
    ]
    
    param_groups = [g for g in param_groups if g['params']]
    
    return torch.optim.AdamW(
        param_groups,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )


def build_scheduler(optimizer):
    """Build Cosine Annealing scheduler"""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS - Config.WARMUP_EPOCHS,
        eta_min=Config.LR_MIN
    )
    return scheduler


def build_loss():
    """Build loss function"""
    return CRISPRLoss(
        manifold_weight=Config.MANIFOLD_WEIGHT,
        temperature=Config.CONTRASTIVE_TEMPERATURE,
        triplet_margin=0.5
    )


def main():
    
    print("="*70)
    print(" CRISPR-GRAM - Training")
    print(" Embedding Dimension = 128, Heads = 8")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Mixed Precision: {Config.MIXED_PRECISION}")
    print("="*70)

    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    dataset, train_loader, val_loader, test_loader = create_dataloaders()

    model = build_model()

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = build_loss()

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        save_dir=Config.OUTPUT_DIR
    )

    print("\n" + "="*70)
    print(" Starting Training...")
    print("="*70 + "\n")

    trainer.fit()

    print("\n" + "="*70)
    print(" Training Completed!")
    print("="*70 + "\n")

    print("Loading Best Model...")
    trainer.load_best_model()

    print("\nRunning Final Evaluation on Test Set...")
    results = trainer.test()

    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)
    for key, value in results.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*70)
    
    print(f"\n✓ Results saved to: {Config.OUTPUT_DIR}/")
    print(f"\n✓ To run Ablation Study, execute: python ablation.py")


if __name__ == "__main__":
    main()
