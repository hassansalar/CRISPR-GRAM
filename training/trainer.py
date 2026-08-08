import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from config import Config


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        save_dir="results"
    ):
        self.model = model
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        self.scaler = amp.GradScaler(enabled=Config.MIXED_PRECISION)

        self.best_auc = 0
        self.best_loss = float('inf')
        self.current_epoch = 0

        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "accuracy": [],
            "auc": [],
            "lr": [],
            "manifold_loss": [],
            "classification_loss": []
        }

        os.makedirs(self.save_dir, exist_ok=True)

    ############################################################

    def train_one_epoch(self):

        self.model.train()
        running_loss = 0
        running_class_loss = 0
        running_manifold_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):

            x = batch["features"].to(self.device)
            y = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()

            with amp.autocast(enabled=Config.MIXED_PRECISION):
                outputs = self.model(x)

                loss_dict = self.criterion(
                    outputs["logits"],
                    outputs["projection"],
                    y
                )

                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                Config.GRAD_CLIP
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            running_class_loss += loss_dict["classification_loss"].item()
            running_manifold_loss += loss_dict.get("manifold_loss", 0).item()

        n_batches = len(self.train_loader)
        return {
            "loss": running_loss / n_batches,
            "classification": running_class_loss / n_batches,
            "manifold": running_manifold_loss / n_batches
        }

    ############################################################

    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()

        losses = []
        probs = []
        labels = []

        for batch in tqdm(loader, desc="Evaluating"):

            x = batch["features"].to(self.device)
            y = batch["label"].float().to(self.device)

            outputs = self.model(x)

            loss_dict = self.criterion(
                outputs["logits"],
                outputs["projection"],
                y
            )

            losses.append(loss_dict["total_loss"].item())

            probs.extend(outputs["probability"].squeeze(-1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        probs = np.array(probs)
        labels = np.array(labels)

        pred = (probs > 0.5).astype(int)

        metrics = {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(labels, pred),
            "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred, zero_division=0),
            "f1": f1_score(labels, pred, zero_division=0),
            "roc_auc": roc_auc_score(labels, probs),
            "pr_auc": average_precision_score(labels, probs)
        }

        return metrics, labels, probs

    ############################################################

    def fit(self):

        patience = 0

        for epoch in range(Config.NUM_EPOCHS):
            self.current_epoch = epoch

            if epoch < Config.WARMUP_EPOCHS:
                warmup_factor = (epoch + 1) / Config.WARMUP_EPOCHS
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = Config.LEARNING_RATE * warmup_factor

            train_metrics = self.train_one_epoch()
            
            metrics, labels, probs = self.evaluate(self.val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["valid_loss"].append(metrics["loss"])
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["auc"].append(metrics["roc_auc"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["manifold_loss"].append(train_metrics.get("manifold", 0))
            self.history["classification_loss"].append(train_metrics.get("classification", 0))

            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(metrics["roc_auc"])

            print(
                f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}"
                f"\n  Train Loss: {train_metrics['loss']:.4f}"
                f"  Classification: {train_metrics.get('classification', 0):.4f}"
                f"  Manifold: {train_metrics.get('manifold', 0):.4f}"
                f"\n  Val Loss: {metrics['loss']:.4f}"
                f"  Acc: {metrics['accuracy']:.4f}"
                f"  AUC: {metrics['roc_auc']:.4f}"
                f"  PR-AUC: {metrics['pr_auc']:.4f}"
                f"\n  LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if metrics["roc_auc"] > self.best_auc:
                self.best_auc = metrics["roc_auc"]
                
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_auc': self.best_auc,
                    },
                    os.path.join(self.save_dir, "best_model.pth")
                )

                self.best_labels = labels
                self.best_probs = probs
                patience = 0
                print("  ✓ New best model saved!")

            else:
                patience += 1

            if patience > Config.EARLY_STOPPING:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        self.plot_results()

    ############################################################

    def load_best_model(self):
        checkpoint = torch.load(
            os.path.join(self.save_dir, "best_model.pth"),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model (AUC: {checkpoint['best_auc']:.4f})")

    ############################################################

    @torch.no_grad()
    def test(self):

        metrics, labels, probs = self.evaluate(self.test_loader)

        self.test_labels = labels
        self.test_probabilities = probs
        self.test_predictions = (probs > 0.5).astype(int)

        self.plot_test_results()

        return metrics

    ############################################################

    def plot_results(self):

        epochs = np.arange(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train")
        axes[0, 0].plot(epochs, self.history["valid_loss"], label="Validation")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title("Training and Validation Loss")
        
        axes[0, 1].plot(epochs, self.history["accuracy"], label="Accuracy")
        axes[0, 1].plot(epochs, self.history["auc"], label="AUC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title("Accuracy and AUC")
        
        axes[1, 0].plot(epochs, self.history["lr"])
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        axes[1, 0].set_title("Learning Rate Schedule")
        
        if "manifold_loss" in self.history and self.history["manifold_loss"]:
            axes[1, 1].plot(epochs, self.history["classification_loss"], label="Classification")
            axes[1, 1].plot(epochs, self.history["manifold_loss"], label="Manifold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_title("Component Losses")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"), dpi=300)
        plt.close()

    ############################################################

    def plot_test_results(self):
        
        fpr, tpr, _ = roc_curve(self.test_labels, self.test_probabilities)
        roc_auc = roc_auc_score(self.test_labels, self.test_probabilities)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "test_roc_curve.png"), dpi=300)
        plt.close()

        p, r, _ = precision_recall_curve(self.test_labels, self.test_probabilities)
        ap = average_precision_score(self.test_labels, self.test_probabilities)

        plt.figure(figsize=(6, 6))
        plt.plot(r, p, label=f"AP = {ap:.4f}", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "test_pr_curve.png"), dpi=300)
        plt.close()

        cm = confusion_matrix(self.test_labels, self.test_predictions)
        
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "test_confusion_matrix.png"), dpi=300)
        plt.close()

        pd.DataFrame(self.history).to_csv(
            os.path.join(self.save_dir, "training_metrics.csv"),
            index=False
        )
