"""
Training pipeline for cognitive action probes
"""

from gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from probe_models import LinearProbe, MultiHeadProbe, CalibratedProbe, save_probe
from dataset_utils import get_idx_to_action_mapping


class ActivationDataset(Dataset):
    """Dataset for loading pre-computed activations"""

    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset

        Args:
            activations: Tensor of shape (num_examples, hidden_dim)
            labels: Tensor of shape (num_examples,)
        """
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


def load_activations_from_hdf5(
    file_path: Path
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load activations from HDF5 file

    Args:
        file_path: Path to HDF5 file

    Returns:
        Tuple of (train_acts, train_labels, val_acts, val_labels, test_acts, test_labels)
    """
    with h5py.File(file_path, 'r') as f:
        # Load activations (saved as float32 for HDF5 compatibility)
        # Convert back to bfloat16 to match original model dtype
        train_acts = torch.from_numpy(f['train']['activations'][:]).bfloat16()
        train_labels = torch.from_numpy(f['train']['labels'][:])

        val_acts = torch.from_numpy(f['val']['activations'][:]).bfloat16()
        val_labels = torch.from_numpy(f['val']['labels'][:])

        test_acts = torch.from_numpy(f['test']['activations'][:]).bfloat16()
        test_labels = torch.from_numpy(f['test']['labels'][:])

    return train_acts, train_labels, val_acts, val_labels, test_acts, test_labels


class ProbeTrainer:
    """Trainer for cognitive action probes"""

    def __init__(
        self,
        probe: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer

        Args:
            probe: Probe model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
        """
        self.probe = probe.to(device)
        self.device = device

        self.optimizer = optim.AdamW(
            self.probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        show_progress: bool = True
    ) -> float:
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training data
            show_progress: Whether to show progress bar

        Returns:
            Average training loss
        """
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(train_loader, desc="Training") if show_progress else train_loader

        for batch_acts, batch_labels in iterator:
            batch_acts = batch_acts.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            logits = self.probe(batch_acts)
            loss = self.criterion(logits, batch_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if show_progress:
                iterator.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(
        self,
        val_loader: DataLoader,
        show_progress: bool = False
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on validation/test set

        Args:
            val_loader: DataLoader for validation data
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (loss, accuracy, predictions, true_labels)
        """
        self.probe.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        iterator = tqdm(val_loader, desc="Evaluating") if show_progress else val_loader

        with torch.no_grad():
            for batch_acts, batch_labels in iterator:
                batch_acts = batch_acts.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.probe(batch_acts)
                loss = self.criterion(logits, batch_labels)

                preds = torch.argmax(logits, dim=-1)

                total_loss += loss.item()
                num_batches += 1

                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())

        avg_loss = total_loss / num_batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy, all_preds, all_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        save_best: bool = True
    ) -> Dict:
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model based on validation accuracy

        Returns:
            Training history dictionary
        """
        best_val_acc = 0.0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val Acc:    {val_acc:.4f}")

            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_dir:
                    save_path = save_dir / "best_probe.pth"
                    save_probe(
                        self.probe,
                        save_path,
                        metadata={
                            'epoch': epoch + 1,
                            'val_accuracy': val_acc,
                            'val_loss': val_loss
                        }
                    )
                    print(f"✓ Saved best model (val_acc: {val_acc:.4f})")

        return history


def compute_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    idx_to_action: Dict[int, str]
) -> Dict:
    """
    Compute comprehensive evaluation metrics

    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        idx_to_action: Mapping from indices to action names

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )

    # Average metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='micro', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'per_class': {
            idx_to_action[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(precision))
        },
        'confusion_matrix': cm.tolist()
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train cognitive action probes")
    parser.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 file with activations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/probes",
        help="Directory to save trained probes"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["linear", "multihead"],
        default="linear",
        help="Type of probe model"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for MultiHeadProbe"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )

    args = parser.parse_args()

    # Handle "auto" device by converting to cuda or cpu
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")

    # Load activations
    print("Loading activations...")
    train_acts, train_labels, val_acts, val_labels, test_acts, test_labels = \
        load_activations_from_hdf5(Path(args.activations))

    print(f"Train: {train_acts.shape}, Val: {val_acts.shape}, Test: {test_acts.shape}")

    # Get input dim and num classes
    input_dim = train_acts.shape[1]
    num_classes = len(torch.unique(train_labels))

    print(f"Input dim: {input_dim}, Num classes: {num_classes}")

    # Create datasets and dataloaders
    train_dataset = ActivationDataset(train_acts, train_labels)
    val_dataset = ActivationDataset(val_acts, val_labels)
    test_dataset = ActivationDataset(test_acts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create probe model
    if args.model_type == "linear":
        probe = LinearProbe(input_dim, num_classes)
    else:
        probe = MultiHeadProbe(input_dim, num_classes, hidden_dim=args.hidden_dim)

    print(f"\nProbe architecture: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in probe.parameters()):,}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("\nStarting training...")
    trainer = ProbeTrainer(
        probe,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )

    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_dir=output_dir,
        save_best=True
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    test_loss, test_acc, test_preds, test_labels_np = trainer.evaluate(
        test_loader,
        show_progress=True
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Compute detailed metrics
    idx_to_action = get_idx_to_action_mapping()
    metrics = compute_metrics(test_labels_np, test_preds, idx_to_action)

    print(f"\nMacro F1: {metrics['f1_macro']:.4f}")
    print(f"Micro F1: {metrics['f1_micro']:.4f}")

    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save final model
    final_path = output_dir / "final_probe.pth"
    save_probe(
        probe,
        final_path,
        metadata={
            'test_accuracy': test_acc,
            'test_f1_macro': metrics['f1_macro'],
            'model_type': args.model_type
        }
    )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: {output_dir / 'best_probe.pth'}")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
