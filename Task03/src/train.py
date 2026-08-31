"""
train.py
--------
Trains the MusicLSTM / MusicLSTMAttention model on preprocessed note sequences.

Features:
  - Loads data produced by preprocessor.py
  - 80/20 train/validation split
  - CrossEntropy loss, Adam optimiser with cosine-annealing LR schedule
  - Gradient clipping to prevent exploding gradients
  - Early stopping with configurable patience
  - Best-model checkpointing (saves to models/saved/)
  - Loss curve saved as PNG

Usage:
    python src/train.py
    python src/train.py --epochs 100 --batch-size 128 --hidden 512
    python src/train.py --arch lstm --epochs 50 --no-cuda
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from model import build_model, model_summary, count_parameters
from preprocessor import load_processed, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = SRC_DIR.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Convert numpy arrays to DataLoaders with an 80/20 split.

    Returns: (train_loader, val_loader)
    """
    X_t = torch.from_numpy(X).long()
    y_t = torch.from_numpy(y).long()

    dataset = TensorDataset(X_t, y_t)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimiser:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    clip_grad:   float = 5.0,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()
        logits, _ = model(X_batch)
        loss      = criterion(logits, y_batch)
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple:
    """Evaluate on a DataLoader. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss       = criterion(logits, y_batch)
            total_loss += loss.item()

            preds    = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    mean_loss = total_loss / len(loader)
    accuracy  = correct / total_samples if total_samples > 0 else 0.0
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:      nn.Module,
    optimiser:  torch.optim.Optimizer,
    epoch:      int,
    val_loss:   float,
    config:     dict,
    path:       Path,
):
    torch.save(
        {
            "epoch":       epoch,
            "val_loss":    val_loss,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "config":      config,
        },
        str(path),
    )


def load_checkpoint(path: Path, model: nn.Module, optimiser=None):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimiser is not None and "optim_state" in ckpt:
        optimiser.load_state_dict(ckpt["optim_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Loss-curve plot
# ---------------------------------------------------------------------------

def plot_losses(
    train_losses: list,
    val_losses:   list,
    save_path:    Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping loss plot.")
        return

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss",      color="steelblue")
    ax.plot(epochs, val_losses,   label="Validation Loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[plot] Loss curve saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    epochs:        int   = 50,
    batch_size:    int   = 64,
    lr:            float = 0.001,
    hidden_size:   int   = 256,
    embed_dim:     int   = 128,
    num_layers:    int   = 2,
    dropout:       float = 0.3,
    architecture:  str   = "lstm_attention",
    clip_grad:     float = 5.0,
    patience:      int   = 10,
    val_split:     float = 0.2,
    use_cuda:      bool  = True,
    seed:          int   = 42,
    processed_dir: Path  = PROCESSED_DIR,
    models_dir:    Path  = MODELS_DIR,
    resume:        bool  = False,
) -> dict:
    """
    Full training run.

    Returns:
        dict with keys: best_val_loss, best_epoch, model_path,
                        train_losses, val_losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Device -----------------------------------------------------------
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"\n[train] Device : {device}")

    # ---- Load data --------------------------------------------------------
    print("[train] Loading preprocessed data …")
    data       = load_processed(processed_dir)
    X, y       = data["X"], data["y"]
    vocab_size = data["vocab_size"]
    seq_len    = data["seq_len"]

    print(f"[train] Samples   : {len(X):,}")
    print(f"[train] Vocab     : {vocab_size}")
    print(f"[train] Seq len   : {seq_len}")

    # ---- DataLoaders ------------------------------------------------------
    train_loader, val_loader = make_dataloaders(
        X, y, batch_size=batch_size, val_split=val_split, seed=seed
    )
    print(f"[train] Train batches : {len(train_loader)}")
    print(f"[train] Val   batches : {len(val_loader)}")

    # ---- Model ------------------------------------------------------------
    model_config = {
        "architecture":  architecture,
        "embed_dim":     embed_dim,
        "hidden_size":   hidden_size,
        "num_layers":    num_layers,
        "dropout":       dropout,
        "bidirectional": False,
        "vocab_size":    vocab_size,
        "seq_len":       seq_len,
    }

    model = build_model(vocab_size, model_config).to(device)
    model_summary(model, seq_len=seq_len)

    # ---- Optimiser & LR scheduler ----------------------------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore <PAD>

    # ---- Optionally resume from checkpoint --------------------------------
    start_epoch  = 0
    best_val_loss = float("inf")
    ckpt_path    = models_dir / "best_model.pt"

    if resume and ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(ckpt_path, model, optimiser)
        print(f"[train] Resumed from epoch {start_epoch}, val_loss={best_val_loss:.4f}")

    # ---- Training loop ----------------------------------------------------
    train_losses: list = []
    val_losses:   list = []
    no_improve       = 0
    best_epoch       = start_epoch

    print(f"\n[train] Starting training for {epochs} epoch(s) …\n")
    header = (
        f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  "
        f"{'Val Acc':>8}  {'LR':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimiser,
                                 criterion, device, clip_grad)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
            f"{val_acc:>7.2%}  {current_lr:>10.2e}  {elapsed:>6.1f}s"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            save_checkpoint(model, optimiser, epoch, val_loss,
                            model_config, ckpt_path)
            print(f"         ✓ New best  val_loss={best_val_loss:.4f}  saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"\n[train] Early stopping triggered — "
                    f"no improvement for {patience} epochs."
                )
                break

    # ---- Save final model & config ----------------------------------------
    final_path = models_dir / "final_model.pt"
    save_checkpoint(model, optimiser, epoch, val_loss, model_config, final_path)

    # Save model config as JSON for easy inspection
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # ---- Loss curve -------------------------------------------------------
    plot_losses(
        train_losses, val_losses,
        save_path=models_dir / "loss_curve.png",
    )

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Training complete")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Model saved to : {ckpt_path}")
    print(f"  Config saved   : {config_path}")
    print(f"{'='*50}\n")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "model_path":    str(ckpt_path),
        "train_losses":  train_losses,
        "val_losses":    val_losses,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the AI music generation model."
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--hidden",     type=int,   default=256,
                        help="LSTM hidden size (default: 256)")
    parser.add_argument("--embed-dim",  type=int,   default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument("--layers",     type=int,   default=2,
                        help="Number of LSTM layers (default: 2)")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--arch",       type=str,   default="lstm_attention",
                        choices=["lstm", "lstm_attention"],
                        help="Model architecture (default: lstm_attention)")
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping patience in epochs (default: 10)")
    parser.add_argument("--clip-grad",  type=float, default=5.0,
                        help="Gradient clipping norm (default: 5.0)")
    parser.add_argument("--no-cuda",    action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from best_model.pt if it exists")
    parser.add_argument("--seed",       type=int,   default=42)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        architecture=args.arch,
        clip_grad=args.clip_grad,
        patience=args.patience,
        use_cuda=not args.no_cuda,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
