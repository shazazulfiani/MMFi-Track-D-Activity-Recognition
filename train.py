"""Pelatihan model 2D-CNN dan CNN-DANN untuk MM-Fi Track D.

Contoh penggunaan di Google Colab:
python train.py --features /content/drive/MyDrive/mmfi_trackD_features.npy \
  --metadata /content/drive/MyDrive/mmfi_trackD_metadata.csv \
  --normalization /content/drive/MyDrive/mmfi_trackD_normalization.npz \
  --output-dir /content/drive/MyDrive/MMFi_Track_D_Final
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from data_utils import (
    DEFAULT_FEATURES, DEFAULT_METADATA, DEFAULT_NORMALIZATION,
    load_experiment_data,
)
from models import CNN2D, CNNDANN, NUM_CLASSES


SEED = 42
DEFAULT_OUTPUT = "/content/drive/MyDrive/MMFi_Track_D_Final"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on_source_validation(model, loader, criterion, device, is_dann=False):
    model.eval()
    y_true, y_pred = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if is_dann:
                logits, _ = model(xb, grl_coefficient=0.0)
            else:
                logits = model(xb)

            loss = criterion(logits, yb)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * len(yb)
            total += len(yb)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc": accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def train_baseline(loaders, criterion, device, epochs, patience, learning_rate,
                   output_dir, seed):
    model = CNN2D().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in loaders["source_train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        val = evaluate_on_source_validation(
            model, loaders["source_val"], criterion, device
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            **val,
        }
        history.append(row)
        print(
            f"2D-CNN | Epoch {epoch + 1:02d}/{epochs} "
            f"| Train loss {row['train_loss']:.4f} "
            f"| Train acc {row['train_acc']:.4f} "
            f"| Val macro-F1 {row['val_macro_f1']:.4f}"
        )

        if val["val_macro_f1"] > best_f1:
            best_f1 = val["val_macro_f1"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            print("2D-CNN: early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training 2D-CNN tidak menghasilkan checkpoint terbaik.")
    model.load_state_dict(best_state)

    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "seed": seed,
        },
        checkpoint_dir / "mmfi_trackD_final_2dcnn.pt",
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "mmfi_trackD_final_2dcnn_history.csv", index=False)
    print(f"2D-CNN terbaik: epoch {best_epoch}, macro-F1 {best_f1:.4f}")
    return model, history_df


def train_dann(loaders, criterion, device, epochs, patience, learning_rate,
               output_dir, seed):
    model = CNNDANN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    domain_criterion = nn.CrossEntropyLoss()
    history = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    patience_count = 0
    total_steps = epochs * len(loaders["source_train"])

    for epoch in range(epochs):
        model.train()
        total_cls, total_dom = 0.0, 0.0
        correct, total = 0, 0
        target_iter = iter(loaders["target_adapt"])

        for step, (xs, ys) in enumerate(loaders["source_train"]):
            try:
                xt = next(target_iter)
            except StopIteration:
                target_iter = iter(loaders["target_adapt"])
                xt = next(target_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            progress = (
                epoch * len(loaders["source_train"]) + step
            ) / max(total_steps - 1, 1)
            grl_lambda = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

            optimizer.zero_grad()
            class_logits, source_domain_logits = model(
                xs, grl_coefficient=float(grl_lambda)
            )
            _, target_domain_logits = model(
                xt, grl_coefficient=float(grl_lambda)
            )

            class_loss = criterion(class_logits, ys)
            source_domain_labels = torch.zeros(
                len(xs), dtype=torch.long, device=device
            )
            target_domain_labels = torch.ones(
                len(xt), dtype=torch.long, device=device
            )
            domain_loss = 0.5 * (
                domain_criterion(source_domain_logits, source_domain_labels)
                + domain_criterion(target_domain_logits, target_domain_labels)
            )
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()

            total_cls += class_loss.item() * len(ys)
            total_dom += domain_loss.item() * len(ys)
            correct += (class_logits.argmax(1) == ys).sum().item()
            total += len(ys)

        val = evaluate_on_source_validation(
            model, loaders["source_val"], criterion, device, is_dann=True
        )
        row = {
            "epoch": epoch + 1,
            "train_class_loss": total_cls / max(total, 1),
            "train_domain_loss": total_dom / max(total, 1),
            "train_acc": correct / max(total, 1),
            **val,
        }
        history.append(row)
        print(
            f"CNN-DANN | Epoch {epoch + 1:02d}/{epochs} "
            f"| Cls loss {row['train_class_loss']:.4f} "
            f"| Domain loss {row['train_domain_loss']:.4f} "
            f"| Val macro-F1 {row['val_macro_f1']:.4f}"
        )

        if val["val_macro_f1"] > best_f1:
            best_f1 = val["val_macro_f1"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            print("CNN-DANN: early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training CNN-DANN tidak menghasilkan checkpoint terbaik.")
    model.load_state_dict(best_state)

    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "seed": seed,
        },
        checkpoint_dir / "mmfi_trackD_final_cnndann.pt",
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "mmfi_trackD_final_cnndann_history.csv", index=False)
    print(f"CNN-DANN terbaik: epoch {best_epoch}, macro-F1 {best_f1:.4f}")
    return model, history_df


def parse_args():
    parser = argparse.ArgumentParser(description="Training 2D-CNN dan CNN-DANN MM-Fi Track D.")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--normalization", default=DEFAULT_NORMALIZATION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(
        args.features, args.metadata, args.normalization,
        batch_size=args.batch_size,
    )
    meta = data["metadata"]
    indices = data["indices"]
    loaders = data["loaders"]
    device = data["device"]

    train_labels = meta.iloc[indices["source_train"]]["Label"].to_numpy()
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = len(train_labels) / (
        NUM_CLASSES * np.maximum(counts, 1)
    )
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n=== TRAINING BASELINE 2D-CNN ===")
    train_baseline(
        loaders, criterion, device, args.epochs, args.patience,
        args.learning_rate, output_dir, args.seed,
    )
    print("\n=== TRAINING CNN-DANN ===")
    train_dann(
        loaders, criterion, device, args.epochs, args.patience,
        args.learning_rate, output_dir, args.seed,
    )
    print("\nTraining selesai. Checkpoint dan riwayat tersimpan di:", output_dir)


if __name__ == "__main__":
    main()
