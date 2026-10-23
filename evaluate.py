"""Evaluasi akhir model MM-Fi Track D pada target_test (lingkungan E04).

Script ini memuat checkpoint yang sudah dilatih; tidak menjalankan training ulang.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_utils import (
    DEFAULT_FEATURES, DEFAULT_METADATA, DEFAULT_NORMALIZATION,
    load_experiment_data,
)
from models import CNN2D, CNNDANN, NUM_CLASSES


DEFAULT_OUTPUT = "/content/drive/MyDrive/MMFi_Track_D_Final"


def load_checkpoint(model, path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}. Jalankan train.py terlebih dahulu "
            "atau berikan path checkpoint yang benar."
        )
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_model(model, loader, device, is_dann=False):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if is_dann:
                logits, _ = model(xb, grl_coefficient=0.0)
            else:
                logits = model(xb)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluasi target test MM-Fi Track D.")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--normalization", default=DEFAULT_NORMALIZATION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--dann-checkpoint", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(
        args.features, args.metadata, args.normalization, batch_size=32
    )
    meta = data["metadata"]
    indices = data["indices"]
    device = data["device"]
    target_test_loader = data["loaders"]["target_test"]

    cnn_path = Path(args.cnn_checkpoint) if args.cnn_checkpoint else (
        checkpoints_dir / "mmfi_trackD_final_2dcnn.pt"
    )
    dann_path = Path(args.dann_checkpoint) if args.dann_checkpoint else (
        checkpoints_dir / "mmfi_trackD_final_cnndann.pt"
    )

    cnn = load_checkpoint(CNN2D(), cnn_path, device)
    dann = load_checkpoint(CNNDANN(), dann_path, device)

    y_true, pred_cnn = predict_model(cnn, target_test_loader, device)
    y_true_dann, pred_dann = predict_model(
        dann, target_test_loader, device, is_dann=True
    )
    if not np.array_equal(y_true, y_true_dann):
        raise RuntimeError("Urutan label target test berbeda antara kedua model.")

    labels = list(range(NUM_CLASSES))
    results = []
    for name, predictions in [("2D-CNN", pred_cnn), ("CNN-DANN", pred_dann)]:
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, predictions),
            "Macro-F1": f1_score(
                y_true, predictions, labels=labels,
                average="macro", zero_division=0,
            ),
            "Weighted-F1": f1_score(
                y_true, predictions, labels=labels,
                average="weighted", zero_division=0,
            ),
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "mmfi_trackD_final_test_results.csv", index=False)

    test_meta = meta.iloc[indices["target_test"]].copy()
    test_meta["TrueLabel"] = y_true
    test_meta["Prediction_2D_CNN"] = pred_cnn
    test_meta["Prediction_CNN_DANN"] = pred_dann
    test_meta.to_csv(
        results_dir / "mmfi_trackD_final_test_predictions.csv", index=False
    )

    report_cnn = classification_report(
        y_true, pred_cnn, labels=labels, output_dict=True, zero_division=0
    )
    report_dann = classification_report(
        y_true, pred_dann, labels=labels, output_dict=True, zero_division=0
    )
    class_names = [f"A{i:02d}" for i in range(1, NUM_CLASSES + 1)]
    per_class = pd.DataFrame({
        "Activity": class_names,
        "CNN_F1": [report_cnn[str(i)]["f1-score"] for i in labels],
        "DANN_F1": [report_dann[str(i)]["f1-score"] for i in labels],
        "CNN_Recall": [report_cnn[str(i)]["recall"] for i in labels],
        "DANN_Recall": [report_dann[str(i)]["recall"] for i in labels],
    })
    per_class["F1_Difference"] = per_class["DANN_F1"] - per_class["CNN_F1"]
    per_class.to_csv(results_dir / "mmfi_trackD_final_per_class.csv", index=False)

    cm_cnn = confusion_matrix(y_true, pred_cnn, labels=labels, normalize="true")
    cm_dann = confusion_matrix(y_true, pred_dann, labels=labels, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, cm, title in [
        (axes[0], cm_cnn, "Baseline 2D-CNN"),
        (axes[1], cm_dann, "CNN-DANN"),
    ]:
        image = ax.imshow(cm, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Prediksi aktivitas")
        ax.set_ylabel("Aktivitas sebenarnya")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Proporsi per kelas sebenarnya")
    fig.savefig(
        results_dir / "mmfi_trackD_final_confusion_matrix.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # Grafik training dibuat bila riwayat dari train.py tersedia.
    cnn_history_path = results_dir / "mmfi_trackD_final_2dcnn_history.csv"
    dann_history_path = results_dir / "mmfi_trackD_final_cnndann_history.csv"
    if cnn_history_path.is_file() and dann_history_path.is_file():
        h_cnn = pd.read_csv(cnn_history_path)
        h_dann = pd.read_csv(dann_history_path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(h_cnn["epoch"], h_cnn["train_loss"], label="2D-CNN train loss")
        axes[0].plot(
            h_dann["epoch"], h_dann["train_class_loss"],
            label="CNN-DANN classification loss",
        )
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            h_cnn["epoch"], h_cnn["val_macro_f1"], marker="o", label="2D-CNN"
        )
        axes[1].plot(
            h_dann["epoch"], h_dann["val_macro_f1"], marker="o", label="CNN-DANN"
        )
        axes[1].set_title("Macro-F1 Validasi Source")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            results_dir / "mmfi_trackD_final_training_curves.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
    else:
        print("Catatan: grafik training tidak dibuat karena file riwayat training belum tersedia.")

    print("\n=== FINAL TARGET TEST RESULTS ===")
    print(results_df.round(4).to_string(index=False))
    print("\n=== CLASSIFICATION REPORT: CNN-DANN ===")
    print(classification_report(y_true, pred_dann, labels=labels, zero_division=0))
    print("\nHasil disimpan di:", results_dir)


if __name__ == "__main__":
    main()
