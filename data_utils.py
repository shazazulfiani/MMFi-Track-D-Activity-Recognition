"""Utilitas data untuk proyek MM-Fi Track D.

Cache fitur, metadata, dan normalisasi harus berasal dari preprocessing yang sama.
Urutan baris metadata dipertahankan agar sesuai dengan indeks di cache fitur.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_FEATURES = "/content/drive/MyDrive/mmfi_trackD_features.npy"
DEFAULT_METADATA = "/content/drive/MyDrive/mmfi_trackD_metadata.csv"
DEFAULT_NORMALIZATION = "/content/drive/MyDrive/mmfi_trackD_normalization.npz"
SPLITS = ("source_train", "source_val", "target_adapt", "target_test")


def parse_valid_column(series: pd.Series) -> np.ndarray:
    """Ubah kolom valid menjadi mask boolean dengan aman."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes"})
        .to_numpy(dtype=bool)
    )


def check_cache_finite(cache, batch_size: int = 32) -> np.ndarray:
    """Periksa nilai NaN/Inf per sampel tanpa memuat seluruh cache ke RAM."""
    result = np.zeros(len(cache), dtype=bool)
    for start in range(0, len(cache), batch_size):
        end = min(start + batch_size, len(cache))
        batch = np.asarray(cache[start:end])
        result[start:end] = np.isfinite(batch).all(axis=(1, 2, 3))
    return result


class MMFiDataset(Dataset):
    """Dataset PyTorch dengan normalisasi per kanal."""

    def __init__(
        self,
        cache,
        metadata: pd.DataFrame,
        indices,
        mean: np.ndarray,
        std: np.ndarray,
        return_label: bool = True,
    ):
        self.cache = cache
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std
        self.return_label = return_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        x = np.array(self.cache[idx], dtype=np.float32, copy=True)
        x = (x - self.mean) / self.std

        if not np.isfinite(x).all():
            raise ValueError(f"Fitur non-finite pada indeks {idx}")

        x_tensor = torch.from_numpy(x)
        if not self.return_label:
            return x_tensor

        y = int(self.metadata.iloc[idx]["Label"])
        return x_tensor, torch.tensor(y, dtype=torch.long)


def load_experiment_data(
    features_path: str = DEFAULT_FEATURES,
    metadata_path: str = DEFAULT_METADATA,
    normalization_path: str = DEFAULT_NORMALIZATION,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Muat cache, validasi keselarasan, dan buat DataLoader semua split."""
    for label, path in [
        ("cache fitur", features_path),
        ("metadata", metadata_path),
        ("normalisasi", normalization_path),
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"File {label} tidak ditemukan: {path}\n"
                "Periksa path file atau berikan argumen path yang benar."
            )

    cache = np.load(features_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)

    with np.load(normalization_path) as norm:
        if "mean" not in norm.files or "std" not in norm.files:
            raise ValueError("File normalisasi harus berisi 'mean' dan 'std'.")
        mean = norm["mean"].astype(np.float32).reshape(36, 1, 1)
        std = norm["std"].astype(np.float32).reshape(36, 1, 1)

    if cache.ndim != 4 or tuple(cache.shape[1:]) != (36, 114, 10):
        raise ValueError(
            f"Shape cache tidak sesuai. Diharapkan (N, 36, 114, 10), "
            f"didapat {cache.shape}"
        )
    if len(cache) != len(metadata):
        raise ValueError(
            f"Jumlah cache ({len(cache)}) dan metadata ({len(metadata)}) berbeda."
        )
    if "valid" not in metadata.columns:
        raise ValueError("Metadata harus memiliki kolom 'valid'.")
    for column in ["Split", "Label", "Student", "Environment", "Action"]:
        if column not in metadata.columns:
            raise ValueError(f"Kolom metadata wajib tidak ditemukan: {column}")

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Mean/std mengandung nilai NaN atau Inf.")
    if (std <= 0).any():
        raise ValueError("Semua nilai std harus lebih besar dari nol.")

    metadata["Label"] = pd.to_numeric(metadata["Label"], errors="raise").astype(int)
    metadata_valid = parse_valid_column(metadata["valid"])
    cache_valid = check_cache_finite(cache)

    if not np.array_equal(metadata_valid, cache_valid):
        mismatch = np.flatnonzero(metadata_valid != cache_valid)
        raise ValueError(
            f"Validitas metadata dan cache tidak cocok pada {len(mismatch)} "
            f"baris. Contoh indeks: {mismatch[:10].tolist()}. "
            "Pastikan cache dan metadata berasal dari proses yang sama."
        )

    indices = {
        split: np.flatnonzero(
            metadata["Split"].eq(split).to_numpy() & metadata_valid
        )
        for split in SPLITS
    }
    for split in SPLITS:
        if len(indices[split]) == 0:
            raise ValueError(f"Tidak ada sampel valid untuk split '{split}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    loaders = {
        "source_train": DataLoader(
            MMFiDataset(cache, metadata, indices["source_train"], mean, std),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "source_val": DataLoader(
            MMFiDataset(cache, metadata, indices["source_val"], mean, std),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "target_adapt": DataLoader(
            MMFiDataset(
                cache, metadata, indices["target_adapt"], mean, std,
                return_label=False,
            ),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "target_test": DataLoader(
            MMFiDataset(cache, metadata, indices["target_test"], mean, std),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    print("=== DATA PROYEK SIAP ===")
    print("Device:", device)
    print("Shape cache:", cache.shape)
    print("Sampel valid:", int(metadata_valid.sum()))
    for split in SPLITS:
        print(f"{split}: {len(indices[split])} sampel")

    return {
        "cache": cache,
        "metadata": metadata,
        "valid_mask": metadata_valid,
        "indices": indices,
        "mean": mean,
        "std": std,
        "loaders": loaders,
        "device": device,
    }
