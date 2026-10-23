"""Preprocessing Wi-Fi CSI untuk proyek MM-Fi Track D.

Pipeline:
1. Membaca manifest sampel berisi Environment, Student, Action, Split,
   SegmentIndex, Start, dan End.
2. Memilih enam frame dari setiap segmen.
3. Memperbaiki nilai amplitude NaN/Inf dengan interpolasi antar-frame jika
   masih tersedia nilai valid.
4. Membersihkan fase: memeriksa nilai non-finite, melakukan phase unwrap,
   dan menghilangkan tren linear sepanjang subcarrier.
5. Mengurangi noise menggunakan Savitzky-Golay filter.
6. Menyimpan cache fitur, metadata, dan catatan error.

Catatan: file dataset MM-Fi dan manifest tidak disertakan di repository.
Sediakan akses ke keduanya sebelum menjalankan script ini.
"""

import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

EXPECTED_FRAME_SHAPE = (3, 114, 10)
SELECTED_FRAMES = 6
FEATURE_SHAPE = (36, 114, 10)
REQUIRED_COLUMNS = {
    "Environment", "Student", "Action", "Split", "SegmentIndex", "Start", "End"
}


def repair_amplitude(amplitude: np.ndarray) -> np.ndarray:
    """Interpolasi nilai amplitude non-finite sepanjang enam frame terpilih.

    amplitude memiliki shape (6, 3, 114, 10). Jika suatu fitur tidak memiliki
    satu pun nilai valid, sampel ditandai gagal agar tidak membuat nilai palsu.
    """
    flat = amplitude.reshape(amplitude.shape[0], -1).copy()

    for col in range(flat.shape[1]):
        valid = np.isfinite(flat[:, col])
        if valid.all():
            continue
        if not valid.any():
            raise ValueError(
                f"Amplitude tidak memiliki nilai valid pada fitur {col}"
            )

        missing_idx = np.flatnonzero(~valid)
        valid_idx = np.flatnonzero(valid)
        flat[missing_idx, col] = np.interp(
            missing_idx, valid_idx, flat[valid_idx, col]
        )

    return flat.reshape(amplitude.shape)


def sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """Validasi, unwrap, dan hilangkan tren linear fase pada sumbu subcarrier."""
    if not np.isfinite(phase).all():
        raise ValueError("Fase mengandung NaN/Inf; sampel tidak diproses")

    # Mengurangi diskontinuitas sudut fase di sekitar -pi dan pi.
    phase = np.unwrap(phase, axis=2)

    # Menghilangkan offset dan tren linear sepanjang 114 subcarrier.
    subcarrier = np.arange(phase.shape[2], dtype=np.float64)
    centered_sc = subcarrier - subcarrier.mean()
    centered_sc = centered_sc[None, None, :, None]
    phase_mean = phase.mean(axis=2, keepdims=True)
    denominator = (centered_sc ** 2).sum(axis=2, keepdims=True)
    slope = (
        (centered_sc * (phase - phase_mean)).sum(axis=2, keepdims=True)
        / denominator
    )
    phase = phase - phase_mean - slope * centered_sc

    if not np.isfinite(phase).all():
        raise ValueError("Fase menjadi non-finite setelah sanitasi")
    return phase


def reduce_noise(array: np.ndarray) -> np.ndarray:
    """Pengurangan noise ringan dengan filter Savitzky-Golay pada sumbu terakhir."""
    if array.shape[-1] < 5:
        raise ValueError("Dimensi terakhir terlalu pendek untuk filter Savitzky-Golay")
    return savgol_filter(
        array,
        window_length=5,
        polyorder=2,
        axis=-1,
        mode="interp",
    )


def preprocess_sample(zip_file: zipfile.ZipFile, row: pd.Series) -> np.ndarray:
    """Baca enam frame untuk satu segmen lalu hasilkan fitur (36, 114, 10)."""
    env = str(row["Environment"]).strip()
    student = str(row["Student"]).strip()
    action = str(row["Action"]).strip()
    start = int(row["Start"])
    end = int(row["End"])

    if start < 1 or end < start or end > 297:
        raise ValueError(f"Rentang frame tidak valid: {start}-{end}")

    frame_ids = np.linspace(start, end, num=SELECTED_FRAMES).round().astype(int)
    amplitudes = []
    phases = []

    for frame_id in frame_ids:
        member = (
            f"{env}/{student}/{action}/wifi-csi/frame{frame_id:03d}.mat"
        )
        try:
            raw_mat = zip_file.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"File frame tidak ditemukan di ZIP: {member}") from exc

        mat = scipy.io.loadmat(io.BytesIO(raw_mat))
        if "CSIamp" not in mat or "CSIphase" not in mat:
            raise KeyError(f"CSIamp/CSIphase tidak ditemukan pada {member}")

        amp = np.asarray(mat["CSIamp"], dtype=np.float64)
        phase = np.asarray(mat["CSIphase"], dtype=np.float64)
        if amp.shape != EXPECTED_FRAME_SHAPE:
            raise ValueError(f"Shape amplitude {member} tidak sesuai: {amp.shape}")
        if phase.shape != EXPECTED_FRAME_SHAPE:
            raise ValueError(f"Shape fase {member} tidak sesuai: {phase.shape}")

        amplitudes.append(amp)
        phases.append(phase)

    amplitude = np.stack(amplitudes, axis=0)
    phase = np.stack(phases, axis=0)

    # Perbaikan nilai amplitude dan sanitasi fase.
    amplitude = repair_amplitude(amplitude)
    phase = sanitize_phase(phase)

    # Pengurangan noise pada dimensi sampel terakhir.
    amplitude = reduce_noise(amplitude)
    phase = reduce_noise(phase)

    # (6 frame, 6 kanal amplitude/fase, 114, 10) -> (36, 114, 10)
    combined = np.concatenate([amplitude, phase], axis=1)
    feature = combined.reshape(FEATURE_SHAPE).astype(np.float32)

    if not np.isfinite(feature).all():
        raise ValueError("Fitur akhir mengandung NaN/Inf")
    return feature


def run_preprocessing(dataset_dir: Path, manifest_path: Path, output_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    samples = pd.read_csv(manifest_path).reset_index(drop=True)
    missing_columns = REQUIRED_COLUMNS - set(samples.columns)
    if missing_columns:
        raise ValueError(
            "Kolom manifest belum lengkap: " + ", ".join(sorted(missing_columns))
        )
    if samples.empty:
        raise ValueError("Manifest tidak memiliki sampel")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "mmfi_trackD_features.npy"
    metadata_path = output_dir / "mmfi_trackD_metadata.csv"
    errors_path = output_dir / "mmfi_trackD_preprocessing_errors.csv"

    n_samples = len(samples)
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, *FEATURE_SHAPE),
    )
    features[:] = np.nan
    errors = []

    for env, group in samples.groupby("Environment", sort=False):
        zip_path = dataset_dir / f"{env}.zip"
        print(f"\nMemproses {env}: {len(group)} sampel")

        if not zip_path.is_file():
            message = f"ZIP lingkungan tidak ditemukan: {zip_path}"
            for idx, row in group.iterrows():
                errors.append({
                    "index": int(idx),
                    "sample": f"{row['Environment']}/{row['Student']}/{row['Action']}",
                    "error": message,
                })
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                for idx, row in tqdm(group.iterrows(), total=len(group), desc=str(env)):
                    try:
                        features[idx] = preprocess_sample(archive, row)
                    except Exception as exc:
                        errors.append({
                            "index": int(idx),
                            "sample": f"{row['Environment']}/{row['Student']}/{row['Action']}",
                            "error": repr(exc),
                        })
        except zipfile.BadZipFile as exc:
            for idx, row in group.iterrows():
                errors.append({
                    "index": int(idx),
                    "sample": f"{row['Environment']}/{row['Student']}/{row['Action']}",
                    "error": f"ZIP rusak/tidak dapat dibuka: {exc}",
                })
        features.flush()

    valid_mask = np.isfinite(features).all(axis=(1, 2, 3))
    metadata = samples.copy()
    metadata["feature_index"] = np.arange(n_samples, dtype=np.int64)
    metadata["valid"] = valid_mask
    metadata.to_csv(metadata_path, index=False)
    pd.DataFrame(errors, columns=["index", "sample", "error"]).to_csv(
        errors_path, index=False
    )

    print("\n=== RINGKASAN PREPROCESSING ===")
    print("Total sampel       :", n_samples)
    print("Sampel berhasil    :", int(valid_mask.sum()))
    print("Sampel gagal       :", int((~valid_mask).sum()))
    print("Shape cache        :", features.shape)
    print("Ukuran cache (MiB) :", round(feature_path.stat().st_size / (1024 ** 2), 2))
    print("Cache fitur        :", feature_path)
    print("Metadata           :", metadata_path)
    print("Catatan error      :", errors_path)

    del features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing Wi-Fi CSI untuk eksperimen MM-Fi Track D."
    )
    parser.add_argument(
        "--dataset-dir",
        default="/content/drive/MyDrive/MMFi Dataset Split",
        help="Folder berisi E01.zip, E02.zip, E03.zip, dan E04.zip.",
    )
    parser.add_argument(
        "--manifest",
        default="/content/drive/MyDrive/mmfi_trackD_metadata.csv",
        help="CSV manifest sampel yang sudah memiliki split dan rentang frame.",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/MMFi_Track_D_Final/preprocessed",
        help="Folder keluaran cache fitur, metadata, dan error.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        dataset_dir=Path(args.dataset_dir),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
    )
