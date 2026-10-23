
# Proyek MM-Fi Track D: Pengenalan Aktivitas Manusia Menggunakan Wi-Fi CSI

## 1. Tentang Proyek

Proyek ini dibuat untuk mengenali aktivitas manusia menggunakan data Wi-Fi Channel State Information (CSI) dari dataset MM-Fi.

Pada proyek ini, saya membandingkan dua model, yaitu 2D-CNN sebagai model dasar dan CNN-DANN yang menggunakan pendekatan domain adaptation. Tujuannya adalah melihat kemampuan kedua model dalam mengenali aktivitas manusia pada lingkungan yang berbeda.

## 2. Metode yang Digunakan

Proyek ini menggunakan beberapa tahapan:

1. **Preprocessing data:** membaca data CSI, menangani nilai yang tidak valid, melakukan sanitasi fase, dan mengurangi noise.
2. **Normalisasi:** menyesuaikan skala fitur agar lebih sesuai untuk proses pelatihan model.
3. **Pembagian data:** menggunakan data source untuk pelatihan dan validasi, data target adaptation untuk adaptasi domain, serta data target test untuk pengujian akhir.
4. **Pelatihan model:** melatih 2D-CNN dan CNN-DANN menggunakan PyTorch.
5. **Evaluasi:** membandingkan hasil model menggunakan accuracy, macro-F1, weighted-F1, classification report, dan confusion matrix.

## 3. Struktur Repository

- `preprocess.py`: script preprocessing data CSI.
- `data_utils.py`: fungsi untuk memuat, memvalidasi, dan menyiapkan data.
- `models.py`: arsitektur model 2D-CNN dan CNN-DANN.
- `train.py`: script pelatihan model.
- `evaluate.py`: script evaluasi model.
- `Track_D_MMFi_Final.ipynb`: notebook eksperimen di Google Colab.
- `requirements.txt`: daftar pustaka Python yang diperlukan.
- `results/`: hasil evaluasi, prediksi, riwayat pelatihan, dan visualisasi.

## 4. Dataset dan Persiapan

Proyek ini menggunakan dataset MM-Fi, khususnya data Wi-Fi CSI.

Dataset mentah tidak disertakan dalam repository karena ukuran filenya besar. File dataset, anotasi, cache fitur, metadata, dan parameter normalisasi perlu disiapkan secara terpisah di Google Drive.

Sebelum menjalankan kode, sesuaikan lokasi file dengan path yang digunakan dalam script.

## 5. Cara Menjalankan Proyek

Proyek dapat dijalankan menggunakan Google Colab dengan GPU yang mendukung PyTorch.

Langkah-langkahnya:

1. Buka notebook `Track_D_MMFi_Final.ipynb` di Google Colab.
2. Hubungkan Google Drive menggunakan `drive.mount()`.
3. Pastikan dataset atau file pendukung yang diperlukan sudah tersedia.
4. Periksa path dataset, cache fitur, metadata, dan parameter normalisasi.
5. Pasang pustaka yang diperlukan berdasarkan `requirements.txt`.
6. Jalankan tahap preprocessing jika memulai dari dataset mentah.
7. Jalankan validasi data dan persiapan DataLoader.
8. Jalankan training 2D-CNN dan CNN-DANN.
9. Jalankan evaluasi pada target test dan simpan hasilnya ke folder `results`.

Script Python juga dipisahkan ke dalam beberapa file agar lebih mudah dibaca dan dikembangkan. Sebelum menjalankan script secara terpisah, pastikan dependensi, path input, dan lokasi penyimpanan sudah sesuai.

## 6. Hasil Eksperimen

Berdasarkan pengujian target test terakhir yang tercatat, diperoleh hasil berikut:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| 2D-CNN | 5,56% | 2,85% | 2,85% |
| CNN-DANN | 6,79% | 4,11% | 4,11% |

Pada pengujian tersebut, CNN-DANN memperoleh hasil lebih baik daripada 2D-CNN pada ketiga metrik. Namun, nilai evaluasinya masih rendah sehingga model perlu dikembangkan lebih lanjut.

## 7. Perangkat dan Pustaka

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Google Colab

Versi pustaka yang digunakan dapat dilihat pada `requirements.txt`.

## 8. Kesimpulan

Proyek ini membandingkan model 2D-CNN dan CNN-DANN untuk pengenalan aktivitas manusia menggunakan Wi-Fi CSI. Hasil awal menunjukkan bahwa CNN-DANN lebih baik pada target test, tetapi performa kedua model masih perlu ditingkatkan.

Repository ini disusun untuk mendokumentasikan kode preprocessing, arsitektur model, pelatihan, evaluasi, dan hasil eksperimen agar proses penelitian lebih mudah dipahami dan dikembangkan.

## 9. Contoh Menjalankan Proyek di Google Colab

Pastikan Google Drive sudah terhubung dan file cache fitur, metadata, serta normalisasi sudah tersedia sesuai lokasi yang digunakan pada script.

### Memasang pustaka

```python
!pip install -r requirements.txt
```

### Melatih model

```python
!python train.py
```

Perintah ini melatih model 2D-CNN dan CNN-DANN serta menyimpan checkpoint dan riwayat pelatihan.

### Mengevaluasi model

```python
!python evaluate.py
```

Evaluasi menggunakan checkpoint yang sudah tersedia dan menyimpan hasil pengujian ke folder `results`.

**Catatan:** Jalankan perintah dari folder repository yang berisi file Python. Pastikan Google Drive sudah di-mount dan file data serta checkpoint tersedia pada lokasi yang sesuai.

