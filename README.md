# Proyek MM-Fi Track D: Pengenalan Aktivitas Manusia Menggunakan Wi-Fi CSI

## 1. Tentang Proyek

Proyek ini bertujuan untuk mengenali aktivitas manusia menggunakan data Wi-Fi Channel State Information (CSI) dari dataset MM-Fi.

Pada proyek ini, dua model *deep learning* dibandingkan, yaitu **2D-CNN sebagai model dasar (baseline)** dan **CNN-DANN (Domain-Adversarial Neural Network)** menggunakan pendekatan *domain adaptation*. Tujuannya adalah mengevaluasi kemampuan kedua model dalam mengenali aktivitas manusia ketika terdapat perbedaan lingkungan antara data pelatihan dan data pengujian.

Eksperimen menggunakan lingkungan E01, E02, dan E03 sebagai *source domain*, sedangkan lingkungan E04 digunakan sebagai *target domain*. Data target dibagi menjadi data adaptasi dan data pengujian untuk mengevaluasi kemampuan model dalam menghadapi lingkungan yang berbeda.

Proyek ini diimplementasikan menggunakan Python dan PyTorch, dengan Google Colab sebagai lingkungan eksperimen.

## 2. Metode yang Digunakan

Proyek ini terdiri atas beberapa tahapan utama berikut.

1. **Preprocessing data:** membaca data Wi-Fi CSI, menangani nilai yang tidak valid, melakukan sanitasi fase, dan mengurangi noise sesuai prosedur preprocessing yang digunakan.
2. **Ekstraksi dan penyusunan fitur:** menggabungkan informasi amplitudo dan fase CSI serta memilih enam frame dari setiap segmen aktivitas untuk membentuk input model.
3. **Normalisasi:** menyesuaikan skala fitur menggunakan parameter mean dan standard deviation yang dihitung dari data pelatihan pada *source domain*.
4. **Pembagian data:** menggunakan data source untuk pelatihan dan validasi, data target adaptation untuk adaptasi domain tanpa label aktivitas, serta data target test untuk evaluasi akhir.
5. **Pelatihan model:** melatih model 2D-CNN dan CNN-DANN menggunakan PyTorch.
6. **Evaluasi model:** membandingkan hasil kedua model menggunakan accuracy, macro-F1, weighted-F1, classification report, dan confusion matrix.

### Model yang digunakan

**2D-CNN (Baseline)**

Model CNN dua dimensi yang digunakan sebagai pembanding untuk mengklasifikasikan aktivitas manusia berdasarkan fitur Wi-Fi CSI.

**CNN-DANN (Domain Adaptation)**

Model CNN yang menggabungkan pengklasifikasi aktivitas dan pengklasifikasi domain. Model ini menggunakan *Gradient Reversal Layer* (GRL) untuk membantu mempelajari fitur yang lebih sesuai digunakan pada lingkungan sumber dan target.

## 3. Struktur Repository

Struktur repository proyek disusun untuk memisahkan preprocessing, pengolahan data, arsitektur model, pelatihan, evaluasi, dan hasil eksperimen.

```text
MMFi_Track_D_Final/
├── README.md
├── requirements.txt
├── preprocess.py
├── data_utils.py
├── models.py
├── train.py
├── evaluate.py
├── Track_D_MMFi_Final.ipynb
├── results/
│   ├── mmfi_trackD_final_test_results.csv
│   ├── mmfi_trackD_final_test_predictions.csv
│   ├── mmfi_trackD_final_per_class.csv
│   ├── mmfi_trackD_final_training_curves.png
│   ├── mmfi_trackD_final_confusion_matrix.png
│   ├── mmfi_trackD_final_2dcnn_history.csv
│   ├── mmfi_trackD_final_cnndann_history.csv
│   └── mmfi_trackD_metadata.csv
├── checkpoints/
│   ├── mmfi_trackD_final_2dcnn.pt
│   └── mmfi_trackD_final_cnndann.pt
└── artifacts/
    └── mmfi_trackD_normalization.npz
```

Keterangan:

- `preprocess.py`: melakukan preprocessing data CSI dan menyiapkan fitur yang dibutuhkan.
- `data_utils.py`: menyediakan fungsi untuk memuat, memvalidasi, dan menyiapkan data serta DataLoader.
- `models.py`: mendefinisikan arsitektur model 2D-CNN dan CNN-DANN.
- `train.py`: menjalankan proses pelatihan model dan menyimpan checkpoint serta riwayat pelatihan.
- `evaluate.py`: menjalankan evaluasi model dan menyimpan hasil pengujian.
- `Track_D_MMFi_Final.ipynb`: notebook eksperimen yang digunakan di Google Colab.
- `requirements.txt`: berisi daftar pustaka Python yang diperlukan.
- `results/`: menyimpan metrik evaluasi, prediksi, hasil per kelas, riwayat pelatihan, metadata, dan visualisasi.
- `checkpoints/`: menyimpan checkpoint model yang telah dilatih.
- `artifacts/`: menyimpan parameter normalisasi yang digunakan selama eksperimen.

*Catatan: sesuaikan struktur di atas dengan file yang benar-benar tersedia di repository. Tidak semua file harus disertakan jika fungsinya hanya dijalankan melalui notebook.*

## 4. Dataset dan Persiapan Data

Proyek ini menggunakan dataset MM-Fi, khususnya modalitas Wi-Fi CSI yang memuat informasi amplitudo dan fase sinyal.

Dataset mentah tidak disertakan dalam repository karena ukuran filenya besar. Oleh karena itu, dataset, anotasi, cache fitur, metadata, dan parameter normalisasi perlu disiapkan secara terpisah di Google Drive sebelum menjalankan eksperimen.

### Konfigurasi data

- **Source domain:** E01, E02, dan E03.
- **Target domain:** E04.
- **Jumlah kelas aktivitas:** 27 kelas.
- **Jumlah frame yang dipilih:** enam frame dari setiap segmen aktivitas.
- **Bentuk input model:** `(36, 114, 10)`.
- **Sampel tidak valid:** satu sampel, yaitu E03/S24/A11, dikeluarkan dari data yang digunakan.

### Pembagian data

Pembagian data yang digunakan pada eksperimen final adalah sebagai berikut.

| Split | Jumlah sampel valid | Lingkungan |
|---|---:|---|
| Source train | 1.940 | E01, E02, E03 |
| Source validation | 486 | E01, E02, E03 |
| Target adaptation | 486 | E04 |
| Target test | 324 | E04 |
| **Total** | **3.236** | |

Data `source_train` digunakan untuk melatih model, sedangkan `source_val` digunakan untuk memantau performa model selama pelatihan. Data `target_adapt` digunakan untuk proses adaptasi domain tanpa menggunakan label aktivitas dalam pelatihan CNN-DANN. Sementara itu, `target_test` digunakan untuk evaluasi akhir pada lingkungan target.

Label aktivitas pada target test digunakan untuk menghitung metrik evaluasi setelah model selesai dilatih.

### Persiapan sebelum menjalankan notebook

Pastikan file berikut tersedia dan path-nya sesuai dengan konfigurasi notebook:

- `mmfi_trackD_features.npy`
- `mmfi_trackD_metadata.csv`
- `mmfi_trackD_normalization.npz`

Jika file tersebut disimpan di folder berbeda, sesuaikan variabel path pada notebook. Parameter normalisasi yang digunakan harus konsisten dengan parameter yang dihasilkan dari data pelatihan.

## 5. Cara Menjalankan Proyek

Proyek dapat dijalankan menggunakan Google Colab dengan GPU yang mendukung PyTorch.

### Langkah 1 — Buka notebook

Buka file `Track_D_MMFi_Final.ipynb` menggunakan Google Colab.

### Langkah 2 — Hubungkan Google Drive

Jalankan kode berikut untuk menghubungkan Google Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Langkah 3 — Periksa file dan path

Pastikan dataset atau file pendukung yang diperlukan sudah tersedia. Periksa path dataset, cache fitur, metadata, parameter normalisasi, dan lokasi penyimpanan checkpoint.

Jika memulai dari dataset mentah, jalankan preprocessing terlebih dahulu. Jika menggunakan cache fitur yang sudah disiapkan, tahap preprocessing dapat dilewati sesuai kebutuhan eksperimen.

### Langkah 4 — Instal dependensi

Jika menjalankan script secara terpisah, pindah ke direktori repository, kemudian instal pustaka yang dibutuhkan.

```python
!pip install -r requirements.txt
```

### Langkah 5 — Jalankan preprocessing

Jika repository menyediakan script preprocessing dan eksperimen dimulai dari dataset mentah, jalankan:

```python
!python preprocess.py
```

Pastikan path dataset dan lokasi output sudah sesuai dengan konfigurasi yang digunakan.

### Langkah 6 — Jalankan pelatihan model

Jika script pelatihan tersedia dan sudah dikonfigurasi untuk menggunakan data yang benar, jalankan:

```python
!python train.py
```

Tahap ini digunakan untuk melatih model 2D-CNN dan CNN-DANN serta menyimpan checkpoint dan riwayat pelatihan.

### Langkah 7 — Jalankan evaluasi

Setelah checkpoint model tersedia, jalankan:

```python
!python evaluate.py
```

Hasil evaluasi disimpan ke folder `results/` sesuai dengan konfigurasi script.

**Penting:** perintah untuk menjalankan script secara terpisah hanya berlaku jika file Python tersebut benar-benar tersedia dan sudah dikonfigurasi. Jika menggunakan notebook sebagai implementasi utama, jalankan sel notebook secara berurutan.

## 6. Hasil Eksperimen

Evaluasi akhir dilakukan pada `target_test`, yang terdiri atas 324 sampel valid dari lingkungan E04. Hasil kedua model ditampilkan menggunakan tiga metrik utama: accuracy, macro-F1, dan weighted-F1.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| 2D-CNN | 5,86% | 3,72% | 3,72% |
| CNN-DANN | 6,48% | 3,83% | 3,83% |

Berdasarkan hasil tersebut, CNN-DANN memperoleh accuracy sebesar 6,48%, lebih tinggi daripada 2D-CNN sebesar 5,86%. CNN-DANN juga menghasilkan macro-F1 dan weighted-F1 sebesar 3,83%, sedikit lebih tinggi daripada kedua metrik pada model baseline yang masing-masing sebesar 3,72%.

Hasil ini menunjukkan bahwa CNN-DANN memberikan peningkatan numerik dibandingkan 2D-CNN pada ketiga metrik yang dievaluasi. Namun, peningkatannya relatif kecil dan performa kedua model pada lingkungan target masih rendah.

Temuan ini menunjukkan bahwa pengenalan aktivitas manusia menggunakan Wi-Fi CSI pada lingkungan yang berbeda masih menjadi tantangan. Perbedaan kondisi lingkungan dapat memengaruhi karakteristik sinyal yang diterima sehingga fitur yang dipelajari dari lingkungan sumber belum tentu dapat digeneralisasikan dengan baik ke lingkungan target.

Hasil tersebut juga menunjukkan bahwa penggunaan domain adaptation belum secara otomatis menghasilkan performa pengenalan aktivitas yang tinggi. Diperlukan eksperimen lebih lanjut untuk mengidentifikasi faktor yang memengaruhi hasil dan meningkatkan kemampuan generalisasi model.

### File hasil evaluasi

Hasil eksperimen disimpan dalam beberapa file berikut:

- `mmfi_trackD_final_test_results.csv`: ringkasan metrik evaluasi kedua model.
- `mmfi_trackD_final_test_predictions.csv`: label sebenarnya, prediksi model, dan metadata sampel pengujian.
- `mmfi_trackD_final_per_class.csv`: hasil evaluasi untuk setiap kelas aktivitas.
- `mmfi_trackD_final_training_curves.png`: visualisasi kurva pelatihan.
- `mmfi_trackD_final_confusion_matrix.png`: visualisasi confusion matrix kedua model.
- `mmfi_trackD_final_2dcnn_history.csv`: riwayat pelatihan 2D-CNN.
- `mmfi_trackD_final_cnndann_history.csv`: riwayat pelatihan CNN-DANN.

## 7. Perangkat dan Pustaka

Perangkat dan pustaka yang digunakan dalam proyek ini meliputi:

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Google Colab

Eksperimen dijalankan menggunakan GPU Tesla T4 pada lingkungan Google Colab. Versi pustaka yang digunakan dapat diperiksa melalui `requirements.txt` atau lingkungan eksperimen yang digunakan.

## 8. Kesimpulan dan Pengembangan Selanjutnya

Proyek ini membandingkan model 2D-CNN dan CNN-DANN untuk pengenalan aktivitas manusia menggunakan Wi-Fi CSI pada lingkungan yang
