# Proyek MM-Fi Track D: Pengenalan Aktivitas Manusia Menggunakan Wi-Fi CSI

## 1. Tentang Proyek

Proyek ini membahas pengenalan aktivitas manusia menggunakan data Wi-Fi Channel State Information (CSI) dari dataset MM-Fi. Pada proyek ini, saya membandingkan dua model, yaitu 2D-CNN sebagai model dasar dan CNN-DANN sebagai model yang menggunakan metode domain adaptation.

Tujuan proyek ini adalah mengetahui bagaimana kedua model mengenali aktivitas manusia ketika data yang digunakan berasal dari lingkungan yang berbeda.

## 2. Metode yang Digunakan

Dalam proyek ini, saya menggunakan dua model:

- **2D-CNN:** model dasar untuk mengenali aktivitas manusia berdasarkan fitur Wi-Fi CSI.
- **CNN-DANN:** model yang menggunakan domain adaptation untuk membantu model menyesuaikan diri dengan perbedaan lingkungan.

Data yang digunakan berasal dari dua komponen Wi-Fi CSI, yaitu amplitudo dan fase. Data tersebut melalui proses preprocessing dan normalisasi sebelum digunakan untuk melatih model.

## 3. Dataset dan Pembagian Data

Dataset MM-Fi yang digunakan terdiri dari empat lingkungan, yaitu E01, E02, E03, dan E04. Lingkungan E01 sampai E03 digunakan sebagai sumber data, sedangkan E04 digunakan sebagai lingkungan target.

Data dibagi menjadi empat bagian:

| Pembagian Data | Jumlah Sampel | Kegunaan |
|---|---:|---|
| Source Train | 1.940 | Melatih model |
| Source Validation | 486 | Memilih model terbaik |
| Target Adaptation | 486 | Penyesuaian model menggunakan data target tanpa label aktivitas saat pelatihan |
| Target Test | 324 | Menguji hasil akhir model |

Terdapat 27 kelas aktivitas yang digunakan dalam penelitian ini. Dari seluruh data yang diproses, terdapat 3.236 sampel valid yang digunakan dalam eksperimen.

## 4. Hasil Pengujian

Pengujian akhir dilakukan menggunakan data target test dari lingkungan E04. Hasil kedua model adalah sebagai berikut.

| Metrik | 2D-CNN | CNN-DANN |
|---|---:|---:|
| Accuracy | 5,56% | 6,79% |
| Macro-F1 | 2,85% | 4,11% |
| Weighted-F1 | 2,85% | 4,11% |

Berdasarkan hasil tersebut, CNN-DANN memperoleh nilai yang lebih tinggi dibandingkan 2D-CNN pada ketiga metrik pengujian. Namun, nilai yang diperoleh kedua model masih rendah. Artinya, model masih mengalami kesulitan dalam mengenali aktivitas ketika diuji pada lingkungan yang berbeda.

## 5. Isi Repository

Repository ini digunakan untuk menyimpan kode, dokumentasi, dan hasil dari proyek MM-Fi Track D.

- `Track_D_MMFi_Final.ipynb`: notebook Google Colab yang berisi proses preprocessing, pembuatan model, pelatihan, dan evaluasi.
- `README.md`: dokumentasi singkat mengenai proyek, metode, pembagian data, dan hasil pengujian.
- `results/`: tempat menyimpan tabel hasil pengujian, prediksi, dan visualisasi.
- `checkpoints/`: tempat menyimpan model yang telah dilatih.
- `artifacts/`: tempat menyimpan parameter normalisasi yang digunakan dalam eksperimen.

Beberapa file berukuran besar, seperti dataset mentah dan cache fitur, tetap disimpan di Google Drive agar repository tidak terlalu besar.

## 6. Kesimpulan

Berdasarkan hasil eksperimen, CNN-DANN menunjukkan hasil yang lebih baik dibandingkan 2D-CNN pada pengujian di lingkungan E04. Meskipun demikian, performa kedua model masih rendah sehingga diperlukan pengembangan lebih lanjut.

Proyek ini menjadi langkah awal untuk mempelajari penerapan deep learning dan domain adaptation dalam pengenalan aktivitas manusia menggunakan Wi-Fi CSI.

## 7. Tools yang Digunakan

- Python
- Google Colab
- PyTorch
- NumPy dan Pandas
- Scikit-learn
- Matplotlib
- Dataset MM-Fi
