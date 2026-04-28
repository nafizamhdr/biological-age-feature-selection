# Biological Age Prediction: Feature Selection & Dimensionality Reduction Experiments

Repositori ini berisi serangkaian eksperimen data mining untuk memprediksi kelompok usia biologis (*Biological Age Group*) berdasarkan data medis dan gaya hidup. Fokus utama dari proyek ini adalah mengeksplorasi dan membandingkan berbagai teknik **Seleksi Fitur** (*Feature Selection*) dan **Reduksi Dimensi** (*Dimensionality Reduction*) untuk mengoptimalkan performa model klasifikasi Machine Learning.

## 📊 Dataset
Dataset yang digunakan berasal dari Kaggle, berisi data rekam medis dan kebiasaan pasien:
- **Jumlah Data**: 3.000 baris (Training) & 3.000 baris (Testing)
- **Fitur**: 27 fitur awal (demografi, biomarker seperti tekanan darah, glukosa, dan kepadatan tulang, serta data gaya hidup) yang diekspansi menjadi 35 fitur setelah preprocessing (One-Hot Encoding, dll).
- **Target Asli**: `Age (years)` (Regresi, rentang usia 18-89 tahun).
- **Target Model**: `Age_Group` (Klasifikasi 4 kelas: Dewasa Muda, Dewasa, Paruh Baya, Lansia), dikonversi melalui teknik *binning* untuk menyeimbangkan distribusi kelas.

##  Struktur Proyek

Proyek ini disusun secara sekuensial ke dalam beberapa Jupyter Notebook:

1. **`00_preprocessing_binning.ipynb`**
   Melakukan data cleaning, imputasi *missing value*, ekstraksi fitur (memecah tekanan darah), scaling (`StandardScaler`), dan binning target variabel dari regresi ke klasifikasi 4 kelas.
2. **`01_baseline_model.ipynb`**
   Skenario *Baseline*. Melatih model **Random Forest**, **SVM (RBF)**, dan **KNN (k=7)** menggunakan seluruh 35 fitur sebagai acuan (*benchmark*) performa.
3. **`02_filter_method.ipynb`**
   Seleksi fitur statistik. Menggunakan **ANOVA F-test** (linear) dan **Mutual Information** (non-linear). Menyaring 35 fitur menjadi 18 fitur terbaik.
4. **`03_wrapper_method.ipynb`**
   Seleksi fitur iteratif. Menggunakan **Recursive Feature Elimination (RFECV)** dengan estimator Random Forest dan LinearSVC.
5. **`04_embedded_method.ipynb`**
   Seleksi fitur *built-in*. Memanfaatkan *feature importance* dari tree-based models (**Random Forest** & **XGBoost**). Berhasil mereduksi hingga 8 fitur kunci (biomarker penuaan).
6. **`05_pca.ipynb`**
   Reduksi dimensi *unsupervised*. Menggunakan **Principal Component Analysis (PCA)** untuk mentransformasi 35 fitur asli ke ruang dimensi baru (26 komponen untuk 95% variance).
7. **`06_lda_komparasi.ipynb`**
   Reduksi dimensi *supervised* menggunakan **Linear Discriminant Analysis (LDA)** (3 komponen) dan komparasi final dari seluruh skenario.

##  Hasil dan Kesimpulan Utama

Eksperimen dievaluasi menggunakan *Stratified 5-Fold Cross Validation*.

| Model | Baseline (35f) | Filter (18f) | Wrapper (25f) | Embedded (8f) | PCA (26pc) |
|-------|---------------|-------------|--------------|--------------|------------|
| Random Forest | 0.8099 | **0.8126** | 0.8108 | 0.8060 | 0.7580 |
| SVM (RBF) | 0.7990 | 0.8020 | 0.8056 | **0.8099** | 0.7580 |
| KNN (k=7) | 0.6117 | 0.6939 | 0.6564 | **0.7715** | 0.6050 |

**Insight Kunci:**
* **Beda Model, Beda Respons**: KNN sangat rentan terhadap *curse of dimensionality* (melompat dari akurasi 61% ke 77% setelah seleksi fitur). Random Forest sangat stabil karena secara internal memiliki mekanisme seleksi fitur saat *splitting*.
* **Feature Selection > PCA**: PCA (Reduksi Dimensi) justru menurunkan performa semua model secara signifikan di dataset ini karena sifatnya yang *unsupervised* (memaksimalkan varians, bukan separasi kelas).
* **Biomarker Inti**: Seluruh metode secara konsisten memilih 8 fitur biologis utama (*Bone Density, Vision Sharpness, Hearing Ability, Cognitive Function, Blood Glucose, Cholesterol, BMI, BP*) sebagai prediktor terkuat usia biologis, selaras dengan realitas medis.

**Rekomendasi Deployment:**
Model **Random Forest + Filter Method (18 fitur)** direkomendasikan karena memberikan performa F1-Score tertinggi (81.2%), fitur tetap dapat diinterpretasi secara medis, dan model sangat *robust*.

##  Cara Menjalankan (Installation & Usage)

1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/biological-age-prediction.git
   cd biological-age-prediction
   ```

2. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Di Windows gunakan: venv\Scripts\activate
   ```

3. Install semua *dependencies*:
   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan Jupyter Notebook dan jalankan skenario secara berurutan:
   ```bash
   jupyter notebook
   ```

---
*Proyek ini adalah studi kasus implementasi Data Mining untuk menyeleksi fitur dan mereduksi dimensi data tabular berdimensi menengah.*
