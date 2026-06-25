<div align="center">

# 🧠 Web Naive Bayes — Analisis Sentimen NBC

**Sistem analisis sentimen masyarakat terhadap Kabinet Prabowo Subianto**  
berdasarkan data Twitter/X menggunakan Naïve Bayes Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://mysql.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=flat-square&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

*Skripsi — Teknik Informatika, Universitas Malikussaleh*  
*Penulis: Adetia Irvanda · Pembimbing: Khairul*

</div>

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Tech Stack](#-tech-stack)
- [Pipeline Sistem](#-pipeline-sistem)
- [Prasyarat](#-prasyarat)
- [Instalasi](#-instalasi)
- [Konfigurasi](#-konfigurasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Skema Database](#-skema-database)
- [Algoritma](#-algoritma)
- [Evaluasi Model](#-evaluasi-model)
- [Referensi](#-referensi)
- [Lisensi](#-lisensi)

---

## 📖 Tentang Proyek

`web_naive_bayes` adalah aplikasi web **analisis sentimen** berbasis Flask yang dirancang untuk mengklasifikasikan opini publik masyarakat Indonesia terhadap Kabinet Prabowo Subianto melalui data Twitter/X.

Sistem ini dibangun sebagai implementasi untuk penelitian skripsi dengan judul:

> **"Analisis Sentimen Masyarakat Terhadap Pemerintah di Era Kabinet Prabowo Subianto berdasarkan Sosial Media X menggunakan Naïve Bayes Classifier"**

### Masalah yang Diselesaikan

Metode pelabelan sentimen tradisional berbasis keyword matching menghasilkan distribusi label yang sangat tidak seimbang (>77% netral) dan tidak memiliki landasan akademik yang kuat. Sistem ini menggantinya dengan **InSet Lexicon** — kamus sentimen berbobot bahasa Indonesia yang telah dipublikasikan secara ilmiah.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🕷️ **Data Management** | Kelola data tweet yang telah di-scraping dari Twitter/X |
| 🧹 **Preprocessing 7 Tahap** | Cleansing → Case Folding → Tokenizing → Stopword Removal → Normalisasi → Stemming (ECS) → Filter |
| 🏷️ **InSet Lexicon Labeling** | Pelabelan otomatis berbasis kamus sentimen berbobot bahasa Indonesia |
| 📊 **TF-IDF Vectorization** | Pembobotan fitur dengan Term Frequency–Inverse Document Frequency |
| 🤖 **Multinomial Naïve Bayes** | Training & testing model klasifikasi sentimen |
| 📈 **Evaluasi Lengkap** | Confusion matrix, akurasi, precision, recall, F1-score per kelas |
| ☁️ **Word Cloud** | Visualisasi kata dominan per kelas sentimen |
| 🧮 **Kalkulasi Manual NBC** | Perhitungan step-by-step untuk 5 sampel (keperluan BAB III) |
| 💬 **Custom Dialog System** | Semua konfirmasi menggunakan dialog box modern (tanpa browser alert) |
| ⚡ **Loading Overlay** | Animasi loading saat proses NBC berjalan |

---

## 🛠️ Tech Stack

### Backend
| Komponen | Teknologi | Versi |
|----------|-----------|-------|
| Framework | Flask | 3.1.1 |
| ORM | SQLAlchemy | 2.0.41 |
| Database | MySQL | 8.0+ |
| ML Library | scikit-learn | 1.7.0 |
| NLP Stemmer | PySastrawi (ECS) | 1.0.1 |
| Data Processing | Pandas, NumPy | 2.3.0, 2.3.0 |
| Visualisasi | Matplotlib, Seaborn | 3.11.0, 0.13.2 |
| Word Cloud | wordcloud | 1.9.6 |

### Frontend
| Komponen | Teknologi |
|----------|-----------|
| CSS Framework | Bootstrap 5.3 |
| UI Template | Star Admin2 |
| Icons | Font Awesome 6 |
| Typography | Google Fonts — Poppins |

---

## 🔄 Pipeline Sistem

Berikut alur end-to-end yang harus diikuti secara berurutan:

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  1. SCRAPING │ →  │ 2. PREPROCESSING │ →  │  3. SENTIMEN     │
│             │    │                 │    │  (InSet Lexicon) │
│ twitter_    │    │ text_preprocess │    │ sentiment_       │
│ scraping    │    │ -ing            │    │ analysis         │
└─────────────┘    └─────────────────┘    └──────────────────┘
                                                   │
                    ┌──────────────────────────────┘
                    ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  4. KONVERSI    │ →  │  5. NBC SPLIT    │ →  │  6. TRAINING │
│  TF-IDF         │    │  70% train       │    │  MultinomialNB│
│                 │    │  30% test        │    │              │
│ tfidf_          │    │ nbc_training     │    │  nbc_model   │
│ conversion      │    │ nbc_testing      │    │              │
└─────────────────┘    └──────────────────┘    └──────────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
         ┌──────────────────┐    ┌────────────────────────┐
         │  7. TESTING      │ →  │  8. EVALUASI           │
         │                 │    │  Accuracy, Precision,   │
         │  Prediksi &     │    │  Recall, F1, Confusion  │
         │  Probabilitas   │    │  Matrix, Word Cloud     │
         └──────────────────┘    └────────────────────────┘
```

---

## 📦 Prasyarat

Pastikan sistem Anda memenuhi persyaratan berikut:

- **Python** 3.10 atau lebih baru
- **MySQL** 8.0 atau lebih baru
- **pip** (Python package manager)
- **Git**

### File Leksikon yang Diperlukan

File berikut **wajib** ada di folder `database/` sebelum menjalankan aplikasi:

| File | Kolom | Keterangan |
|------|-------|------------|
| `inset_lexicon_positive.csv` | `word`, `weight` | Kata positif berbobot +1 s/d +5 |
| `inset_lexicon_negative.csv` | `word`, `weight` | Kata negatif berbobot -1 s/d -5 |
| `kamus_alay.csv` | `slang`, `baku` | Normalisasi slang Twitter |
| `dictionary_baku_nonbaku.csv` | `word`, `wrong` | Kamus KBBI baku-nonbaku |

> **Sumber InSet Lexicon:** [github.com/Abaddon-Beza/InSet](https://github.com/Abaddon-Beza/InSet)  
> **Sumber Kamus Alay:** [github.com/nasalsabila/kamus-alay](https://github.com/nasalsabila/kamus-alay)

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Khairul122/web_naive_bayes.git
cd web_naive_bayes
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Setup Database MySQL

```sql
CREATE DATABASE db_naive_bayes CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. Siapkan File Leksikon

```bash
# Buat folder database jika belum ada
mkdir database

# Letakkan file-file berikut di folder database/:
# - inset_lexicon_positive.csv
# - inset_lexicon_negative.csv
# - kamus_alay.csv
# - dictionary_baku_nonbaku.csv
```

### 6. Jalankan Aplikasi

```bash
python run.py
```

Aplikasi akan berjalan di `http://localhost:5000`

Tabel database dibuat **otomatis** saat aplikasi pertama kali dijalankan.

---

## ⚙️ Konfigurasi

Konfigurasi database ada di `app/__init__.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:password@localhost/db_naive_bayes'
app.config['SECRET_KEY'] = 'your-secret-key-here'
```

Atau gunakan file `.env`:

```env
DATABASE_URI=mysql+mysqlconnector://root:password@localhost/db_naive_bayes
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

### Parameter Default Sistem

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| InSet threshold positif | `+0.5` | Batas skor untuk label positif |
| InSet threshold negatif | `-0.5` | Batas skor untuk label negatif |
| TF-IDF max features | `1000` | Jumlah maksimum fitur vocabulary |
| TF-IDF min_df | `0.01` | Kata minimal di 1% dokumen |
| TF-IDF max_df | `0.95` | Kata tidak lebih dari 95% dokumen |
| NBC alpha (Laplace) | `1.0` | Smoothing parameter |
| Split test size | `30%` | Proporsi data testing |
| Split random state | `42` | Seed untuk reproducibility |

---

## 📖 Cara Penggunaan

### Langkah 1 — Login

Akses `http://localhost:5000/auth/login` dan masuk dengan akun yang terdaftar.

### Langkah 2 — Import Data

Masuk ke menu **Scrapping** dan kelola data tweet yang sudah dikumpulkan.

### Langkah 3 — Preprocessing

1. Buka menu **Preprocessing**
2. (Opsional) Pergi ke **Pengaturan** → Import kamus alay dan KBBI
3. Klik **Proses Tweet** untuk menjalankan preprocessing 7 tahap

### Langkah 4 — Labeling Sentimen

1. Buka menu **Sentimen**
2. (Opsional) Sesuaikan threshold InSet di form konfigurasi
3. Klik **Jalankan Auto-Labeling** untuk memberi label dengan InSet Lexicon

### Langkah 5 — Konversi TF-IDF

1. Buka menu **Konversi**
2. Sesuaikan parameter (max features, min/max df)
3. Klik **Konversi** untuk menghasilkan vektor fitur TF-IDF

### Langkah 6 — NBC Training & Testing

1. Buka menu **Naive Bayes Classifier**
2. **Split Data** — pilih proporsi test size dan random state
3. **Train Model** — masukkan alpha dan jalankan training
4. **Test Model** — jalankan prediksi pada data testing
5. Klik **Lihat Hasil** untuk melihat evaluasi lengkap

---

## 📁 Struktur Proyek

```
web_naive_bayes/
├── app/
│   ├── __init__.py              # Flask app factory & db.create_all()
│   ├── extension.py             # SQLAlchemy instance
│   ├── models/
│   │   ├── AuthModel.py         # User model
│   │   ├── ScrappingModel.py    # TwitterScraping model
│   │   ├── PreprocessingModel.py# TextPreprocessing, Settings, Stopword, Normalization
│   │   ├── SentimenModel.py     # SentimentAnalysis, SentimentSettings
│   │   ├── KonversiModel.py     # TfidfConversion, TfidfVocabulary
│   │   └── NBCModel.py          # NBCTraining, NBCTesting, NBCModel
│   ├── routes/
│   │   ├── AuthRoute.py         # Login, Logout
│   │   ├── DashboardRoute.py    # Halaman ringkasan
│   │   ├── ScrappingRoute.py    # Manajemen data tweet
│   │   ├── PreprocessingRoute.py# Pipeline preprocessing 7 tahap
│   │   ├── SentimenRoute.py     # InSet Lexicon labeling
│   │   ├── KonversiRoute.py     # TF-IDF vectorization
│   │   └── NBCRoute.py          # Training, testing, evaluasi NBC
│   ├── templates/
│   │   ├── layout.html          # Base template + AppDialog global
│   │   ├── sidebar.html         # Navigasi sidebar
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── scrapping/
│   │   ├── preprocessing/
│   │   ├── sentimen/
│   │   ├── konversi/
│   │   └── nbc/
│   └── static/
│       ├── js/
│       ├── css/
│       └── vendors/
├── database/
│   ├── inset_lexicon_positive.csv   # ← Wajib ada
│   ├── inset_lexicon_negative.csv   # ← Wajib ada
│   ├── kamus_alay.csv
│   └── dictionary_baku_nonbaku.csv
├── PRD.md                       # Product Requirements Document
├── transfer_knowledge.md        # Tacit knowledge untuk developer
├── studi.md                     # Bahan studi akademik skripsi
├── requirement.txt
└── run.py
```

---

## 🗃️ Skema Database

```
users ──────────────────────────────────────────────────────┐
  │                                                          │
  ├── twitter_scraping (data tweet mentah)                   │
  │         │                                               │
  │         └── text_preprocessing (hasil 7 tahap)          │
  │                     │                                    │
  │                     └── sentiment_analysis               │
  │                                 │                        │
  │                                 └── tfidf_conversion ────┤
  │                                           │              │
  │                                    ┌──────┴──────┐       │
  │                              nbc_training  nbc_testing   │
  │                                                          │
  ├── preprocessing_settings                                  │
  ├── normalization_dict                                      │
  ├── stopword_list                                           │
  ├── sentiment_settings                                      │
  ├── tfidf_vocabulary                                        │
  └── nbc_model ─────────────────────────────────────────────┘
```

**Total: 13 tabel**

---

## 🔬 Algoritma

### InSet Lexicon Scoring

```
skor_total = Σ weight(token)   untuk token ∈ pos_dict
           + Σ weight(token)   untuk token ∈ neg_dict

Label:
  skor_total > +0.5  →  positif
  skor_total < -0.5  →  negatif
  selain itu         →  netral

Confidence = min(|skor_total| / 5.0, 1.0)
```

### TF-IDF

```
TF(t, d)      = frekuensi term t dalam dokumen d / total term d
IDF(t, D)     = log((1 + N) / (1 + df(t))) + 1
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

### Multinomial Naïve Bayes

```
Ĉ = argmax_c [ log P(C=c) + Σᵢ xᵢ × log P(tᵢ | C=c) ]

P(tᵢ | C=c) = (count(tᵢ,c) + α) / (count(all,c) + α×V)
α = 1.0 (Laplace smoothing)
```

---

## 📊 Evaluasi Model

Sistem menghitung 4 metrik evaluasi utama:

| Metrik | Formula | Keterangan |
|--------|---------|------------|
| **Accuracy** | (TP total) / (total data) | Persentase prediksi benar keseluruhan |
| **Precision** | TP_c / (TP_c + FP_c) | Ketepatan prediksi per kelas |
| **Recall** | TP_c / (TP_c + FN_c) | Kelengkapan deteksi per kelas |
| **F1-Score** | 2 × P × R / (P + R) | Rata-rata harmonik precision & recall |

Semua metrik dihitung sebagai **weighted average** untuk menangani ketidakseimbangan kelas.

---

## 📚 Referensi

```
[1] Koto, F., & Rahmaningtyas, G. H. (2017). InSet Lexicon: Evaluation of a Word
    List for Indonesian Sentiment Analysis in Microblogs. ICAICTA 2017, 391–396.
    https://doi.org/10.1109/ICAICTA.2017.8090993

[2] Nasalsabila. (2020). Kamus Alay. GitHub.
    https://github.com/nasalsabila/kamus-alay

[3] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to
    Information Retrieval. Cambridge University Press.

[4] McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for Naive
    Bayes Text Classification. AAAI-98 Workshop.

[5] Asian, J. et al. (2007). Stemming Indonesian. ACSC 2007, 307–314.
```

---

## 👥 Kontribusi

Proyek ini dibuat untuk keperluan skripsi. Kontribusi tidak dibuka untuk saat ini.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<div align="center">

Dibuat dengan ❤️ untuk keperluan penelitian skripsi  
**Teknik Informatika — Universitas Malikussaleh**

</div>
