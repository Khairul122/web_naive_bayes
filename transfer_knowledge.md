# Transfer Knowledge — Tacit Knowledge Document
## Sistem Analisis Sentimen `web_naive_bayes`

| | |
|---|---|
| **Proyek** | Analisis Sentimen Kabinet Prabowo Subianto |
| **Stack** | Python 3 · Flask · MySQL · Bootstrap 5 (Star Admin2) |
| **Penulis TK** | Khairul |
| **Tanggal** | 26 Juni 2026 |
| **Status Sistem** | Production-ready untuk keperluan skripsi |

---

## 1. Konteks dan Tujuan Sistem

Sistem ini dibangun untuk skripsi mahasiswa **Adetia Irvanda** (Teknik Informatika, Universitas Malikussaleh) dengan judul:

> *"Analisis Sentimen Masyarakat Terhadap Pemerintah di Era Kabinet Prabowo Subianto berdasarkan Sosial Media X menggunakan Naïve Bayes Classifier"*

**Tujuan utama sistem:**
1. Scraping data tweet dari Twitter/X tentang topik kabinet
2. Preprocessing teks bahasa Indonesia (cleansing → stemming)
3. Pelabelan sentimen otomatis menggunakan InSet Lexicon (bukan keyword biasa)
4. Pembobotan fitur dengan TF-IDF
5. Klasifikasi sentimen dengan Multinomial Naive Bayes
6. Evaluasi hasil (confusion matrix, accuracy, precision, recall, F1)

---

## 2. Arsitektur Sistem — Gambaran Besar

```
[Browser User]
      │
      ▼
[Flask App — app/__init__.py]
      │
      ├── Blueprint: auth_bp      (/auth)       → Login, Logout
      ├── Blueprint: dashboard_bp (/)           → Halaman ringkasan
      ├── Blueprint: scrapping_bp (/scrapping)  → Kelola tweet scraping
      ├── Blueprint: preprocessing_bp (/preprocessing) → Bersihkan teks
      ├── Blueprint: sentimen_bp  (/sentimen)   → Labeling InSet Lexicon
      ├── Blueprint: konversi_bp  (/konversi)   → TF-IDF vectorization
      └── Blueprint: nbc_bp       (/nbc)        → Training/testing NBC
            │
            ▼
      [MySQL — db_naive_bayes]
```

**Setiap blueprint = 1 file route + 1 file model (umumnya).**

---

## 3. Alur Pipeline End-to-End

Urutan ini HARUS diikuti secara berurutan. Melewati satu langkah akan menyebabkan data kosong di langkah berikutnya.

```
STEP 1  Scrapping      → tabel: twitter_scraping
STEP 2  Preprocessing  → tabel: text_preprocessing
STEP 3  Sentimen       → tabel: sentiment_analysis
STEP 4  Konversi       → tabel: tfidf_conversion + tfidf_vocabulary
STEP 5  NBC Split      → tabel: nbc_training + nbc_testing
STEP 6  NBC Train      → tabel: nbc_model (parameter disimpan)
STEP 7  NBC Test       → update nbc_testing + nbc_model (akurasi)
STEP 8  NBC Results    → /nbc/results (baca semua tabel di atas)
```

**Jika ingin ulang dari awal:** reset harus dilakukan dari bawah ke atas (NBC → Konversi → Sentimen → Preprocessing), karena ada foreign key dependency.

---

## 4. Struktur File yang Penting

```
web_naive_bayes/
├── app/
│   ├── __init__.py              ← Factory app Flask, db.create_all() di sini
│   ├── extension.py             ← Inisialisasi db (SQLAlchemy instance)
│   ├── models/
│   │   ├── AuthModel.py         ← User (id_user, username, password, role)
│   │   ├── ScrappingModel.py    ← TwitterScraping (tweet raw)
│   │   ├── PreprocessingModel.py← TextPreprocessing, PreprocessingSettings,
│   │   │                          StopwordList, NormalizationDict
│   │   ├── SentimenModel.py     ← SentimentAnalysis, SentimentSettings
│   │   ├── KonversiModel.py     ← TfidfConversion, TfidfVocabulary
│   │   └── NBCModel.py          ← NBCTraining, NBCTesting, NBCModel
│   ├── routes/
│   │   ├── AuthRoute.py
│   │   ├── DashboardRoute.py
│   │   ├── ScrappingRoute.py
│   │   ├── PreprocessingRoute.py← CORE: preprocess_text(), stemming, normalisasi
│   │   ├── SentimenRoute.py     ← CORE: inset_label_sentiment(), load_inset_lexicon()
│   │   ├── KonversiRoute.py     ← CORE: TfidfVectorizer, simpan vocab
│   │   └── NBCRoute.py          ← CORE: MultinomialNB, evaluasi, wordcloud, manual calc
│   └── templates/
│       ├── layout.html          ← Base template + AppDialog global dialog JS
│       ├── sidebar.html         ← Navigasi + logout confirm
│       ├── auth/login.html      ← Standalone (tidak extend layout.html)
│       └── [modul]/index.html
├── database/
│   ├── inset_lexicon_positive.csv  ← Kolom: word, weight (skor +1 s/d +5)
│   ├── inset_lexicon_negative.csv  ← Kolom: word, weight (skor -1 s/d -5)
│   ├── kamus_alay.csv              ← Kolom: slang, baku
│   └── dictionary_baku_nonbaku.csv ← Kolom: word (baku), wrong (tidak baku)
├── PRD.md                       ← Product Requirements Document (referensi utama)
├── transfer_knowledge.md        ← File ini
└── studi.md                     ← Bahan studi untuk skripsi
```

---

## 5. Database Schema & Relasi

```
users (AuthModel)
  id_user  PK
     │
     ├──────────────────────────────────────────────────┐
     │                                                  │
twitter_scraping (ScrappingModel)              preprocessing_settings
  id  PK                                       normalization_dict
  tweet_id_str  UNIQUE                         stopword_list
  full_text                                    sentiment_settings
  scraped_by  FK→users.id_user
     │
     ▼
text_preprocessing (PreprocessingModel)
  id  PK
  tweet_id  FK→twitter_scraping.id
  original_text → cleaned → case_folded → tokenized → filtered → normalized → stemmed → final_text
  processed_by  FK→users.id_user
     │
     ▼
sentiment_analysis (SentimenModel)
  id  PK
  tweet_id  FK→twitter_scraping.id
  preprocessing_id  FK→text_preprocessing.id (nullable — bisa skip)
  sentiment_label  ENUM('positif','negatif','netral')
  confidence_score  FLOAT (0.0–1.0)
  positive_keywords, negative_keywords  TEXT (dipisah koma)
  labeled_by  FK→users.id_user
     │
     ▼
tfidf_conversion (KonversiModel)
  id  PK
  sentiment_id  FK→sentiment_analysis.id
  feature_vector  TEXT (JSON array of float)  ← PENTING: disimpan sebagai JSON string
  feature_names   TEXT (JSON array of string)
  total_features  INT (jumlah kolom TF-IDF)
  converted_by  FK→users.id_user
     │
     ├──→ tfidf_vocabulary (per user, global untuk semua sentimen user tsb)
     │      term, feature_index, idf_score
     │
     ├──→ nbc_training
     │      feature_vector  TEXT (JSON)
     │      label  ('positif'/'negatif'/'netral')
     │
     └──→ nbc_testing
            feature_vector  TEXT (JSON)
            true_label, predicted_label, prediction_probability
            is_correct  BOOLEAN

nbc_model (satu per user — selalu overwrite saat train ulang)
  feature_log_prob  TEXT (JSON 2D array)
  class_log_prior   TEXT (JSON 1D array)
  classes           TEXT (JSON array of string)
  accuracy, precision_score, recall_score, f1_score  FLOAT
  classification_report  TEXT (JSON dict)
```

### Hal Penting yang Tidak Tertulis di Kode

- **`feature_vector` disimpan sebagai JSON string**, bukan BLOB. Saat dibaca, harus selalu `json.loads(data.feature_vector)` dan dikonversi ke `np.array()`.
- **`nbc_model` selalu di-DELETE sebelum disimpan ulang.** Tidak ada versioning model — setiap training menghapus model lama.
- **`nbc_testing.prediction_probability`** adalah JSON array of float (satu nilai per kelas, urutan sesuai `model.classes_`).
- **`TfidfVocabulary` juga di-DELETE setiap kali konversi dijalankan** — bukan append.
- **`sentiment_analysis.labeled_by`** menggunakan kolom `labeled_by` (bukan `analyzed_by`). Ada inkonsistensi nama di satu tempat di results route — perhatikan ini.

---

## 6. Cara Menjalankan Sistem

### Prasyarat
```
Python 3.10+
MySQL 8.0+ dengan database: db_naive_bayes
pip install -r requirement.txt
```

### File konfigurasi database
File `.env` atau langsung di `app/__init__.py`:
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:password@localhost/db_naive_bayes'
```

### Jalankan
```bash
# Dari folder web_naive_bayes/
python run.py
# atau
flask run
```

### Inisialisasi tabel
Tabel dibuat otomatis saat pertama kali app dijalankan (`db.create_all()` di `app/__init__.py`). Pastikan semua model di-import sebelum `create_all()`.

### File leksikon yang wajib ada
Sistem GAGAL saat labeling jika file berikut tidak ada di `database/`:
- `inset_lexicon_positive.csv` (kolom: `word`, `weight`)
- `inset_lexicon_negative.csv` (kolom: `word`, `weight`)

Tanpa file ini, sistem fallback ke keyword matching lama (metode yang tidak valid secara akademik).

---

## 7. Komponen Inti — Penjelasan Tacit

### 7.1 InSet Lexicon Loader (`SentimenRoute.py`)

```python
def load_inset_lexicon():
```

- Hanya di-load **sekali** dari disk, lalu di-cache di `current_app.config['INSET_POS']` dan `['INSET_NEG']`
- Kalau server di-restart, cache hilang dan file dibaca ulang otomatis
- Pada lingkungan multi-worker (gunicorn), setiap worker punya cache sendiri — tidak masalah untuk skripsi (single-worker)
- Kalau file CSV tidak ditemukan, fungsi akan throw exception yang di-catch di caller → fallback ke keyword matching

### 7.2 Fungsi `inset_label_sentiment()` (`SentimenRoute.py`)

```python
def inset_label_sentiment(text, pos_dict, neg_dict, threshold_pos=0.5, threshold_neg=-0.5):
```

**Logika inti:**
1. Split teks jadi token (lowercase, split spasi)
2. Cek tiap token di `pos_dict` dan `neg_dict`
3. Jumlahkan semua weight yang cocok → `total_score`
4. `total_score > threshold_pos` → positif
5. `total_score < threshold_neg` → negatif
6. Selain itu → netral
7. `confidence = min(abs(total_score) / 5.0, 1.0)`

**Hal yang perlu diketahui:**
- Satu token bisa masuk ke KEDUA list (pos dan neg) sekaligus — jarang terjadi tapi mungkin
- InSet menggunakan integer weight, range -5 s/d +5
- `total_score` bisa sangat besar/kecil jika teks panjang dengan banyak kata bermakna
- Default threshold: `+0.5` (positif) dan `-0.5` (negatif) — bisa diubah per user via `SentimentSettings`

### 7.3 Preprocessing Pipeline (`PreprocessingRoute.py`)

Fungsi utama: `preprocess_text(text, settings)` — dipanggil per tweet.

```
Urutan tahap:
1. cleansing_text()    → hapus URL, mention, hashtag, angka, tanda baca
2. case_folding()      → lowercase semua
3. tokenizing()        → split per spasi
4. stopword_removal()  → hapus kata dari tabel stopword_list (DB) atau default
5. normalization()     → ganti slang dengan kata baku (dari tabel normalization_dict)
6. stemming_ecs()      → Sastrawi stemmer (Enhanced Confix Stripping)
7. filter panjang kata → min_word_length ≤ len(token) ≤ max_word_length
```

**Hal yang perlu diketahui:**
- Stopword diambil dari DB setiap kali preprocessing dijalankan — tidak di-cache
- Jika tabel `stopword_list` kosong, digunakan **default stopword hardcoded** (38 kata)
- Jika tabel `normalization_dict` kosong, digunakan **default normalization hardcoded** (34 kata)
- `stemming_ecs()` menggunakan library Sastrawi (Bahasa Indonesia) — menggunakan ECS (Enhanced Confix Stripping)
- Setiap tahap hasilnya disimpan sebagai kolom terpisah di `text_preprocessing` → bisa dilihat di halaman detail

### 7.4 TF-IDF Conversion (`KonversiRoute.py`)

```python
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=0.01,
    max_df=0.95,
    lowercase=True,
    token_pattern=r'\b\w+\b'
)
```

**Hal yang perlu diketahui:**
- `min_df=0.01` artinya kata harus muncul di minimal 1% dokumen — penting untuk filter noise
- `max_df=0.95` artinya kata yang muncul di lebih dari 95% dokumen dihilangkan — filter kata terlalu umum
- `feature_vector` per dokumen disimpan sebagai JSON array float → ukuran bisa besar (1000 elemen per dokumen)
- `TfidfVocabulary` menyimpan `idf_score` per term — data ini yang dipakai untuk display vocabulary
- **Seluruh vocabulary di-reset tiap kali konversi dijalankan** (`TfidfVocabulary.query.filter_by(user_id).delete()`)

### 7.5 NBC Training & Testing (`NBCRoute.py`)

**Training:**
```python
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
# Simpan ke DB:
feature_log_prob  = model.feature_log_prob_.tolist()   # shape: (n_classes, n_features)
class_log_prior   = model.class_log_prior_.tolist()    # shape: (n_classes,)
classes           = model.classes_.tolist()             # e.g., ['negatif', 'netral', 'positif']
```

**Testing (rekonstruksi model dari DB):**
```python
model = MultinomialNB(alpha=alpha)
model.feature_log_prob_ = np.array(json.loads(model_data.feature_log_prob))
model.class_log_prior_  = np.array(json.loads(model_data.class_log_prior))
model.classes_          = np.array(json.loads(model_data.classes))
# Tidak perlu fit() — langsung predict()
```

**Hal yang perlu diketahui:**
- `MultinomialNB` membutuhkan nilai **non-negatif** di feature vector. TF-IDF selalu non-negatif → aman
- Kelas diurutkan alfabetik oleh sklearn: `['negatif', 'netral', 'positif']`
- `classification_report` disimpan sebagai JSON dict di kolom `nbc_model.classification_report`
- `predict_proba()` menghasilkan probabilitas per kelas, urutan sesuai `model.classes_`
- **`n_features` penting untuk konsistensi** — jika jumlah fitur berubah antara training dan testing, model AKAN error

### 7.6 Manual NBC Calculation (`NBCRoute.py`)

Fungsi `calculate_manual_naive_bayes()` menghitung NBC secara manual (tanpa sklearn) untuk **5 sampel testing pertama** saja — tujuannya untuk keperluan BAB III skripsi (menunjukkan proses matematis).

Hasil perhitungan manual ini ditampilkan di halaman `/nbc/results` sebagai bukti bahwa algoritma dipahami, bukan sekadar black box.

---

## 8. UI & Frontend — Hal yang Perlu Diketahui

### 8.1 Global Dialog System (`layout.html`)

Semua halaman yang extends `layout.html` mendapatkan `window.AppDialog` secara otomatis:

```javascript
// Confirm dialog
AppDialog.confirm("Pesan", "Judul", function() { /* onConfirm */ }, { type: 'danger' });

// Alert dialog
AppDialog.alert("Pesan", "Judul", "warning");
```

Tipe yang tersedia: `question` (biru), `warning` (oranye), `danger` (merah), `success` (hijau), `info` (biru muda).

### 8.2 Cara Menambah Confirm ke Form Baru

Tambahkan atribut `data-confirm` ke form — AppDialog akan intercept otomatis:

```html
<form method="POST" action="..."
      data-confirm="Pesan konfirmasi di sini"
      data-confirm-title="Judul Dialog"
      data-confirm-type="danger"
      data-confirm-text="Ya, Hapus">
```

Tidak perlu JavaScript tambahan. AppDialog di `layout.html` sudah handle semuanya.

### 8.3 Loading Overlay NBC (`nbc/index.html`)

Khusus halaman NBC, ada full-page loading overlay yang muncul saat split/train/test/reset. Overlay ini koordinasi dengan AppDialog:
- Pertama: AppDialog confirm dialog muncul
- Setelah dikonfirmasi: loading overlay muncul, form submit

**Penting:** Submit handler loading overlay HARUS cek `e.defaultPrevented`:
```javascript
form.addEventListener('submit', function(e) {
    if (!e.defaultPrevented) showNBCLoading(...);
});
```
Jika tidak, loading overlay akan muncul bahkan saat dialog belum dikonfirmasi.

### 8.4 Login Page

`login.html` adalah halaman **standalone** — tidak extends `layout.html`. Karena itu ia punya dialog system sendiri (`authDialogModal` + `showAuthDialog()`), terpisah dari `AppDialog` global.

---

## 9. Konfigurasi Default yang Penting

| Setting | Default | Lokasi |
|---------|---------|--------|
| InSet threshold positif | `+0.5` | `SentimenRoute.py` — bisa diubah per user di `sentiment_settings` |
| InSet threshold negatif | `-0.5` | Sama seperti di atas |
| TF-IDF max features | `1000` | Form konversi — bisa diubah user |
| TF-IDF min_df | `0.01` | Form konversi |
| TF-IDF max_df | `0.95` | Form konversi |
| NBC alpha (smoothing) | `1.0` | Form training — Laplace smoothing |
| Split test_size | `0.3` | Form split (70% train, 30% test) |
| Split random_state | `42` | Form split — untuk reproducibility |
| Min word length | `2` | `PreprocessingSettings` |
| Max word length | `50` | `PreprocessingSettings` |

---

## 10. Known Issues & Workarounds

### Issue 1: `sentiment_analysis.analyzed_by` vs `labeled_by`
Di `NBCRoute.py` fungsi `wordcloud()` (route `/nbc/wordcloud`), ada query yang menggunakan `analyzed_by` padahal kolom yang benar adalah `labeled_by`. Route ini jarang diakses langsung tapi perlu diperbaiki jika digunakan.

### Issue 2: Feature vector dimension mismatch
Jika TF-IDF di-reset dan di-generate ulang dengan parameter berbeda, jumlah fitur bisa berubah. Model NBC lama yang disimpan di DB akan error saat testing karena dimensi tidak cocok. **Solusi: selalu reset NBC setelah reset konversi.**

### Issue 3: Teks kosong setelah preprocessing
Beberapa tweet yang sangat pendek atau hanya berisi URL/mention bisa menghasilkan `final_text = ""` setelah preprocessing. TF-IDF akan skip dokumen ini. NBC mungkin tidak punya representasi untuk tweet tersebut. Bukan error fatal tapi akan ada gap antara jumlah sentimen dan jumlah TF-IDF.

### Issue 4: Stratified split membutuhkan minimal 1 sampel per kelas
Jika salah satu kelas sentimen hanya punya 1 data, `train_test_split` dengan `stratify=y` akan error. Pastikan distribusi data cukup (minimal 2–3 per kelas) sebelum split.

### Issue 5: InSet Lexicon tidak mencakup kata domain politik
InSet Lexicon adalah kamus umum — kata seperti "makan siang gratis", "kabinet merah putih" tidak ada. Tweet yang hanya membahas topik ini tanpa kata sentimen umum akan berlabel netral. Bisa diatasi dengan tambah kata domain manual ke file CSV leksikon.

---

## 11. Cara Menambah Fitur Baru

### Tambah route baru
1. Buat fungsi di file route yang relevan
2. Dekorasi dengan `@blueprint.route()` dan `@login_required`
3. Daftarkan di `app/routes/__init__.py` jika blueprint baru

### Tambah model/tabel baru
1. Buat class `db.Model` di file model yang sesuai
2. Import di `app/__init__.py` sebelum `db.create_all()`
3. Jalankan aplikasi — tabel dibuat otomatis

### Tambah halaman baru
1. Buat template HTML baru yang extend `layout.html`
2. Pastikan ada `{% block content %}` dan `{% block extra_js %}`
3. Template sudah dapat AppDialog, sidebar, dan navbar otomatis

---

## 12. Referensi Penting

| Referensi | URL/Lokasi |
|-----------|------------|
| InSet Lexicon source | `github.com/Abaddon-Beza/InSet` |
| Kamus Alay source | `github.com/nasalsabila/kamus-alay` |
| Sastrawi (stemmer) | PyPI: `PySastrawi` |
| scikit-learn MultinomialNB | `sklearn.naive_bayes.MultinomialNB` |
| PRD lengkap | `PRD.md` di root proyek |
| Design system | `design.md` di root proyek (jika ada) |

---

## 13. Checklist Deploy / Serah Terima

- [ ] File `database/inset_lexicon_positive.csv` ada dan terbaca pandas
- [ ] File `database/inset_lexicon_negative.csv` ada dan terbaca pandas
- [ ] File `database/kamus_alay.csv` ada (untuk import normalisasi alay)
- [ ] File `database/dictionary_baku_nonbaku.csv` ada (untuk import KBBI)
- [ ] Database `db_naive_bayes` sudah dibuat di MySQL
- [ ] Semua tabel terbuat otomatis saat `python run.py` pertama kali
- [ ] User terdaftar di tabel `users` (perlu insert manual atau via Auth route)
- [ ] Pipeline end-to-end berjalan: scraping → preprocessing → sentimen → konversi → NBC → evaluasi
- [ ] Distribusi label InSet tidak ada kelas < 5% (cek di halaman Sentimen)
- [ ] Confusion matrix NBC tidak ada baris/kolom all-zero (cek di /nbc/results)
- [ ] Akurasi NBC ≥ 70%
