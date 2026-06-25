# Product Requirements Document
## Perbaikan Sistem Analisis Sentimen `web_naive_bayes`

| | |
|---|---|
| **Versi** | 2.0 |
| **Tanggal** | 25 Juni 2026 |
| **Repository** | `github.com/Khairul122/web_naive_bayes` |
| **Framework** | Python Flask + MySQL + Bootstrap 5 (Star Admin2) |
| **Status** | Aktif — Revisi Skripsi Adetia Irvanda |
| **Penulis** | Khairul |

---

## Daftar Isi

1. [Latar Belakang](#1-latar-belakang)
2. [Kondisi Sistem Saat Ini](#2-kondisi-sistem-saat-ini)
3. [Tujuan Perbaikan](#3-tujuan-perbaikan)
4. [Ruang Lingkup](#4-ruang-lingkup)
5. [Arsitektur Sistem](#5-arsitektur-sistem)
6. [Spesifikasi Fungsional](#6-spesifikasi-fungsional)
7. [Spesifikasi Database](#7-spesifikasi-database)
8. [Spesifikasi API dan Route](#8-spesifikasi-api-dan-route)
9. [Spesifikasi UI](#9-spesifikasi-ui)
10. [Spesifikasi Non-Fungsional](#10-spesifikasi-non-fungsional)
11. [Struktur File](#11-struktur-file)
12. [Rencana Implementasi](#12-rencana-implementasi)
13. [Kriteria Penerimaan](#13-kriteria-penerimaan)
14. [Risiko dan Mitigasi](#14-risiko-dan-mitigasi)
15. [Referensi dan Sitasi](#15-referensi-dan-sitasi)

---

## 1. Latar Belakang

### 1.1 Deskripsi Proyek

`web_naive_bayes` adalah aplikasi web berbasis Flask untuk analisis sentimen masyarakat terhadap Kabinet Prabowo Subianto menggunakan data Twitter/X. Aplikasi ini merupakan implementasi sistem untuk skripsi mahasiswa Teknik Informatika Universitas Malikussaleh dengan judul **"Analisis Sentimen Masyarakat Terhadap Pemerintah di Era Kabinet Prabowo Subianto berdasarkan Sosial Media X menggunakan Naïve Bayes Classifier"**.

### 1.2 Masalah yang Ditemukan

Berdasarkan audit kode yang dilakukan pada 25 Juni 2026, ditemukan **satu masalah utama** yang memengaruhi validitas akademik seluruh penelitian:

**Metode pelabelan data (ground truth) di `app/routes/SentimenRoute.py` menggunakan keyword matching sederhana**, bukan leksikon sentimen berbobot yang dapat disitasi secara ilmiah.

Kode bermasalah saat ini (fungsi `auto_label_sentiment()`):

```python
# KONDISI SAAT INI — BERMASALAH
neutral_keywords = [
    'mungkin', 'sepertinya', 'kayaknya', 'agak', 'cukup', 'lumayan',
    'akan', 'sedang', 'lagi', 'sudah', 'telah', 'pernah', 'belum', 'masih'
]
```

Kata-kata seperti `sudah`, `lagi`, `masih`, `akan` muncul di hampir semua tweet, sehingga hampir semua data terlabeli `netral` secara otomatis.

### 1.3 Dampak Masalah

| Dampak | Bukti di Kode/Data |
|--------|-------------------|
| Distribusi label tidak seimbang | 77,7% netral, 13,1% negatif, 9,2% positif |
| Confusion matrix tidak valid | Kolom `negatif` dan `netral` bernilai 0 semua |
| NBC hanya memprediksi satu kelas | Model cenderung selalu memprediksi `positif` |
| Tidak ada sitasi akademik | Keyword dibuat manual tanpa referensi |
| Narasi BAB III tidak dapat dipertahankan | Tidak ada landasan teori untuk metode ini |

### 1.4 Komponen yang Sudah Benar

Perbaikan **hanya** difokuskan pada masalah di atas. Komponen berikut sudah berfungsi dengan benar dan tidak diubah:

| Komponen | File | Status |
|----------|------|--------|
| Scraping Twitter | `ScrappingRoute.py` | ✓ Tidak diubah |
| Preprocessing teks | `PreprocessingRoute.py` | ✓ Minor — tambah kamus alay |
| Pembobotan TF-IDF | `KonversiRoute.py` | ✓ Tidak diubah |
| NBC Training & Testing | `NBCRoute.py` | ✓ Tidak diubah |
| Evaluasi & Confusion Matrix | `NBCRoute.py` | ✓ Tidak diubah |
| Manual NBC Calculation | `NBCRoute.py` | ✓ Tidak diubah |

---

## 2. Kondisi Sistem Saat Ini

### 2.1 Alur Pipeline Saat Ini

```
TwitterScraping  →  TextPreprocessing  →  SentimentAnalysis     →  TfidfConversion  →  NBC
(scraping)          (cleansing, stem)      (keyword matching)       (TF-IDF)            (training/testing)
                                           ← MASALAH DI SINI →
```

### 2.2 Alur Pipeline Target Setelah Perbaikan

```
TwitterScraping  →  TextPreprocessing     →  SentimentAnalysis    →  TfidfConversion  →  NBC
(scraping)          (+ kamus alay baru)      (InSet Lexicon)          (TF-IDF)            (training/testing)
                    ← Perbaikan Minor →      ← Perbaikan Utama →
```

### 2.3 Struktur Database Saat Ini

Tabel-tabel yang sudah ada dan **tidak akan diubah skemanya**:

| Tabel | Kunci Kolom Relevan |
|-------|---------------------|
| `twitter_scraping` | `id`, `full_text`, `username`, `scraped_by` |
| `text_preprocessing` | `id`, `tweet_id`, `final_text`, `processed_by` |
| `sentiment_analysis` | `id`, `tweet_id`, `sentiment_label`, `confidence_score`, `positive_keywords`, `negative_keywords`, `neutral_keywords`, `labeling_method`, `labeled_by` |
| `tfidf_conversion` | `id`, `sentiment_id`, `feature_vector`, `converted_by` |
| `nbc_training` | `id`, `conversion_id`, `feature_vector`, `label` |
| `nbc_testing` | `id`, `conversion_id`, `true_label`, `predicted_label`, `is_correct` |
| `nbc_model` | `id`, `accuracy`, `precision_score`, `recall_score`, `f1_score` |
| `normalization_dict` | `id`, `slang_word`, `standard_word`, `is_active` |
| `preprocessing_settings` | `id`, `user_id`, semua flag boolean |

---

## 3. Tujuan Perbaikan

### 3.1 Tujuan Utama

| # | Tujuan | Indikator Keberhasilan |
|---|--------|------------------------|
| T-01 | Mengganti metode labeling dari keyword matching ke InSet Lexicon | Fungsi `auto_label_sentiment()` dihapus, diganti `inset_label_sentiment()` |
| T-02 | Memperbaiki distribusi label agar tidak bias netral | Tidak ada kelas dengan proporsi < 5% dari total data |
| T-03 | Menghasilkan confusion matrix yang valid (non-zero di semua sel) | Semua 9 sel confusion matrix 3×3 terisi nilai > 0 |
| T-04 | Menambahkan normalisasi kata alay Twitter | Kamus alay ter-import ke `normalization_dict` |
| T-05 | Menjadikan metode labeling dapat disitasi di skripsi | BAB III mencantumkan referensi InSet Lexicon (Koto & Rahmaningtyas, 2017) |

### 3.2 Tujuan Sekunder

| # | Tujuan | Indikator Keberhasilan |
|---|--------|------------------------|
| T-06 | Menyediakan konfigurasi threshold labeling per user | Tabel `sentiment_settings` dibuat dan berfungsi |
| T-07 | Menampilkan skor InSet di tabel hasil | Kolom `InSet Score` dan `Kata Cocok` tampil di halaman Sentimen |

---

## 4. Ruang Lingkup

### 4.1 Dalam Lingkup (In Scope)

- Penggantian fungsi `auto_label_sentiment()` di `SentimenRoute.py` dengan implementasi InSet Lexicon
- Penambahan dua file leksikon InSet ke `database/`
- Penambahan kamus alay ke `database/` dan route import baru di `PreprocessingRoute.py`
- Penambahan model `SentimentSettings` di `SentimenModel.py`
- Penambahan route `POST /sentimen/save_settings` di `SentimenRoute.py`
- Penambahan route `POST /preprocessing/import_alay` di `PreprocessingRoute.py`
- Penyesuaian tampilan halaman `sentimen/index.html` (form threshold + kolom skor)
- Registrasi model baru ke `db.create_all()` melalui `app/__init__.py`

### 4.2 Di Luar Lingkup (Out of Scope)

- Perubahan arsitektur Flask, konfigurasi `app/__init__.py`, atau `extension.py`
- Perubahan skema tabel yang sudah ada (`sentiment_analysis`, `normalization_dict`, dll)
- Perubahan pada `NBCRoute.py`, `KonversiRoute.py`, `ScrappingRoute.py`
- Scraping ulang dataset dari nol
- Perubahan algoritma NBC ke metode lain
- Perubahan sistem autentikasi (`AuthRoute.py`, `AuthModel.py`)
- Perubahan CSS/tema visual (`theme-admin.css`, `style.css`, `design.md`)

---

## 5. Arsitektur Sistem

### 5.1 Stack Teknologi (Tidak Berubah)

| Layer | Teknologi |
|-------|-----------|
| Backend | Python 3, Flask, SQLAlchemy |
| Database | MySQL (`db_naive_bayes`) |
| Frontend | Bootstrap 5, Star Admin2, Poppins font, Font Awesome 6 |
| ML Library | scikit-learn (`MultinomialNB`, `TfidfVectorizer`, `train_test_split`) |
| NLP | Sastrawi (stemming), custom stopword list |

### 5.2 Blueprint yang Sudah Terdaftar di `routes/__init__.py`

```python
auth_bp          # url_prefix='/auth'   — tidak diubah
dashboard_bp     # url_prefix='/'       — tidak diubah
scrapping_bp     # url_prefix='/scrapping'  — tidak diubah
preprocessing_bp # url_prefix='/preprocessing' — ditambah 1 route
sentimen_bp      # url_prefix='/sentimen'    — ditambah 2 route, ubah 1 fungsi
konversi_bp      # url_prefix='/konversi'    — tidak diubah
nbc_bp           # url_prefix='/nbc'         — tidak diubah
```

---

## 6. Spesifikasi Fungsional

### 6.1 F-01 — Implementasi InSet Lexicon sebagai Labeling Engine

**File yang diubah:** `app/routes/SentimenRoute.py`

#### 6.1.1 Deskripsi

Fungsi `auto_label_sentiment(text)` yang saat ini menggunakan keyword matching sederhana diganti seluruhnya dengan fungsi `inset_label_sentiment(text, pos_dict, neg_dict, threshold_pos, threshold_neg)` yang menggunakan InSet Lexicon — kamus kata berbobot sentimen bahasa Indonesia yang dipublikasikan secara ilmiah.

#### 6.1.2 Cara Kerja InSet Lexicon

InSet Lexicon memetakan kata bahasa Indonesia ke skor sentimen:
- **Leksikon Positif:** skor `+1` hingga `+5` (makin besar = makin positif)
- **Leksikon Negatif:** skor `-1` hingga `-5` (makin kecil = makin negatif)
- Kata yang tidak ada di kedua leksikon = skor `0`

Contoh entri:
```
# inset_lexicon_positive.csv
word,weight
bagus,3
luar biasa,5
keren,2
setuju,2

# inset_lexicon_negative.csv
word,weight
buruk,-3
korupsi,-5
gagal,-3
kecewa,-2
```

#### 6.1.3 Logika Scoring

```
skor_total = Σ weight(token) untuk setiap token dalam teks yang ada di leksikon

Contoh: teks = "prabowo sudah maju keren tapi korupsi masih ada"
Token yang cocok: keren (+2), korupsi (-5)
skor_total = 2 + (-5) = -3

if skor_total > threshold_pos (+0.5)  → label = 'positif'
if skor_total < threshold_neg (-0.5)  → label = 'negatif'
else (-0.5 ≤ skor_total ≤ +0.5)      → label = 'netral'

Hasil: skor -3 < -0.5 → label = 'negatif'
```

#### 6.1.4 Cara Hitung `confidence_score`

```python
# Normalisasi skor ke rentang 0.0–1.0
# Skor InSet maksimal adalah 5 (atau -5)
abs_score    = abs(skor_total)
confidence   = min(round(abs_score / 5.0, 2), 1.0)

# Contoh:
# skor_total = -3  → confidence = min(3/5, 1.0) = 0.6
# skor_total = 0   → confidence = 0.0
# skor_total = 5   → confidence = 1.0
# skor_total = 7   → confidence = min(7/5, 1.0) = 1.0 (dibatasi)
```

#### 6.1.5 Output Fungsi

Output harus kompatibel dengan cara penyimpanan di route `auto_label` (baris 78–94 `SentimenRoute.py`):

```python
def inset_label_sentiment(text, pos_dict, neg_dict, threshold_pos=0.5, threshold_neg=-0.5):
    """
    pos_dict : dict {word: weight}  — dari inset_lexicon_positive.csv
    neg_dict : dict {word: weight}  — dari inset_lexicon_negative.csv
    """
    tokens = text.lower().split()

    positive_found = []
    negative_found = []

    for token in tokens:
        if token in pos_dict:
            positive_found.append(token)
        if token in neg_dict:
            negative_found.append(token)

    pos_score = sum(pos_dict[w] for w in positive_found)
    neg_score = sum(neg_dict[w] for w in negative_found)
    total_score = pos_score + neg_score

    abs_score = abs(total_score)
    confidence = min(round(abs_score / 5.0, 2), 1.0)

    if total_score > threshold_pos:
        label = 'positif'
    elif total_score < threshold_neg:
        label = 'negatif'
    else:
        label = 'netral'

    return {
        'label'            : label,           # → sentiment_label
        'confidence'       : confidence,       # → confidence_score
        'positive_keywords': positive_found,   # → positive_keywords (join ', ')
        'negative_keywords': negative_found,   # → negative_keywords (join ', ')
        'neutral_keywords' : [],               # → neutral_keywords (selalu kosong)
    }
```

> **Penting:** Key `label`, `confidence`, `positive_keywords`, `negative_keywords`, `neutral_keywords` harus tetap ada karena dipakai langsung di route `auto_label` untuk membangun objek `SentimentAnalysis`. Tidak ada field baru yang ditambahkan agar skema DB tidak berubah.

#### 6.1.6 Fungsi Loader Leksikon

```python
def load_inset_lexicon():
    """
    Load InSet dari file CSV ke dalam dict Python.
    Dipanggil sekali, disimpan di current_app.config agar tidak reload tiap request.
    """
    import pandas as pd
    from flask import current_app
    import os

    if 'INSET_POS' not in current_app.config:
        base = os.path.join(current_app.root_path, '..', 'database')
        pos_df = pd.read_csv(os.path.join(base, 'inset_lexicon_positive.csv'))
        neg_df = pd.read_csv(os.path.join(base, 'inset_lexicon_negative.csv'))
        current_app.config['INSET_POS'] = dict(zip(pos_df['word'], pos_df['weight']))
        current_app.config['INSET_NEG'] = dict(zip(neg_df['word'], neg_df['weight']))

    return current_app.config['INSET_POS'], current_app.config['INSET_NEG']
```

#### 6.1.7 Modifikasi Route `auto_label` yang Sudah Ada

Di route `POST /sentimen/auto_label`, ganti baris:
```python
# HAPUS:
sentiment_result = auto_label_sentiment(text_to_analyze)

# GANTI DENGAN:
pos_dict, neg_dict = load_inset_lexicon()
settings = SentimentSettings.query.filter_by(user_id=user_id).first()
t_pos = settings.threshold_pos if settings else 0.5
t_neg = settings.threshold_neg if settings else -0.5
sentiment_result = inset_label_sentiment(text_to_analyze, pos_dict, neg_dict, t_pos, t_neg)
```

---

### 6.2 F-02 — Integrasi Kamus Alay ke Normalisasi Preprocessing

**File yang diubah:** `app/routes/PreprocessingRoute.py`
**File baru:** `database/kamus_alay.csv`

#### 6.2.1 Deskripsi

Kamus alay Twitter ditambahkan ke tabel `normalization_dict` melalui route import baru. Kamus ini berisi kata-kata tidak baku yang umum dipakai di tweet bahasa Indonesia, seperti singkatan dan ejaan informal, yang **tidak tercakup** oleh `dictionary_baku_nonbaku.csv` (yang berisi koreksi kata formal KBBI, bukan slang Twitter).

#### 6.2.2 Mengapa Kamus Alay Penting

| Kata di Tweet | Ada di KBBI dict | Ada di kamus alay |
|---------------|-----------------|-------------------|
| `gak`, `ga`, `nggak` | ✗ Tidak | ✓ → `tidak` |
| `bgt` | ✗ Tidak | ✓ → `banget` |
| `yg` | ✗ Tidak | ✓ → `yang` |
| `dgn` | ✗ Tidak | ✓ → `dengan` |
| `korupsi` | ✓ Ya (kata baku) | — |

Tanpa kamus alay, kata-kata seperti `gak bagus` tidak cocok dengan kata `bagus` di InSet Lexicon karena teks sebelum normalisasi mungkin mengandung `gk bagus` atau `ga bagus`.

#### 6.2.3 Format File `kamus_alay.csv`

Kolom berbeda dari `dictionary_baku_nonbaku.csv` yang ada. Route import terpisah dibutuhkan.

```csv
slang,baku
gak,tidak
ga,tidak
nggak,tidak
ngga,tidak
bgt,banget
bngt,banget
yg,yang
dgn,dengan
utk,untuk
krn,karena
tp,tapi
tpi,tapi
udh,sudah
udah,sudah
sdh,sudah
blm,belum
blom,belum
sm,sama
jg,juga
lg,lagi
klo,kalau
klu,kalau
gmn,bagaimana
gimana,bagaimana
emg,memang
emang,memang
nih,ini
tuh,itu
gue,saya
gw,saya
lu,kamu
lo,kamu
deh,lah
dong,lah
sih,lah
beneran,benar
bner,benar
bnr,benar
mantul,mantap betul
```

**Sumber:** `github.com/nasalsabila/kamus-alay` — diunduh dan dikonversi ke format `slang,baku`.

#### 6.2.4 Route Import Baru

```python
@preprocessing_bp.route('/import_alay', methods=['POST'])
@login_required
def import_alay():
    """Import kamus alay dari database/kamus_alay.csv ke NormalizationDict."""
    from flask import current_app
    import pandas as pd

    csv_path = os.path.join(
        os.path.dirname(current_app.root_path), 'database', 'kamus_alay.csv'
    )

    if not os.path.exists(csv_path):
        flash('File kamus_alay.csv tidak ditemukan di folder database/', 'danger')
        return redirect(url_for('preprocessing.index'))

    df = pd.read_csv(csv_path)
    imported_count = 0
    skipped_count  = 0

    for _, row in df.iterrows():
        slang  = str(row['slang']).strip().lower()
        baku   = str(row['baku']).strip().lower()

        if not slang or not baku or slang == 'nan' or baku == 'nan':
            skipped_count += 1
            continue
        if slang == baku:
            skipped_count += 1
            continue

        existing = NormalizationDict.query.filter_by(slang_word=slang).first()
        if existing:
            skipped_count += 1
            continue

        entry = NormalizationDict(
            slang_word    = slang,
            standard_word = baku,
            category      = 'alay',    # membedakan dari entri KBBI
            is_active     = True
        )
        db.session.add(entry)
        imported_count += 1

    db.session.commit()
    flash(
        f'Import kamus alay berhasil! {imported_count} kata ditambahkan, '
        f'{skipped_count} dilewati (duplikat/kosong).',
        'success'
    )
    return redirect(url_for('preprocessing.index'))
```

> **Catatan:** Fungsi `normalization()` yang sudah ada di `PreprocessingRoute.py` (baris 418–420) sudah membaca **semua** entri `NormalizationDict.query.filter_by(is_active=True)`, sehingga entri kamus alay yang baru di-import **langsung digunakan** tanpa perlu mengubah logika normalisasi.

---

### 6.3 F-03 — Model dan Route Konfigurasi Threshold

**File yang diubah:** `app/models/SentimenModel.py`
**File yang diubah:** `app/routes/SentimenRoute.py`

#### 6.3.1 Deskripsi

Threshold labeling (`threshold_pos`, `threshold_neg`) dan pilihan engine labeling (`inset` atau `keyword`) disimpan per user di tabel baru `sentiment_settings`, mengikuti pola yang sudah ada di `PreprocessingSettings`.

#### 6.3.2 Class Model Baru

Ditambahkan di akhir file `app/models/SentimenModel.py`:

```python
class SentimentSettings(db.Model):
    __tablename__ = 'sentiment_settings'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(
                          db.Integer,
                          db.ForeignKey('users.id_user'),
                          nullable=False,
                          unique=True        # satu baris per user
                      )
    threshold_pos   = db.Column(db.Float, default=0.5)
    threshold_neg   = db.Column(db.Float, default=-0.5)
    labeling_engine = db.Column(
                          db.Enum('inset', 'keyword'),
                          default='inset'
                      )
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at      = db.Column(
                          db.DateTime,
                          default=datetime.utcnow,
                          onupdate=datetime.utcnow
                      )

    def __repr__(self):
        return f'<SentimentSettings user={self.user_id} pos={self.threshold_pos}>'
```

#### 6.3.3 Import di Route

Tambahkan import di `SentimenRoute.py`:
```python
from app.models.SentimenModel import SentimentAnalysis, SentimentSettings
```

#### 6.3.4 Route Simpan Pengaturan

```python
@sentimen_bp.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    user_id      = session['user_id']
    threshold_pos = float(request.form.get('threshold_pos', 0.5))
    threshold_neg = float(request.form.get('threshold_neg', -0.5))
    engine        = request.form.get('labeling_engine', 'inset')

    # Validasi nilai threshold
    if threshold_pos <= 0:
        flash('Threshold positif harus lebih besar dari 0', 'warning')
        return redirect(url_for('sentimen.index'))
    if threshold_neg >= 0:
        flash('Threshold negatif harus lebih kecil dari 0', 'warning')
        return redirect(url_for('sentimen.index'))

    settings = SentimentSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = SentimentSettings(user_id=user_id)
        db.session.add(settings)

    settings.threshold_pos   = threshold_pos
    settings.threshold_neg   = threshold_neg
    settings.labeling_engine = engine
    db.session.commit()

    flash('Pengaturan labeling berhasil disimpan', 'success')
    return redirect(url_for('sentimen.index'))
```

---

### 6.4 F-04 — Tampilan Skor InSet di Halaman Sentimen

**File yang diubah:** `app/routes/SentimenRoute.py` (route `index`)
**File yang diubah:** `app/templates/sentimen/index.html`

#### 6.4.1 Deskripsi

Halaman Sentimen menampilkan dua kolom tambahan di tabel: skor InSet yang direkonstruksi dan daftar kata yang cocok. Data ini direkonstruksi dari kolom `positive_keywords` dan `negative_keywords` yang sudah tersimpan di DB — **tanpa mengubah skema tabel**.

#### 6.4.2 Rekonstruksi Skor di Route `index`

Di route `GET /sentimen/`, sebelum `render_template`, tambahkan:

```python
# Load leksikon untuk rekonstruksi skor tampilan
try:
    pos_dict, neg_dict = load_inset_lexicon()
except Exception:
    pos_dict, neg_dict = {}, {}

# Tambahkan skor ke setiap item pagination
enriched_items = []
for sentiment, tweet in sentiment_data.items:
    pos_words = [w.strip() for w in (sentiment.positive_keywords or '').split(',') if w.strip()]
    neg_words = [w.strip() for w in (sentiment.negative_keywords or '').split(',') if w.strip()]
    score = sum(pos_dict.get(w, 0) for w in pos_words) + \
            sum(neg_dict.get(w, 0) for w in neg_words)
    all_words = pos_words + neg_words
    enriched_items.append({
        'sentiment' : sentiment,
        'tweet'     : tweet,
        'inset_score': round(score, 2),
        'matched_words': ', '.join(all_words) if all_words else '—'
    })
```

---

## 7. Spesifikasi Database

### 7.1 Tabel Baru: `sentiment_settings`

```sql
CREATE TABLE `sentiment_settings` (
  `id`              INT(11)      NOT NULL AUTO_INCREMENT,
  `user_id`         INT(11)      NOT NULL,
  `threshold_pos`   FLOAT        NOT NULL DEFAULT 0.5,
  `threshold_neg`   FLOAT        NOT NULL DEFAULT -0.5,
  `labeling_engine` ENUM('inset','keyword') NOT NULL DEFAULT 'inset',
  `created_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP,
  `updated_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_sentiment_settings_user` (`user_id`),
  FOREIGN KEY (`user_id`) REFERENCES `users`(`id_user`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 7.2 Tabel yang Tidak Berubah

Skema `sentiment_analysis` dipertahankan sepenuhnya:

```sql
-- TIDAK DIUBAH — referensi saja
CREATE TABLE `sentiment_analysis` (
  `id`               INT(11)  NOT NULL AUTO_INCREMENT,
  `tweet_id`         INT(11)  NOT NULL,
  `preprocessing_id` INT(11),
  `username`         VARCHAR(100),
  `tweet_text`       TEXT     NOT NULL,
  `processed_text`   TEXT,
  `sentiment_label`  ENUM('positif','negatif','netral'),
  `confidence_score` FLOAT    DEFAULT 0.0,
  `positive_keywords` TEXT,
  `negative_keywords` TEXT,
  `neutral_keywords`  TEXT,
  `labeling_method`  ENUM('auto','manual') DEFAULT 'auto',
  `labeled_at`       DATETIME,
  `labeled_by`       INT(11)  NOT NULL,
  `created_at`       DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
);
```

### 7.3 Pembuatan Tabel Baru

Tabel `sentiment_settings` dibuat otomatis oleh SQLAlchemy saat `db.create_all()` dipanggil. Tidak perlu menjalankan SQL manual, cukup pastikan `SentimentSettings` di-import sebelum `create_all`.

Tambahkan di `app/__init__.py` setelah baris `from app.routes import register_routes`:

```python
with app.app_context():
    from app.models.SentimenModel import SentimentSettings  # pastikan model dikenal
    db.create_all()
```

---

## 8. Spesifikasi API dan Route

### 8.1 Daftar Lengkap Route Setelah Perbaikan

#### Blueprint `sentimen_bp` (`/sentimen`)

| Method | Path | Fungsi | Perubahan |
|--------|------|--------|-----------|
| GET | `/sentimen/` | `index()` | **Diubah:** tambah rekonstruksi skor InSet |
| POST | `/sentimen/auto_label` | `auto_label()` | **Diubah:** ganti ke InSet Lexicon |
| POST | `/sentimen/save_settings` | `save_settings()` | **Baru** |
| POST | `/sentimen/reset` | `reset()` | Tidak berubah |
| GET | `/sentimen/export` | `export()` | Tidak berubah |

#### Blueprint `preprocessing_bp` (`/preprocessing`)

| Method | Path | Fungsi | Perubahan |
|--------|------|--------|-----------|
| GET | `/preprocessing/` | `index()` | Tidak berubah |
| POST | `/preprocessing/process` | `process_texts()` | Tidak berubah |
| GET/POST | `/preprocessing/settings` | `settings()` | Tidak berubah |
| POST | `/preprocessing/import_kbbi` | `import_kbbi()` | Tidak berubah |
| POST | `/preprocessing/import_alay` | `import_alay()` | **Baru** |
| POST | `/preprocessing/reset` | `reset()` | Tidak berubah |
| POST | `/preprocessing/reset_normalization` | `reset_normalization()` | Tidak berubah |
| GET | `/preprocessing/export` | `export()` | Tidak berubah |

### 8.2 Detail Request/Response

#### `POST /sentimen/save_settings`

**Request (form-data):**

| Field | Tipe | Wajib | Default | Validasi |
|-------|------|-------|---------|----------|
| `threshold_pos` | float | Tidak | `0.5` | Harus > 0 |
| `threshold_neg` | float | Tidak | `-0.5` | Harus < 0 |
| `labeling_engine` | string | Tidak | `inset` | Enum: `inset` \| `keyword` |

**Response:** Redirect ke `GET /sentimen/` dengan flash message sukses/warning.

#### `POST /preprocessing/import_alay`

**Request:** Tidak membutuhkan body — membaca langsung dari `database/kamus_alay.csv`.

**Response:** Redirect ke `GET /preprocessing/` dengan flash message jumlah kata yang berhasil diimport.

---

## 9. Spesifikasi UI

### 9.1 Halaman Sentimen (`sentimen/index.html`)

Halaman ini menggunakan template yang extend `layout.html` dengan Bootstrap 5 dan Star Admin2. Semua penambahan mengikuti design system di `design.md`.

#### 9.1.1 Tambahan: Card Pengaturan Labeling

Tambahkan card baru di atas tabel, mengikuti pola card header dekoratif yang ada:

```html
<div class="card shadow-lg border-0 mb-4">
  <div class="card-header bg-gradient-info text-white py-3">
    <h5 class="mb-0 fw-bold">
      <i class="fas fa-sliders-h"></i> Konfigurasi Labeling
    </h5>
  </div>
  <div class="card-body p-4">
    <form method="POST" action="{{ url_for('sentimen.save_settings') }}">
      <div class="row g-3 align-items-end">

        <div class="col-md-3">
          <label class="form-label fw-semibold">Engine Labeling</label>
          <select name="labeling_engine" class="form-select">
            <option value="inset"
              {% if settings and settings.labeling_engine == 'inset' %}selected{% endif %}>
              InSet Lexicon
            </option>
            <option value="keyword"
              {% if settings and settings.labeling_engine == 'keyword' %}selected{% endif %}>
              Keyword Matching
            </option>
          </select>
        </div>

        <div class="col-md-2">
          <label class="form-label fw-semibold">Threshold Positif</label>
          <input type="number" name="threshold_pos" step="0.1" min="0.1" max="5"
                 class="form-control"
                 value="{{ settings.threshold_pos if settings else 0.5 }}">
        </div>

        <div class="col-md-2">
          <label class="form-label fw-semibold">Threshold Negatif</label>
          <input type="number" name="threshold_neg" step="0.1" min="-5" max="-0.1"
                 class="form-control"
                 value="{{ settings.threshold_neg if settings else -0.5 }}">
        </div>

        <div class="col-md-2">
          <button type="submit" class="btn btn-primary w-100">
            <i class="fas fa-save"></i> Simpan
          </button>
        </div>

        <div class="col-md-3">
          <form method="POST" action="{{ url_for('sentimen.auto_label') }}">
            <button type="submit" class="btn btn-success w-100">
              <i class="fas fa-tags"></i> Jalankan Auto-Labeling
            </button>
          </form>
        </div>

      </div>
    </form>
  </div>
</div>
```

#### 9.1.2 Tambahan: Kolom InSet Score di Tabel

Tambahkan dua kolom di `<thead>` dan `<tbody>` tabel hasil sentimen:

```html
<!-- Di <thead> — tambahkan setelah kolom Confidence -->
<th>InSet Score</th>
<th>Kata Cocok</th>

<!-- Di <tbody> — tambahkan untuk setiap baris -->
<td>
  <span class="badge
    {% if item.inset_score > 0 %}bg-success
    {% elif item.inset_score < 0 %}bg-danger
    {% else %}bg-secondary{% endif %}">
    {{ item.inset_score }}
  </span>
</td>
<td>
  <small class="text-muted">{{ item.matched_words }}</small>
</td>
```

#### 9.1.3 Tambahan: Tombol Import Kamus Alay di Halaman Preprocessing

Di `preprocessing/index.html`, tambahkan tombol di area normalisasi (di dekat tombol import KBBI):

```html
<form method="POST" action="{{ url_for('preprocessing.import_alay') }}"
      class="d-inline">
  <button type="submit" class="btn btn-outline-primary btn-sm">
    <i class="fas fa-upload"></i> Import Kamus Alay
  </button>
</form>
```

---

## 10. Spesifikasi Non-Fungsional

| Kategori | Requirement | Penjelasan |
|----------|-------------|------------|
| **Kompatibilitas DB** | Tidak mengubah skema tabel yang sudah ada | Hanya tabel baru `sentiment_settings` yang ditambahkan |
| **Kompatibilitas kode** | Output `inset_label_sentiment()` identik strukturnya dengan output `auto_label_sentiment()` | Key dict yang sama: `label`, `confidence`, `positive_keywords`, `negative_keywords`, `neutral_keywords` |
| **Performa leksikon** | InSet Lexicon hanya di-load sekali, di-cache di `current_app.config` | Menghindari I/O disk tiap request labeling |
| **Fallback leksikon** | Jika file InSet tidak ditemukan, fallback ke keyword matching lama | Error tidak boleh merusak seluruh pipeline |
| **Backward compatibility** | `labeling_method` di tabel `sentiment_analysis` tetap `'auto'` | Tidak perlu migrasi data lama |
| **Konsistensi tema** | Komponen UI menggunakan class Bootstrap dan design system yang ada (`design.md`) | Tidak ada warna hex baru, tidak ada CSS inline |
| **Validasi threshold** | Threshold positif harus > 0, threshold negatif harus < 0 | Dicek di server sebelum disimpan |

---

## 11. Struktur File

### 11.1 File yang Diubah

```
app/
├── models/
│   └── SentimenModel.py          DIUBAH: tambah class SentimentSettings
├── routes/
│   ├── SentimenRoute.py          DIUBAH: ganti auto_label_sentiment(),
│   │                                     tambah load_inset_lexicon(),
│   │                                     tambah inset_label_sentiment(),
│   │                                     tambah route save_settings,
│   │                                     ubah route index (rekonstruksi skor)
│   └── PreprocessingRoute.py     DIUBAH: tambah route import_alay
└── templates/
    ├── sentimen/
    │   └── index.html            DIUBAH: card konfigurasi + kolom skor
    └── preprocessing/
        └── index.html            DIUBAH: tombol import kamus alay
```

### 11.2 File Baru

```
database/
├── inset_lexicon_positive.csv    BARU — kolom: word, weight (skor positif)
├── inset_lexicon_negative.csv    BARU — kolom: word, weight (skor negatif)
└── kamus_alay.csv                BARU — kolom: slang, baku
```

### 11.3 File Tidak Diubah

```
app/
├── __init__.py                   TIDAK BERUBAH (kecuali tambah import model)
├── extension.py                  TIDAK BERUBAH
├── models/
│   ├── AuthModel.py              TIDAK BERUBAH
│   ├── ScrappingModel.py         TIDAK BERUBAH
│   ├── PreprocessingModel.py     TIDAK BERUBAH
│   ├── KonversiModel.py          TIDAK BERUBAH
│   └── NBCModel.py               TIDAK BERUBAH
├── routes/
│   ├── AuthRoute.py              TIDAK BERUBAH
│   ├── DashboardRoute.py         TIDAK BERUBAH
│   ├── ScrappingRoute.py         TIDAK BERUBAH
│   ├── KonversiRoute.py          TIDAK BERUBAH
│   └── NBCRoute.py               TIDAK BERUBAH
database/
└── dictionary_baku_nonbaku.csv   TIDAK BERUBAH
```

---

## 12. Rencana Implementasi

### Sprint 1 — Persiapan File Leksikon (Estimasi: 1–2 jam)

| # | Task | Output | Status |
|---|------|--------|--------|
| 1.1 | Clone repo InSet: `git clone https://github.com/Abaddon-Beza/InSet` | Folder InSet di lokal | Belum |
| 1.2 | Konversi file positif InSet ke `inset_lexicon_positive.csv` dengan kolom `word,weight` | `database/inset_lexicon_positive.csv` | Belum |
| 1.3 | Konversi file negatif InSet ke `inset_lexicon_negative.csv` dengan kolom `word,weight` | `database/inset_lexicon_negative.csv` | Belum |
| 1.4 | Clone repo kamus alay: `git clone https://github.com/nasalsabila/kamus-alay` | Folder kamus-alay di lokal | Belum |
| 1.5 | Konversi ke `kamus_alay.csv` dengan kolom `slang,baku` | `database/kamus_alay.csv` | Belum |
| 1.6 | Verifikasi ketiga file terbaca pandas tanpa error | Output `df.head()` di terminal | Belum |

### Sprint 2 — Implementasi Model dan Fungsi Core (Estimasi: 2–3 jam)

| # | Task | File | Status |
|---|------|------|--------|
| 2.1 | Tambah class `SentimentSettings` di `SentimenModel.py` | `SentimenModel.py` | Belum |
| 2.2 | Update import di `SentimenRoute.py` untuk include `SentimentSettings` | `SentimenRoute.py` | Belum |
| 2.3 | Tulis fungsi `load_inset_lexicon()` dengan caching | `SentimenRoute.py` | Belum |
| 2.4 | Tulis fungsi `inset_label_sentiment()` sesuai spesifikasi 6.1.5 | `SentimenRoute.py` | Belum |
| 2.5 | Ubah route `auto_label` untuk memanggil `inset_label_sentiment()` | `SentimenRoute.py` | Belum |
| 2.6 | Tambah route `save_settings` sesuai spesifikasi 6.3.4 | `SentimenRoute.py` | Belum |
| 2.7 | Tambah route `import_alay` sesuai spesifikasi 6.2.4 | `PreprocessingRoute.py` | Belum |
| 2.8 | Pastikan `db.create_all()` mengenali `SentimentSettings` | `app/__init__.py` | Belum |

### Sprint 3 — Update UI (Estimasi: 1 jam)

| # | Task | File | Status |
|---|------|------|--------|
| 3.1 | Ubah route `index` sentimen untuk mengirim `settings` dan `enriched_items` ke template | `SentimenRoute.py` | Belum |
| 3.2 | Tambah card konfigurasi labeling di halaman Sentimen | `sentimen/index.html` | Belum |
| 3.3 | Tambah kolom InSet Score dan Kata Cocok di tabel | `sentimen/index.html` | Belum |
| 3.4 | Tambah tombol Import Kamus Alay di halaman Preprocessing | `preprocessing/index.html` | Belum |

### Sprint 4 — Verifikasi End-to-End (Estimasi: 1–2 jam)

| # | Task | Verifikasi | Status |
|---|------|------------|--------|
| 4.1 | Import kamus alay via tombol baru | `NormalizationDict.query.count()` bertambah | Belum |
| 4.2 | Reset data sentimen lama | Tabel `sentiment_analysis` kosong | Belum |
| 4.3 | Jalankan ulang preprocessing dengan kamus alay aktif | Kata `gak` → `tidak` di teks hasil | Belum |
| 4.4 | Jalankan ulang auto-labeling dengan InSet | Distribusi label tidak lagi 77% netral | Belum |
| 4.5 | Reset TF-IDF dan NBC | Tabel `tfidf_conversion`, `nbc_*` kosong | Belum |
| 4.6 | Jalankan ulang TF-IDF → split → training → testing | Akurasi muncul di halaman NBC | Belum |
| 4.7 | Cek confusion matrix di `/nbc/results` | Tidak ada kolom all-zero | Belum |

---

## 13. Kriteria Penerimaan

### 13.1 Fungsional

- [ ] Fungsi `inset_label_sentiment()` ada di `SentimenRoute.py` dan menghasilkan label berdasarkan skor kata berbobot, bukan hitungan keyword
- [ ] Fungsi `auto_label_sentiment()` lama sudah dihapus atau tidak lagi dipanggil
- [ ] File `database/inset_lexicon_positive.csv` ada dengan kolom `word`, `weight`
- [ ] File `database/inset_lexicon_negative.csv` ada dengan kolom `word`, `weight`
- [ ] File `database/kamus_alay.csv` ada dengan kolom `slang`, `baku`
- [ ] Route `POST /sentimen/save_settings` berjalan dan menyimpan data ke tabel `sentiment_settings`
- [ ] Route `POST /preprocessing/import_alay` berjalan dan menambah entri ke `normalization_dict`
- [ ] Kata alay (`gak`, `bgt`, `yg`, dll) terpetakan dengan benar setelah import

### 13.2 Kualitas Data

- [ ] Distribusi label setelah relabeling dengan InSet: tidak ada kelas dengan proporsi < 5%
- [ ] Confusion matrix NBC setelah training ulang: tidak ada kolom atau baris yang semua nilainya 0
- [ ] Akurasi NBC masih di atas 70% setelah relabeling (jika data cukup seimbang)

### 13.3 Teknis

- [ ] Tabel `sentiment_settings` dibuat otomatis oleh SQLAlchemy tanpa error
- [ ] InSet Lexicon hanya di-load dari disk satu kali per session (tidak reload tiap request)
- [ ] Output dict `inset_label_sentiment()` berisi key: `label`, `confidence`, `positive_keywords`, `negative_keywords`, `neutral_keywords` — tidak lebih, tidak kurang
- [ ] Pipeline end-to-end tidak menghasilkan error: scraping → preprocessing → labeling → TF-IDF → NBC → evaluasi

### 13.4 Akademik

- [ ] Narasi BAB III skripsi menyebut "pelabelan berbasis InSet Lexicon" bukan "pelabelan berbasis keyword"
- [ ] Referensi Koto & Rahmaningtyas (2017) dicantumkan di DAFTAR PUSTAKA
- [ ] Referensi kamus alay (Nasalsabila, 2020) dicantumkan di DAFTAR PUSTAKA

---

## 14. Risiko dan Mitigasi

| Risiko | Kemungkinan | Dampak | Mitigasi |
|--------|-------------|--------|----------|
| InSet tidak mencakup kosakata domain politik/kabinet (misalnya: "makan siang gratis", "kabinet merah putih") | Tinggi | Tweet relevan mendapat skor 0 → tetap berlabel netral | Tambahkan kata kunci domain secara manual ke `inset_lexicon_positive.csv` dan `inset_lexicon_negative.csv` sebelum digunakan |
| Distribusi label masih tidak seimbang meski sudah pakai InSet | Sedang | Confusion matrix masih buruk karena data training tidak seimbang | Terapkan `class_weight='balanced'` di `MultinomialNB` atau gunakan SMOTE pada data training di `NBCRoute.py` (perubahan minor di luar scope utama) |
| Format file InSet dari GitHub berbeda dari yang diharapkan | Sedang | Perlu konversi manual tambahan | Verifikasi header kolom CSV setelah download, buat script konversi Python jika diperlukan |
| Route `import_alay` gagal karena `category` bukan kolom wajib di `NormalizationDict` | Rendah | Error saat insert | Kolom `category` ada di model dengan default `'general'`, aman diisi `'alay'` |
| `current_app.config['INSET_POS']` tidak persistent antar request di production server multi-worker | Rendah | Setiap worker reload file tersendiri, tidak efisien tapi tidak error | Untuk skripsi (single-worker dev server), ini tidak masalah. Jika masalah, gunakan global variable di modul sebagai alternatif cache |
| Data sentimen lama (dengan label keyword) sudah digunakan untuk TF-IDF dan NBC | Pasti | Hasil NBC lama tidak valid | Jalankan reset semua data di urutan: sentimen reset → konversi reset → NBC reset → labeling ulang → konversi ulang → training ulang |

---

## 15. Referensi dan Sitasi

### 15.1 Referensi Teknis

| # | Sumber | URL | Digunakan Untuk |
|---|--------|-----|-----------------|
| 1 | InSet Lexicon (file) | `github.com/Abaddon-Beza/InSet` | Download file leksikon positif/negatif |
| 2 | Kamus Alay | `github.com/nasalsabila/kamus-alay` | Download kamus slang Twitter |
| 3 | Scikit-learn MultinomialNB | `scikit-learn.org/stable/modules/naive_bayes.html` | Implementasi NBC (tidak berubah) |

### 15.2 Sitasi untuk DAFTAR PUSTAKA Skripsi

```
Koto, F., & Rahmaningtyas, G. H. (2017). InSet Lexicon: Evaluation of a Word List
  for Indonesian Sentiment Analysis in Microblogs. 2017 International Conference
  on Advanced Computer Science and Information Systems (ICAICTA), 391–396.
  https://doi.org/10.1109/ICAICTA.2017.8090993

Nasalsabila. (2020). Kamus Alay: Kamus Kata Tidak Baku Bahasa Indonesia.
  GitHub Repository. https://github.com/nasalsabila/kamus-alay
```

### 15.3 Dokumen Internal Terkait

| Dokumen | Keterangan |
|---------|------------|
| `design.md` | Design system frontend — wajib diikuti saat modifikasi template |
| `database/db_naive_bayes (1).sql` | Skema database awal sebagai referensi |
| `requirement.txt` | Dependency Python yang sudah terpasang |