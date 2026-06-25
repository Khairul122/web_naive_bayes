# Bahan Studi Skripsi
## Analisis Sentimen Kabinet Prabowo Subianto menggunakan Naïve Bayes Classifier

| | |
|---|---|
| **Judul Skripsi** | Analisis Sentimen Masyarakat Terhadap Pemerintah di Era Kabinet Prabowo Subianto berdasarkan Sosial Media X menggunakan Naïve Bayes Classifier |
| **Penulis** | Adetia Irvanda |
| **Institusi** | Teknik Informatika, Universitas Malikussaleh |
| **Tanggal Dokumen** | 26 Juni 2026 |

---

## BAGIAN 1 — LATAR BELAKANG

### 1.1 Konteks Penelitian

Kabinet Prabowo Subianto yang dilantik pada Oktober 2024 menjadi topik pembicaraan luas di media sosial, khususnya Twitter/X. Masyarakat mengekspresikan berbagai sentimen — dukungan, penolakan, maupun netralitas — melalui tweet dalam bahasa Indonesia informal.

Analisis manual terhadap ribuan tweet tidak praktis. Dibutuhkan sistem otomatis yang dapat:
1. Mengumpulkan data tweet (scraping)
2. Membersihkan teks bahasa Indonesia tidak baku
3. Menentukan sentimen tiap tweet secara akurat
4. Mengklasifikasi sentimen menggunakan machine learning
5. Mengevaluasi performa model secara ilmiah

### 1.2 Rumusan Masalah

1. Bagaimana cara menentukan sentimen tweet bahasa Indonesia yang mengandung bahasa alay dan slang secara akurat dan dapat disitasi secara ilmiah?
2. Bagaimana performa algoritma Multinomial Naïve Bayes dalam mengklasifikasi sentimen masyarakat terhadap Kabinet Prabowo Subianto?
3. Seberapa besar akurasi, precision, recall, dan F1-Score yang dihasilkan sistem?

### 1.3 Tujuan Penelitian

| Kode | Tujuan |
|------|--------|
| T-01 | Mengimplementasikan InSet Lexicon sebagai metode pelabelan sentimen yang dapat disitasi secara akademik |
| T-02 | Mengimplementasikan normalisasi kata alay Twitter untuk meningkatkan kualitas preprocessing |
| T-03 | Melatih model Multinomial Naïve Bayes menggunakan representasi TF-IDF |
| T-04 | Mengevaluasi model dengan confusion matrix, akurasi, precision, recall, dan F1-Score |

---

## BAGIAN 2 — LANDASAN TEORI

### 2.1 Twitter/X sebagai Sumber Data Opini Publik

Twitter (sekarang X) adalah platform microblogging yang membatasi tweet hingga 280 karakter. Karakteristik data Twitter yang relevan untuk penelitian ini:

- **Bahasa informal:** Banyak singkatan, kata alay, emotikon
- **Temporal:** Tweet bertanggal, bisa dianalisis tren waktu
- **Engagement metrics:** like (favorite), retweet, reply, quote menunjukkan dampak tweet
- **Multi-tipe konten:** tweet biasa, retweet, reply, quote tweet

Sistem ini menyimpan 12 atribut per tweet: `tweet_id_str`, `username`, `full_text`, `favorite_count`, `retweet_count`, `reply_count`, `quote_count`, `created_at`, `is_retweet`, `is_reply`, `is_quote`, `lang`.

---

### 2.2 Text Preprocessing — Tahapan Detail

Preprocessing mengubah teks mentah Twitter menjadi representasi bersih yang siap diolah mesin. Sistem mengimplementasikan **7 tahap berurutan**.

#### Tahap 1: Cleansing (Pembersihan)

Menghapus elemen non-linguistik dari tweet:

| Yang Dihapus | Pola Regex | Contoh |
|--------------|-----------|--------|
| URL | `http[s]?://...` | `https://t.co/xyz` → `` |
| Mention | `@\w+` | `@prabowo` → `` |
| Hashtag | `#\w+` | `#KabinetMerahPutih` → `` (opsional) |
| Angka | `\d+` | `2024` → `` |
| Tanda baca | `string.punctuation` | `!.,?` → `` |

**Contoh:**
```
Input : "@prabowo Kabinet baru 2024! Semoga berhasil https://t.co/abc"
Output: "Kabinet baru   Semoga berhasil"
```

#### Tahap 2: Case Folding (Penyeragaman Huruf)

Mengubah semua huruf menjadi huruf kecil (lowercase):

```
Input : "Kabinet Baru Semoga Berhasil"
Output: "kabinet baru semoga berhasil"
```

**Alasan:** Algoritma harus memperlakukan "Bagus" dan "bagus" sebagai token yang sama.

#### Tahap 3: Tokenizing (Pemecahan Token)

Memecah teks menjadi daftar kata individual (split berdasarkan spasi):

```
Input : "kabinet baru semoga berhasil"
Output: ['kabinet', 'baru', 'semoga', 'berhasil']
```

#### Tahap 4: Stopword Removal (Penghapusan Kata Henti)

Menghapus kata-kata yang sering muncul namun tidak membawa makna sentimen.

**Sumber stopword:** Tabel `stopword_list` di database. Jika kosong, digunakan default 36 kata termasuk:

```
dan, atau, yang, di, ke, dari, untuk, dengan, pada, dalam,
adalah, akan, telah, sudah, juga, tetapi, namun,
karena, oleh, agar, jika, kalau, bila, ketika, ini, itu, maka
```

> **Catatan penting:** Kata `"tidak"` dan `"bukan"` **tidak dimasukkan** ke default stopword karena keduanya adalah kata negasi yang memiliki bobot semantik. Memasukkannya akan menghapus informasi penting sebelum labeling sentimen.

**Contoh:**
```
Input : ['kabinet', 'baru', 'yang', 'akan', 'berhasil', 'untuk', 'rakyat']
Output: ['kabinet', 'baru', 'berhasil', 'rakyat']
Dihapus: ['yang', 'akan', 'untuk']
```

#### Tahap 5: Normalisasi (Slang → Baku)

Mengganti kata-kata tidak baku Twitter dengan padanan bakunya.

**Dua sumber kamus normalisasi:**

| Sumber | Kolom CSV | Contoh | Kategori di DB |
|--------|-----------|--------|----------------|
| KBBI (dictionary_baku_nonbaku.csv) | word, wrong | `cuma → hanya` | `kbbi` |
| Kamus Alay (kamus_alay.csv) | slang, baku | `gak → tidak`, `bgt → banget` | `alay` |

**Mengapa normalisasi penting sebelum InSet Lexicon:**
InSet Lexicon memiliki entri "tidak" tetapi tidak memiliki "gak" atau "ngga". Tanpa normalisasi, tweet seperti "gak bagus sama sekali" tidak akan mendeteksi kata sentimen apapun.

**Contoh normalisasi:**
```
Input : ['gak', 'bgt', 'bagus', 'sih', 'yg', 'baru']
Output: ['tidak', 'banget', 'bagus', 'lah', 'yang', 'baru']
```

#### Tahap 6: Stemming — Enhanced Confix Stripping (ECS)

Mengubah kata berimbuhan menjadi kata dasar menggunakan algoritma ECS yang diimplementasikan dalam library **Sastrawi** (Python).

**ECS vs. Porter Stemmer:**
| Aspek | Porter Stemmer | ECS (Sastrawi) |
|-------|---------------|----------------|
| Bahasa target | Inggris | Indonesia |
| Metode | Suffix removal berurutan | Confix stripping (awalan + akhiran sekaligus) |
| Akurasi Bahasa Indonesia | Rendah | Tinggi |

**Contoh stemming ECS:**
```
"mempermasalahkan" → "masalah"
"ketidakberhasilan" → "hasil"
"pemerintahan"     → "perintah"
"kebijaksanaan"    → "bijaksana"
"peningkatan"      → "tingkat"
```

#### Tahap 7: Filter Panjang Kata

Menghapus token yang terlalu pendek atau terlalu panjang:
- Default: `2 ≤ len(token) ≤ 50`
- Kata < 2 karakter biasanya noise (sisa tanda baca, dll.)

**Hasil Akhir Preprocessing:**
```
Tweet asli  : "@prabowo Kabinet baru 2024! Emang bgt keren bgt, maju terus!"
Setelah semua tahap: "kabinet baru benar keren maju"
```

---

### 2.3 InSet Lexicon — Metode Pelabelan Ground Truth

**Referensi akademik:**
> Koto, F., & Rahmaningtyas, G. H. (2017). InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs. *2017 International Conference on Advanced Computer Science and Information Systems (ICAICTA)*, 391–396. https://doi.org/10.1109/ICAICTA.2017.8090993

#### 2.3.1 Apa itu InSet Lexicon?

InSet (Indonesian Sentiment Lexicon) adalah kamus kata berbobot sentimen untuk bahasa Indonesia yang dikembangkan dari data Twitter dan divalidasi secara ilmiah. Berbeda dengan keyword matching biasa, InSet memberikan **bobot numerik** pada setiap kata.

| Leksikon | Rentang Skor | Arti |
|----------|-------------|------|
| Positif | +1 hingga +5 | Makin besar = makin positif |
| Negatif | -5 hingga -1 | Makin kecil = makin negatif |
| Tidak ada di leksikon | 0 | Tidak berkontribusi |

**Contoh entri leksikon:**
```
# Positif
bagus     → +3
luar biasa → +5
keren     → +2
setuju    → +2
mendukung → +3

# Negatif
buruk     → -3
korupsi   → -5
gagal     → -3
kecewa    → -2
berbohong → -4
```

#### 2.3.2 Algoritma Scoring InSet

> **Input teks:** InSet Lexicon menganalisis `case_folded_text` — teks setelah cleansing dan lowercase, **sebelum** stopword removal. Ini penting agar kata negasi seperti `"tidak"` tetap hadir saat pencocokan leksikon (meski `"tidak"` sendiri tidak ada di entri InSet, kehadirannya menjaga konteks).

```
GIVEN: teks = case_folded_text (pre-stopword-removal)
       pos_dict = {kata: skor_positif}  (skor > 0)
       neg_dict = {kata: skor_negatif}  (skor < 0)

PROSES:
  skor_total = 0
  kata_positif = []
  kata_negatif = []

  FOR token IN teks.split():
    IF token IN pos_dict:
      skor_total += pos_dict[token]
      kata_positif.append(token)
    IF token IN neg_dict:
      skor_total += neg_dict[token]
      kata_negatif.append(token)

KLASIFIKASI:
  IF skor_total > threshold_pos (+0.5):
    label = 'positif'
  ELIF skor_total < threshold_neg (-0.5):
    label = 'negatif'
  ELSE:
    label = 'netral'

CONFIDENCE:
  confidence = min(|skor_total| / 5.0, 1.0)
```

#### 2.3.3 Contoh Perhitungan Manual InSet

**Tweet:** "prabowo sudah maju keren tapi korupsi masih ada"

| Token | Di Pos Dict | Di Neg Dict | Skor |
|-------|-------------|-------------|------|
| prabowo | ✗ | ✗ | 0 |
| sudah | ✗ | ✗ | 0 |
| maju | +1 | ✗ | +1 |
| keren | +2 | ✗ | +2 |
| tapi | ✗ | ✗ | 0 |
| korupsi | ✗ | -5 | -5 |
| masih | ✗ | ✗ | 0 |
| ada | ✗ | ✗ | 0 |

```
skor_total   = 1 + 2 + (-5) = -2
threshold    = -2 < -0.5 → label = 'negatif'
confidence   = min(|-2| / 5.0, 1.0) = min(0.4, 1.0) = 0.40
```

**Hasil:** label=`negatif`, confidence=`0.40`, kata_positif=`[maju, keren]`, kata_negatif=`[korupsi]`

#### 2.3.4 Keunggulan InSet vs Keyword Matching Sederhana

| Aspek | Keyword Matching | InSet Lexicon |
|-------|-----------------|---------------|
| Bobot kata | Semua kata dihitung sama | Kata berbeda punya bobot berbeda |
| Sitasi akademik | Tidak ada | Koto & Rahmaningtyas (2017) |
| Sensitivitas | Rendah (hanya hitungan frekuensi) | Tinggi (mempertimbangkan intensitas) |
| Contoh | "bagus" = "luar biasa" | "bagus" (+3) ≠ "luar biasa" (+5) |
| Distribusi label | Bias netral (77%+) | Lebih seimbang |
| Threshold | Tidak ada | Dapat dikonfigurasi per user |

---

### 2.4 Term Frequency — Inverse Document Frequency (TF-IDF)

TF-IDF adalah metode pembobotan kata yang mengukur seberapa penting sebuah kata dalam suatu dokumen relatif terhadap seluruh koleksi dokumen.

#### 2.4.1 Formula TF-IDF

**Term Frequency (TF):**
```
TF(t, d) = jumlah kemunculan term t dalam dokumen d
           ──────────────────────────────────────────
           total jumlah term dalam dokumen d
```

**Inverse Document Frequency (IDF):**
```
IDF(t, D) = log( N / df(t) ) + 1

Keterangan:
  N     = jumlah total dokumen
  df(t) = jumlah dokumen yang mengandung term t
```

*Catatan: scikit-learn menggunakan versi smooth IDF:*
```
IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1
```

**TF-IDF Score:**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

#### 2.4.2 Contoh Perhitungan TF-IDF

**Koleksi dokumen (3 tweet setelah preprocessing):**
```
D1: "kabinet bagus maju rakyat"
D2: "kabinet gagal kecewa rakyat"
D3: "kabinet netral tidak jelas"
```

**Hitung IDF:**
| Term | df(t) | N | IDF = log(N/df)+1 |
|------|-------|---|------------------|
| kabinet | 3 | 3 | log(3/3)+1 = 0+1 = 1.0 |
| bagus | 1 | 3 | log(3/1)+1 = 1.099+1 = 2.099 |
| maju | 1 | 3 | log(3/1)+1 = 2.099 |
| rakyat | 2 | 3 | log(3/2)+1 = 0.405+1 = 1.405 |
| gagal | 1 | 3 | 2.099 |
| kecewa | 1 | 3 | 2.099 |

**Hitung TF-IDF untuk D1 = "kabinet bagus maju rakyat" (total 4 kata):**
| Term | TF | IDF | TF-IDF |
|------|----|-----|--------|
| kabinet | 1/4 = 0.25 | 1.0 | 0.25 |
| bagus | 1/4 = 0.25 | 2.099 | 0.525 |
| maju | 1/4 = 0.25 | 2.099 | 0.525 |
| rakyat | 1/4 = 0.25 | 1.405 | 0.351 |

**Vektor fitur D1** = `[0.25, 0.525, 0.525, 0.351, 0.0, 0.0]` (diurutkan sesuai vocabulary)

#### 2.4.3 Parameter TF-IDF dalam Sistem

| Parameter | Nilai Default | Penjelasan |
|-----------|--------------|------------|
| `max_features` | 1000 | Maksimum kata dalam vocabulary (diambil yang paling informatif) |
| `min_df` | 0.01 (1%) | Kata harus muncul minimal di 1% dokumen |
| `max_df` | 0.95 (95%) | Kata yang terlalu umum (>95% dokumen) dihapus |
| `lowercase` | True | Semua kata diubah ke huruf kecil |

**Alasan `min_df=0.01`:** Kata yang hanya muncul di 1–2 tweet saja tidak reliabel sebagai fitur prediksi.

**Alasan `max_df=0.95`:** Kata yang ada di hampir semua tweet tidak membedakan antar kelas.

---

### 2.5 Naïve Bayes Classifier (NBC) — Multinomial

#### 2.5.1 Teorema Bayes

Teorema Bayes adalah fondasi dari algoritma NBC:

```
P(C | X) = P(X | C) × P(C)
            ──────────────
                P(X)

Keterangan:
  P(C | X)  = Probabilitas posterior — probabilitas kelas C diberikan fitur X
  P(X | C)  = Likelihood — probabilitas fitur X muncul di kelas C
  P(C)      = Prior probability — probabilitas kelas C secara umum
  P(X)      = Evidence — probabilitas fitur X (konstan, sering diabaikan)
```

#### 2.5.2 Multinomial Naïve Bayes

Multinomial NB digunakan untuk data dengan fitur berupa **frekuensi/hitungan** (atau TF-IDF). Cocok untuk text classification.

**Asumsi "Naïve" (polos):** Setiap fitur (kata) independen satu sama lain given kelasnya. Meski tidak realistis, asumsi ini bekerja dengan baik dalam praktik.

**Formula prediksi:**
```
Ĉ = argmax_c [ log P(C=c) + Σᵢ xᵢ × log P(tᵢ | C=c) ]
                              i=1

Keterangan:
  Ĉ           = kelas yang diprediksi
  c           = setiap kelas yang mungkin (positif, negatif, netral)
  log P(C=c)  = log prior probability kelas c
  xᵢ          = nilai fitur ke-i (TF-IDF score)
  P(tᵢ | C=c) = probabilitas term tᵢ di kelas c
```

**Mengapa log?** Menghindari underflow (nilai probabilitas sangat kecil yang mendekati 0 ketika dikalikan).

#### 2.5.3 Laplace Smoothing (Alpha)

Masalah: jika sebuah kata tidak pernah muncul di kelas tertentu dalam training, `P(t | C) = 0` → seluruh produk menjadi 0.

**Solusi — Laplace Smoothing:**
```
P(tᵢ | C=c) = count(tᵢ, c) + α
               ──────────────────
               count(seluruh kata, c) + α × V

Keterangan:
  count(tᵢ, c) = jumlah kemunculan term tᵢ di semua dokumen kelas c
  α            = parameter smoothing (default = 1.0 dalam sistem ini)
  V            = ukuran vocabulary (total fitur unik)
```

Dengan `α = 1.0` (Laplace smoothing), setiap kata dianggap muncul minimal 1 kali di setiap kelas.

#### 2.5.4 Training NBC — Step by Step

**Input:** `X_train` (matrix TF-IDF, shape: n_sampel × n_fitur), `y_train` (array label)

**Step 1: Hitung Prior Probability per Kelas**
```
P(C=positif) = jumlah_dokumen_positif / total_dokumen
P(C=negatif) = jumlah_dokumen_negatif / total_dokumen
P(C=netral)  = jumlah_dokumen_netral  / total_dokumen
```

**Step 2: Hitung Likelihood P(tᵢ | C=c) dengan Laplace Smoothing**
```
UNTUK SETIAP KELAS c:
  - Kumpulkan semua dokumen kelas c
  - Jumlahkan TF-IDF tiap fitur di kelas c → count(tᵢ, c)
  - Hitung P(tᵢ | c) = (count(tᵢ,c) + 1) / (Σcount(t,c) + V)
```

**Parameter yang disimpan ke database:**
- `feature_log_prob` — array 2D shape (n_kelas × n_fitur): `log P(tᵢ | C=c)`
- `class_log_prior` — array 1D shape (n_kelas): `log P(C=c)`
- `classes` — daftar nama kelas: `['negatif', 'netral', 'positif']`

#### 2.5.5 Testing NBC — Step by Step

**Input:** `X_test` (matrix TF-IDF dari data testing), model dari database

**Rekonstruksi model:**
```python
model = MultinomialNB(alpha=alpha)
model.feature_log_prob_ = array([[...]])  # dibaca dari DB
model.class_log_prior_  = array([...])    # dibaca dari DB
model.classes_          = array([...])    # dibaca dari DB
```

**Prediksi untuk satu dokumen X:**
```
score_negatif = log P(negatif) + Σ xᵢ × log P(tᵢ | negatif)
score_netral  = log P(netral)  + Σ xᵢ × log P(tᵢ | netral)
score_positif = log P(positif) + Σ xᵢ × log P(tᵢ | positif)

Prediksi = kelas dengan score tertinggi
```

#### 2.5.6 Contoh Perhitungan Manual NBC

**Data Training:**
```
Positif (2 dokumen): "bagus keren maju" | "luar biasa bagus"
Negatif (2 dokumen): "gagal korupsi"    | "kecewa gagal buruk"
Netral  (2 dokumen): "biasa saja"       | "standar netral"
```

**Prior Probability:**
```
P(positif) = 2/6 = 0.333
P(negatif) = 2/6 = 0.333
P(netral)  = 2/6 = 0.333
```

**Vocabulary:** {bagus, keren, maju, luar, biasa, gagal, korupsi, kecewa, buruk, saja, standar, netral}
V = 12 kata

**Likelihood P("bagus" | positif) dengan α=1:**
```
count("bagus", positif) = 2 (muncul di kedua dok positif)
Σcount(seluruh kata, positif) = 3 + 3 = 6  (kata total di semua dok positif)

P("bagus" | positif) = (2 + 1) / (6 + 1×12) = 3/18 = 0.167
```

**Prediksi untuk tweet baru: "bagus maju"**
```
log_score_positif = log(0.333) + log(P("bagus"|pos)) + log(P("maju"|pos))
                  = -1.099 + log(3/18) + log(2/18)
                  = -1.099 + (-1.792) + (-2.197)
                  = -5.088

log_score_negatif = log(0.333) + log(P("bagus"|neg)) + log(P("maju"|neg))
                  = -1.099 + log(1/17) + log(1/17)   # kata tidak muncul, pakai smoothing
                  = -1.099 + (-2.833) + (-2.833)
                  = -6.765

log_score_netral  = log(0.333) + log(P("bagus"|neu)) + log(P("maju"|neu))
                  = -1.099 + log(1/14) + log(1/14)
                  = -1.099 + (-2.639) + (-2.639)
                  = -6.377

Prediksi: positif (score tertinggi = -5.088)
```

---

### 2.6 Pembagian Data: Training dan Testing (Train-Test Split)

Sistem menggunakan **stratified train-test split** dari scikit-learn:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `test_size` | 0.3 | 30% data untuk testing, 70% untuk training |
| `random_state` | 42 | Angka seed untuk reproducibility — hasil split selalu sama |
| `stratify=y` | ✓ | Proporsi kelas dijaga sama antara train dan test |

**Mengapa stratify?**
Tanpa stratify, jika data tidak seimbang, bisa terjadi training set dengan kelas minoritas sangat sedikit. Stratify memastikan rasio positif:negatif:netral sama di train dan test.

---

### 2.7 Evaluasi Model

#### 2.7.1 Confusion Matrix

Confusion matrix menunjukkan perbandingan antara label aktual dan label prediksi:

```
                  PREDIKSI
                Pos   Neg   Net
AKTUAL  Pos  [  TP   FPN   FNe  ]
        Neg  [ FPP   TN   FNt   ]   (untuk multiclass, setiap kelas
        Net  [ FPP  FPN    TN   ]    dilihat sebagai one-vs-rest)
```

Untuk klasifikasi 3 kelas (positif, negatif, netral), confusion matrix berbentuk 3×3:

```
              Predicted
              Pos  Neg  Net
Actual  Pos [ a    b    c  ]
        Neg [ d    e    f  ]
        Net [ g    h    i  ]

a = TP_positif (positif diprediksi positif) — BENAR
e = TP_negatif (negatif diprediksi negatif) — BENAR
i = TP_netral  (netral diprediksi netral)   — BENAR
```

**Confusion matrix yang baik:** Diagonal (a, e, i) bernilai besar, off-diagonal kecil.

**Masalah sebelum perbaikan InSet:** Kolom `negatif` dan `netral` hampir semua bernilai 0 karena model selalu memprediksi `positif` — akibat distribusi label yang sangat tidak seimbang.

#### 2.7.2 Metrik Evaluasi

**Accuracy (Akurasi):**
```
Accuracy = jumlah prediksi benar / total prediksi

         = TP_pos + TP_neg + TP_net
           ──────────────────────────
                  total data
```

**Precision (Presisi) per kelas:**
```
Precision_c = TP_c / (TP_c + FP_c)

Arti: Dari semua yang diprediksi kelas c, berapa yang benar-benar kelas c?
```

**Recall (Sensitivitas) per kelas:**
```
Recall_c = TP_c / (TP_c + FN_c)

Arti: Dari semua data yang benar-benar kelas c, berapa yang berhasil diprediksi?
```

**F1-Score per kelas:**
```
F1_c = 2 × Precision_c × Recall_c
           ─────────────────────────
           Precision_c + Recall_c

Arti: Rata-rata harmonik precision dan recall. Berguna ketika data tidak seimbang.
```

**Weighted Average (yang dipakai sistem):**
```
Precision_weighted = Σ (support_c × Precision_c) / total_data
Recall_weighted    = Σ (support_c × Recall_c)    / total_data
F1_weighted        = Σ (support_c × F1_c)        / total_data

Keterangan: support_c = jumlah sampel kelas c di testing
```

#### 2.7.3 Contoh Perhitungan Metrik

**Misalkan confusion matrix:**
```
              Pred_Pos  Pred_Neg  Pred_Net
Actual_Pos  [   20        2         3   ]  → support = 25
Actual_Neg  [    3       18         4   ]  → support = 25
Actual_Net  [    2        2        21   ]  → support = 25
                                             total = 75
```

**Accuracy:**
```
Accuracy = (20 + 18 + 21) / 75 = 59/75 = 0.787 (78.7%)
```

**Precision untuk kelas Positif:**
```
TP_pos = 20
FP_pos = 3 + 2 = 5  (Neg dan Net yang salah diprediksi sebagai Pos)
Precision_pos = 20 / (20 + 5) = 20/25 = 0.80
```

**Recall untuk kelas Positif:**
```
TP_pos = 20
FN_pos = 2 + 3 = 5  (Pos yang salah diprediksi sebagai Neg atau Net)
Recall_pos = 20 / (20 + 5) = 20/25 = 0.80
```

**F1-Score untuk kelas Positif:**
```
F1_pos = 2 × 0.80 × 0.80 / (0.80 + 0.80) = 0.80
```

---

## BAGIAN 3 — IMPLEMENTASI SISTEM

### 3.1 Arsitektur Aplikasi

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAPISAN PRESENTASI                          │
│  Bootstrap 5 + Star Admin2 Template + Font Awesome 6            │
│  HTML/CSS/JavaScript + AppDialog (dialog custom global)         │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP Request/Response
┌─────────────────────────────────────────────────────────────────┐
│                     LAPISAN APLIKASI                            │
│  Flask 3.1 + Blueprint Pattern + Session Authentication         │
│  7 Blueprint: auth, dashboard, scrapping, preprocessing,        │
│               sentimen, konversi, nbc                           │
└─────────────────────────────────────────────────────────────────┘
                              │ SQLAlchemy ORM
┌─────────────────────────────────────────────────────────────────┐
│                     LAPISAN DATA                                │
│  MySQL 8.0 + SQLAlchemy 2.0                                     │
│  12 Tabel: users, twitter_scraping, text_preprocessing,         │
│  preprocessing_settings, stopword_list, normalization_dict,     │
│  sentiment_analysis, sentiment_settings, tfidf_conversion,      │
│  tfidf_vocabulary, nbc_training, nbc_testing, nbc_model         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Modul-Modul Sistem

#### Modul 1: Scrapping
- **Fungsi:** Menyimpan data tweet dari Twitter/X ke database
- **Tabel:** `twitter_scraping`
- **Atribut kunci:** `full_text`, `username`, `created_at`, `favorite_count`, `retweet_count`
- **Langkah selanjutnya:** Data mentah dari modul ini menjadi input modul Preprocessing

#### Modul 2: Preprocessing
- **Fungsi:** Membersihkan dan menormalisasi teks tweet
- **Tabel:** `text_preprocessing` (menyimpan hasil setiap tahap)
- **7 Tahap:** Cleansing → Case Folding → Tokenizing → Stopword Removal → Normalisasi → Stemming → Filter
- **Konfigurasi:** Via `preprocessing_settings` dan `normalization_dict` (database-driven)
- **Output:** `final_text` — teks siap masuk InSet Lexicon dan TF-IDF

#### Modul 3: Sentimen
- **Fungsi:** Memberi label sentimen pada setiap tweet secara otomatis
- **Tabel:** `sentiment_analysis`, `sentiment_settings`
- **Engine:** InSet Lexicon (default) atau Keyword Matching (fallback)
- **Input teks:** `case_folded_text` untuk InSet Lexicon (pre-stopword), `final_text` untuk keyword fallback
- **Output:** `sentiment_label` ∈ {positif, negatif, netral}, `confidence_score` ∈ [0.0, 1.0]
- **Threshold:** Dapat dikonfigurasi per user (`threshold_pos`, `threshold_neg`)

#### Modul 4: Konversi TF-IDF
- **Fungsi:** Mengubah teks menjadi representasi numerik (vektor fitur)
- **Tabel:** `tfidf_conversion`, `tfidf_vocabulary`
- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Output:** Setiap tweet direpresentasikan sebagai vektor float berukuran `max_features`
- **Penyimpanan:** Vektor disimpan sebagai JSON string di kolom `feature_vector`

#### Modul 5: NBC (Naive Bayes Classifier)
- **Fungsi:** Melatih, menguji, dan mengevaluasi model klasifikasi sentimen
- **Tabel:** `nbc_training`, `nbc_testing`, `nbc_model`
- **Sub-proses:**
  1. **Split:** Bagi data TF-IDF ke training (70%) dan testing (30%)
  2. **Training:** Latih MultinomialNB, simpan parameter ke DB
  3. **Testing:** Rekonstruksi model dari DB, prediksi data testing
  4. **Evaluasi:** Hitung accuracy, precision, recall, F1, confusion matrix
  5. **Kalkulasi Manual:** Hitung NBC step-by-step untuk 5 sampel (BAB III)

---

### 3.3 Alur Data Lengkap (Dengan Nama Kolom)

```
twitter_scraping.full_text
        │
        ▼ [PreprocessingRoute.preprocess_text()]
text_preprocessing.case_folded_text  ←── setelah cleansing + lowercase (pre-stopword)
text_preprocessing.final_text        ←── teks bersih setelah 7 tahap lengkap
        │
        ├─→ [SentimenRoute.inset_label_sentiment()]  ← pakai case_folded_text
sentiment_analysis.sentiment_label     = 'positif' | 'negatif' | 'netral'
sentiment_analysis.confidence_score    = 0.0 – 1.0
sentiment_analysis.positive_keywords  = kata positif yang cocok (comma-separated)
sentiment_analysis.negative_keywords  = kata negatif yang cocok (comma-separated)
        │
        ▼ [KonversiRoute.convert_tfidf()]
tfidf_conversion.feature_vector  = "[0.0, 0.125, 0.0, 0.342, ...]" (JSON, panjang=max_features)
tfidf_vocabulary.term            = setiap kata unik dalam vocabulary
tfidf_vocabulary.idf_score       = nilai IDF per kata
        │
        ▼ [NBCRoute.split_data()]
nbc_training.feature_vector + nbc_training.label   ← 70% data
nbc_testing.feature_vector + nbc_testing.true_label ← 30% data
        │
        ▼ [NBCRoute.train_model()]
nbc_model.feature_log_prob    = log P(tᵢ|c) untuk semua fitur × kelas
nbc_model.class_log_prior     = log P(c) untuk semua kelas
nbc_model.classes             = ['negatif', 'netral', 'positif']
        │
        ▼ [NBCRoute.test_model()]
nbc_testing.predicted_label      = hasil prediksi
nbc_testing.prediction_probability = probabilitas per kelas (JSON)
nbc_testing.is_correct           = True/False
nbc_model.accuracy               = nilai akurasi final
nbc_model.precision_score        = nilai precision weighted
nbc_model.recall_score           = nilai recall weighted
nbc_model.f1_score               = nilai F1 weighted
nbc_model.classification_report  = laporan lengkap per kelas (JSON)
```

---

## BAGIAN 4 — KRITERIA KEBERHASILAN PENELITIAN

### 4.1 Target Distribusi Label (Setelah InSet Lexicon)

| Kondisi | Sebelum InSet | Target InSet |
|---------|--------------|-------------|
| Positif | ~9.2% | ≥ 10% |
| Negatif | ~13.1% | ≥ 15% |
| Netral | ~77.7% | ≤ 75% |
| Kelas < 5% | Ada | Tidak ada |

### 4.2 Target Kualitas Model

| Metrik | Target Minimum |
|--------|---------------|
| Akurasi | ≥ 70% |
| Precision (weighted) | ≥ 65% |
| Recall (weighted) | ≥ 65% |
| F1-Score (weighted) | ≥ 65% |
| Confusion matrix | Tidak ada baris/kolom yang semua nilainya = 0 |

### 4.3 Cara Membaca Hasil di Sistem

Di halaman `/nbc/results`:
1. **Metrik utama** ditampilkan sebagai card (Akurasi, Precision, Recall, F1)
2. **Confusion matrix** divisualisasikan sebagai heatmap
3. **Performance chart** menampilkan precision/recall/F1 per kelas secara bar chart
4. **Word cloud** menampilkan kata paling sering per kelas sentimen
5. **Manual calculation** menampilkan perhitungan NBC step-by-step untuk 5 dokumen pertama

---

## BAGIAN 5 — DAFTAR PUSTAKA & SITASI

### Referensi Wajib Dicantumkan di Skripsi

```
[1] Koto, F., & Rahmaningtyas, G. H. (2017). InSet Lexicon: Evaluation of a Word List
    for Indonesian Sentiment Analysis in Microblogs. 2017 International Conference on
    Advanced Computer Science and Information Systems (ICAICTA), 391–396.
    https://doi.org/10.1109/ICAICTA.2017.8090993

[2] Nasalsabila. (2020). Kamus Alay: Kamus Kata Tidak Baku Bahasa Indonesia.
    GitHub Repository. https://github.com/nasalsabila/kamus-alay

[3] Devika, M. D., Sunitha, C., & Ganesh, A. (2016). Sentiment Analysis: A Comparative
    Study on Different Approaches. Procedia Computer Science, 87, 44–49.
    https://doi.org/10.1016/j.procs.2016.05.124

[4] Pak, A., & Paroubek, P. (2010). Twitter as a Corpus for Sentiment Analysis and
    Opinion Mining. Proceedings of LREC 2010.

[5] Zhang, L., Wang, S., & Liu, B. (2018). Deep Learning for Sentiment Analysis: A Survey.
    WIREs Data Mining and Knowledge Discovery.
    https://doi.org/10.1002/widm.1253

[6] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information
    Retrieval. Cambridge University Press.
    (Referensi untuk TF-IDF — Chapter 6)

[7] McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for Naive Bayes
    Text Classification. AAAI-98 Workshop on Learning for Text Categorization.
    (Referensi untuk Multinomial Naïve Bayes)

[8] Asian, J., Williams, H. E., & Tahaghoghi, S. M. M. (2007). Stemming Indonesian.
    Proceedings of the 28th Australasian Conference on Computer Science, 307–314.
    (Referensi untuk ECS stemming — dasar Sastrawi)
```

---

## BAGIAN 6 — PERTANYAAN YANG SERING DITANYAKAN PENGUJI

### Q1: Mengapa menggunakan Multinomial NB dan bukan Gaussian NB?

**Jawaban:** Multinomial NB dirancang untuk data dengan fitur berupa frekuensi/hitungan, yang persis sesuai dengan representasi TF-IDF. Gaussian NB mengasumsikan distribusi Gaussian pada fitur — tidak cocok untuk data teks yang sebagian besar bernilai 0 (sparse). Multinomial NB juga terbukti bekerja baik untuk text classification dalam banyak penelitian.

### Q2: Mengapa Laplace smoothing dengan alpha=1?

**Jawaban:** Alpha=1 adalah nilai default yang dikenal sebagai Laplace smoothing. Nilai ini memberikan "pseudocount" 1 untuk setiap kata di setiap kelas, menghindari probabilitas nol untuk kata yang tidak muncul di training set. Nilai ini merupakan pilihan standar yang sering menghasilkan performa optimal.

### Q3: Mengapa tidak menggunakan deep learning?

**Jawaban:** Penelitian ini memilih Naïve Bayes karena:
1. Sesuai dengan judul skripsi yang sudah ditetapkan
2. NBC efisien secara komputasi — tidak membutuhkan GPU
3. NBC interpretable — mudah dijelaskan secara matematis
4. Dataset relatif kecil (ribuan tweet) — deep learning cenderung overfit pada dataset kecil
5. NBC dengan preprocessing yang baik bisa mencapai akurasi kompetitif

### Q4: Apa kekurangan InSet Lexicon untuk topik ini?

**Jawaban:** InSet Lexicon adalah kamus umum yang tidak mencakup kosakata domain politik/kabinet seperti "makan siang gratis", "kabinet merah putih", atau nama-nama kebijakan spesifik. Tweet yang membahas topik ini tanpa kata sentimen umum akan berlabel netral meski sebenarnya mengandung opini. Ini merupakan limitasi penelitian yang perlu diakui.

### Q5: Mengapa split 70:30?

**Jawaban:** Pembagian 70% training dan 30% testing adalah konvensi umum dalam machine learning yang memberikan keseimbangan antara:
- Cukup data untuk training model
- Cukup data untuk evaluasi yang representatif

Dengan random_state=42, hasil split selalu reproducible — peneliti lain dapat mereplikasi eksperimen.

### Q6: Bagaimana menangani ketidakseimbangan kelas (class imbalance)?

**Jawaban:** Ketidakseimbangan kelas diatasi melalui:
1. **InSet Lexicon** — metode pelabelan yang lebih sensitif dan menghasilkan distribusi lebih seimbang dibanding keyword matching
2. **Stratified split** — memastikan proporsi kelas sama di train dan test
3. **Weighted metrics** — precision/recall/F1 dihitung dengan pembobotan (weighted average) untuk memperhitungkan imbalance
4. Jika masalah persists, bisa ditambahkan `class_weight='balanced'` di MultinomialNB (di luar scope penelitian saat ini)

### Q7: Apa yang dimaksud confidence score?

**Jawaban:** Confidence score dalam sistem ini adalah normalisasi skor InSet Lexicon ke rentang [0.0, 1.0]:
```
confidence = min(|skor_total| / 5.0, 1.0)
```
Nilai mendekati 1.0 berarti tweet memiliki kata-kata sentimen yang kuat. Nilai mendekati 0.0 berarti skor InSet dekat nol (label netral atau tidak ada kata sentimen yang cocok). Ini BUKAN probabilitas dalam arti statistis — hanya indikator kekuatan sentimen.

### Q8: Bagaimana sistem menangani kata negasi seperti "tidak bagus"?

**Jawaban:** Ini adalah salah satu keterbatasan InSet Lexicon yang perlu diakui secara jujur dalam penelitian. Sistem menggunakan `case_folded_text` (teks setelah cleansing dan lowercase, sebelum stopword removal) sebagai input InSet Lexicon — bukan `final_text` — sehingga kata `"tidak"` dan `"bukan"` tetap hadir dalam analisis. Namun, `"tidak"` bukan entri di InSet Lexicon (yang berisi kata sifat dan kata bermakna sentimen, bukan partikel negasi), sehingga frasa `"tidak bagus"` tetap menghasilkan skor `+3` dari kata `"bagus"` saja.

Penanganan negasi yang benar membutuhkan parser sintaksis (misalnya: jika token sebelum kata leksikon adalah kata negasi, balik tanda skornya). Ini di luar scope penelitian ini dan merupakan rekomendasi pengembangan lebih lanjut yang dapat disebutkan di BAB V (Saran).
