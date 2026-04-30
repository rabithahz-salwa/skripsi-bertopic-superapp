# Temporal Topic Modeling of Indonesian Banking Superapp Reviews

Tugas Akhir S1 Statistika — Institut Teknologi Sepuluh Nopember (ITS) Surabaya.

Penelitian ini menganalisis keluhan pengguna superapp perbankan Indonesia (**wondr by BNI** dan **BYOND by BSI**) menggunakan temporal topic modeling dengan pendekatan **BERTopic + Dynamic Topic Modeling** dan **IndoBERT embeddings**, divalidasi statistik menggunakan **Mann-Kendall trend test** dan **Sen's slope**.

**Periode analisis:** 12 bulan post-launch per app (window terpisah).

---

## Methodology Overview

Pipeline penelitian terdiri dari 8 langkah:

1. **Data collection** — Scraping review Google Play Store (`google-play-scraper`)
2. **Preprocessing** — Cleaning, slang normalization, filtering (8 sub-tahap)
3. **Complaint filter** — Filter rating 1-2 sebagai proxy keluhan teknis (integrated di Tahap 1 preprocessing)
4. **Manual validation** — Validasi 300 sampel per app
5. **Topic modeling** — BERTopic + DTM dengan IndoBERT embeddings, fit terpisah per app
6. **Temporal analysis** — Modified Mann-Kendall (Hamed-Rao) + Sen's slope + cosine similarity c-TF-IDF
7. **Comparative analysis** — wondr vs BYOND
8. **Evaluation** — Coherence Cv, topic diversity, visualisasi

---

## Project Structure

```
TA/
├── data/
│   ├── raw/                          # Raw scraping data (gitignored)
│   │   ├── wondr_by_BNI_raw.csv      # 44,627 reviews
│   │   └── BYOND_by_BSI_raw.csv      # 48,757 reviews
│   └── processed/                    # Output preprocessing (gitignored)
│       ├── wondr_bertopic.csv        # Slim version untuk BERTopic
│       ├── wondr_full.csv            # Full audit version
│       ├── byond_bertopic.csv
│       └── byond_full.csv
├── dictionaries/
│   ├── banking_extension.csv         # Custom slang dict (64 entries, banking-specific)
│   ├── salsabila.csv                 # Salsabila lexicon (gitignored, download separately)
│   └── .gitkeep
├── utils/
│   ├── __init__.py
│   └── preprocessing.py              # Preprocessing functions module
├── 00_scraping.ipynb                 # Data collection notebook
├── 01_preprocessing_wondr.ipynb      # Pipeline preprocessing wondr (clean)
├── 02_preprocessing_byond.ipynb      # Pipeline preprocessing BYOND (clean)
├── 99_dev_wondr.ipynb                # Development & audit notebook (debug history)
├── scrape_superapp_reviews.py        # Scraping script
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Anaconda / Miniconda
- Python 3.11
- Git

### 1. Clone repository

```bash
git clone https://github.com/rabithahz-salwa/skripsi-bertopic-superapp.git
cd skripsi-bertopic-superapp
```

### 2. Create conda environment

```bash
conda create -n skripsi python=3.11
conda activate skripsi
pip install -r requirements.txt
```

### 3. Setup external resources

Beberapa file tidak di-commit ke repository karena lisensi atau ukuran. Download manual:

**a. Salsabila colloquial Indonesian lexicon**

Download `colloquial-indonesian-lexicon.csv` dari Salsabila et al. (2018):
https://github.com/nasalsabila/kamus-alay

Save sebagai `dictionaries/salsabila.csv`.

> **Citation:** Salsabila, N. A. F., et al. (2018). Colloquial Indonesian Lexicon. *2018 International Conference on Asian Language Processing (IALP)*. https://ieeexplore.ieee.org/document/8629151

**b. fasttext language detection model** (optional)

Download `lid.176.bin` (~126MB):
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Save di root folder project. Diperlukan hanya jika ingin reproduce audit yang menyebabkan deprecation language filter.

**c. Raw data**

Raw scraping data tidak di-commit. Untuk regenerate:

```bash
python scrape_superapp_reviews.py
```

Atau hubungi penulis untuk data backup.

### 4. Run notebooks

```bash
conda activate skripsi
jupyter notebook
```

Buka notebook sesuai tahap:
- **Preprocessing wondr:** `01_preprocessing_wondr.ipynb` (Restart Kernel → Run All)
- **Preprocessing BYOND:** `02_preprocessing_byond.ipynb` (Restart Kernel → Run All)

---

## Status

### ✅ Completed
- Data collection (44,627 + 48,757 raw reviews)
- Preprocessing pipeline 8 tahap (wondr + BYOND)
- Iterative refinement banking_extension dictionary

### 🔜 Next steps
- Manual validation 300 sampel per app
- BERTopic + DTM topic modeling (terpisah per app)
- Temporal trend analysis (Mann-Kendall + Sen's slope)
- Comparative analysis wondr vs BYOND

**Target sidang:** Juli 2026

---

## Methodology Notes

Beberapa keputusan metodologis penting:

- **Bucketing temporal:** Relative month (1-12 dari launch date), bukan calendar month — fair comparison antar app yang launch beda waktu
- **Stopword removal:** TIDAK di preprocessing — di-handle di tahap c-TF-IDF (BERTopic best practice)
- **Slang normalization:** Salsabila lexicon + 64-entry banking extension (compiled dari analisis vocabulary kedua app)
- **Language filter:** Deprecated berdasarkan data-driven audit (fasttext lid.176 punya systemic bias di domain banking Indonesia)

---

## Author

**Rabithah Zahiratus Salwa**
Mahasiswa S1 Statistika — ITS Surabaya
GitHub: [@rabithahz-salwa](https://github.com/rabithahz-salwa)

---

## License

Project ini bersifat akademik. Semua hak terkait penelitian merupakan bagian dari karya tugas akhir penulis.

External resources (Salsabila lexicon, fasttext lid.176, IndoBERT) tetap dimiliki oleh kreator masing-masing dan digunakan sesuai dengan lisensi yang berlaku.
