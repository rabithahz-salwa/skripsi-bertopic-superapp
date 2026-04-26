\# Temporal Topic Modeling of Indonesian Banking Superapp Reviews



Tugas Akhir S1 Statistika — Institut Teknologi Sepuluh Nopember (ITS) Surabaya.



Penelitian ini menganalisis keluhan pengguna superapp perbankan Indonesia (wondr by BNI dan BYOND by BSI) menggunakan \*\*temporal topic modeling\*\* dengan pendekatan BERTopic dan IndoBERT embeddings, divalidasi statistik menggunakan Mann-Kendall trend test.



\## Methodology



1\. \*\*Data collection\*\* — Scraping review Google Play Store menggunakan `google-play-scraper`

2\. \*\*Complaint filter\*\* — Filter berbasis rating untuk mengisolasi review keluhan

3\. \*\*Preprocessing\*\* — Cleaning teks, language detection, normalisasi slang

4\. \*\*Topic modeling\*\* — BERTopic dengan IndoBERT sebagai embedding model

5\. \*\*Trend analysis\*\* — Mann-Kendall test + Sen's slope untuk validasi tren temporal



\## Project Structure



TA/

├── data/

│   ├── raw/              # Raw scraping data (gitignored, regenerable)

│   └── processed/        # Cleaned data (gitignored)

├── dictionaries/         # Slang dictionaries

├── utils/                # Reusable Python modules

├── 00\_scraping.ipynb     # Data collection notebook

└── scrape\_superapp\_reviews.py



\## Setup



\### Prerequisites

\- Anaconda / Miniconda

\- Python 3.11

\- Git



\### Installation

```bash

\# Clone repository

git clone https://github.com/rabithahz-salwa/skripsi-bertopic-superapp.git

cd skripsi-bertopic-superapp



\# Create conda environment

conda create -n skripsi python=3.11

conda activate skripsi



\# Install dependencies

pip install -r requirements.txt

```



\### Running notebooks

```bash

conda activate skripsi

jupyter notebook

```



\## Data



Raw data tidak di-commit ke repository karena ukurannya besar dan dapat di-regenerate. Untuk reproduksi:

```bash

python scrape\_superapp\_reviews.py

```



\## Author



\*\*Rabithah Zahiratus Salwa\*\* — Mahasiswa S1 Statistika ITS Surabaya  



\## License



Project ini bersifat akademik. Semua hak terkait penelitian merupakan bagian dari karya tugas akhir penulis.

