# Steel Ball Run Trailer - Text Mining & Word Cloud Pipeline

Proyek ini merupakan pipeline end-to-end Text Mining dan Word Cloud untuk analisis sentimen dan persepsi publik terhadap trailer **STEEL BALL RUN JoJo’s Bizarre Adventure | Official Trailer | Netflix** (`https://youtu.be/k4mcxk8IZ2Y`).

## Struktur Direktori
```text
.
├── .agent/
│   ├── rules.md
│   └── workflows.md
├── data/
│   ├── raw/                # Data mentah hasil scraping (sbr_comments_raw.csv)
│   └── processed/          # Data bersih setelah NLP preprocessing (sbr_comments_clean.csv)
├── output/
│   ├── sbr_wordcloud.png   # Visualisasi Word Cloud resolusi tinggi (1600x900 @ 300 DPI)
│   └── top_words.csv       # Tabel frekuensi Top-20 kata terbanyak
├── src/
│   ├── 01_scrape.py        # Modul scraper komentar YouTube
│   ├── 02_preprocess.py    # Modul pembersihan & preprocessing teks NLP
│   ├── 03_visualize.py     # Modul pembuatan Word Cloud & analisis frekuensi
│   └── main.py             # Pipeline orchestrator
├── AGENTS.md               # Pedoman Agent & Instruksi Proyek
├── requirements.txt        # Daftar dependensi Python
└── README.md
```

## Setup & Instalasi
1. Pastikan Python 3.10+ terinstal.
2. Buat virtual environment dan pasang dependensi:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Menjalankan Pipeline
Jalankan seluruh pipeline dari awal hingga akhir:
```bash
python src/main.py --target 3000
```
Atau jalankan per tahap secara modular:
```bash
# 1. Scraping komentar mentah
python src/01_scrape.py --limit 3000

# 2. Preprocessing & Lemmatization
python src/02_preprocess.py

# 3. Visualisasi Word Cloud & Top-20 Words
python src/03_visualize.py
```

