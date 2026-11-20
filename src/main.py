"""
Master Pipeline Orchestrator: Text Mining & Word Cloud
Project: Steel Ball Run Trailer (Netflix)
Target Data: Minimal 2,500 - 3,000 komentar publik
"""

import os
import sys
import time
import argparse

# Tambahkan src ke sys.path jika diperlukan
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module

def run_pipeline(target_count: int = 3000, skip_scrape: bool = False):
    start_total = time.time()
    print("=" * 70)
    print("   PIPELINE END-TO-END TEXT MINING & WORD CLOUD (STEEL BALL RUN)   ")
    print("=" * 70)

    raw_path = "data/raw/sbr_comments_raw.csv"
    
    # 1. Tahap Scraping
    if skip_scrape and os.path.exists(raw_path):
        import pandas as pd
        df_raw = pd.read_csv(raw_path)
        raw_count = len(df_raw)
        print(f"\n>>> [LANGKAH 1/3] Melewati scraping (--skip-scrape). Menggunakan data yang ada ({raw_count:,} baris).")
    else:
        print("\n>>> [LANGKAH 1/3] Menjalankan Scraping Komentar YouTube...")
        scrape_mod = import_module("01_scrape")
        raw_path, raw_count = scrape_mod.scrape_youtube_comments(
            url="https://youtu.be/k4mcxk8IZ2Y",
            limit=target_count,
            output_path=raw_path
        )

        if raw_count < 2500:
            print(f"[PERINGATAN] Jumlah komentar awal ({raw_count}) < 2.500. Mencoba pengunduhan tambahan...")
            raw_path, raw_count = scrape_mod.scrape_youtube_comments(
                url="https://youtu.be/k4mcxk8IZ2Y",
                limit=target_count,
                output_path=raw_path,
                sort_mode=0
            )

    # 2. Tahap Preprocessing
    print("\n>>> [LANGKAH 2/3] Menjalankan NLP Preprocessing & Lemmatization...")
    preprocess_mod = import_module("02_preprocess")
    clean_path, clean_count = preprocess_mod.run_preprocessing(
        input_path=raw_path,
        output_path="data/processed/sbr_comments_clean.csv"
    )

    # 3. Tahap Visualisasi
    print("\n>>> [LANGKAH 3/3] Menghasilkan Word Cloud & Analisis Frekuensi...")
    visualize_mod = import_module("03_visualize")
    wc_path, top_words_path = visualize_mod.generate_visualizations(
        input_path=clean_path,
        output_dir="output"
    )

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"   SELURUH PIPELINE SELESAI DENGAN SUKSES DENGAN WAKTU {total_time:.2f} DETIK!   ")
    print("=" * 70)
    print(f"Data Mentah   : {raw_path} ({raw_count:,} baris)")
    print(f"Data Bersih   : {clean_path} ({clean_count:,} baris)")
    print(f"Word Cloud    : {wc_path}")
    print(f"Top 20 Kata   : {top_words_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Text Mining Pipeline for SBR Trailer")
    parser.add_argument("--target", type=int, default=3000, help="Target comment limit (default: 3000)")
    parser.add_argument("--skip-scrape", action="store_true", help="Gunakan data raw yang sudah ada tanpa scraping ulang")
    args = parser.parse_args()

    run_pipeline(target_count=args.target, skip_scrape=args.skip_scrape)

