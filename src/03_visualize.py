"""
Module 03: High-Resolution Word Cloud & Frequency Analysis
Project: Steel Ball Run Trailer Text Mining & Word Cloud
Standar Teknis:
  - Resolusi Word Cloud minimal 1600x900 px, 300 DPI (sbr_wordcloud.png)
  - Colormap: viridis (sesuai standar NLP text mining EN)
  - Collocations: False (mencegah duplikasi frasa)
  - Output Top-20 kata: output/top_words.csv
  - Visualisasi tambahan: Bar Chart Top-20 kata beresolusi tinggi (output/top_20_words_chart.png)
"""

import os
import sys
import argparse
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_visualizations(
    input_path: str = "data/processed/sbr_comments_clean.csv",
    output_dir: str = "output"
):
    print("=" * 65)
    print(f"[03_VISUALIZE] Memulai Pembuatan Word Cloud & Analisis Frekuensi")
    print(f"Input Data : {input_path}")
    print(f"Output Dir : {output_dir}")
    print("=" * 65)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File '{input_path}' tidak ditemukan. Jalankan 02_preprocess.py terlebih dahulu.")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[INFO] Memuat {len(df):,} baris data bersih.")

    # Kumpulkan seluruh token kata
    all_words = []
    for text in df["clean_text"].dropna():
        words = str(text).split()
        all_words.extend(words)

    total_words = len(all_words)
    unique_words = len(set(all_words))
    print(f"[INFO] Total kata dalam korpus   : {total_words:,}")
    print(f"       Total kosakata unik (vocab): {unique_words:,}")

    # Hitung frekuensi kata
    word_counts = Counter(all_words)
    top_20 = word_counts.most_common(20)

    # 1. Simpan Top-20 kata ke CSV
    os.makedirs(output_dir, exist_ok=True)
    df_top20 = pd.DataFrame(top_20, columns=["word", "frequency"])
    top_words_path = os.path.join(output_dir, "top_words.csv")
    df_top20.to_csv(top_words_path, index=False, encoding="utf-8")
    
    print("-" * 65)
    print(f"[TABEL FREKUENSI] Top-20 Kata Terbanyak (Setelah Stopword Refinement):")
    print("-" * 65)
    for idx, (w, f) in enumerate(top_20, 1):
        print(f"{idx:2d}. {w:<15} : {f:>5,d} kali")
    print(f"\n[INFO] Tabel Top-20 kata tersimpan di: {top_words_path}")

    # 2. Render Word Cloud Resolusi Tinggi (1600x900 px, 300 DPI, Colormap viridis)
    print("[INFO] Merender Word Cloud resolusi tinggi (1600x900 px @ 300 DPI, colormap viridis)...")
    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=150,
        random_state=42,
        collocations=False,
        prefer_horizontal=0.85
    ).generate_from_frequencies(word_counts)

    wordcloud_path = os.path.join(output_dir, "sbr_wordcloud.png")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[SUKSES] Word Cloud tersimpan di: {wordcloud_path}")

    # 3. Render Grafik Bar Chart Top-20 Kata (300 DPI)
    chart_path = os.path.join(output_dir, "top_20_words_chart.png")
    plt.figure(figsize=(12, 7.5), dpi=300)
    words = [item[0] for item in reversed(top_20)]
    counts = [item[1] for item in reversed(top_20)]

    # Gunakan gradasi warna viridis untuk bar chart
    colors = plt.cm.viridis([i / len(words) for i in range(len(words))])
    bars = plt.barh(words, counts, color=colors, edgecolor="black", alpha=0.9, height=0.7)

    plt.xlabel("Frekuensi Kemunculan Kata", fontsize=12, fontweight="bold")
    plt.ylabel("Kata Kunci (Lemmatized)", fontsize=12, fontweight="bold")
    plt.title("Top-20 Kata Kunci Dominan Trailer JoJo: Steel Ball Run", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Tambahkan label angka presisi
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(counts) * 0.012, bar.get_y() + bar.get_height()/2, f"{int(width):,}",
                 va="center", ha="left", fontsize=9.5, fontweight="bold")

    plt.xlim(0, max(counts) * 1.12)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik Top-20 kata tersimpan di: {chart_path}")

    print("-" * 65)
    print(f"[VERIFIKASI TAHAP 3] Visualisasi Selesai!")
    print(f"Output files:")
    print(f"  1. {top_words_path}")
    print(f"  2. {wordcloud_path} (1600x900 px @ 300 DPI)")
    print(f"  3. {chart_path} (300 DPI)")
    print("-" * 65)

    return wordcloud_path, top_words_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Cloud & Frequency Visualization")
    parser.add_argument("--input", type=str, default="data/processed/sbr_comments_clean.csv", help="Input clean CSV path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")

    args = parser.parse_args()
    generate_visualizations(input_path=args.input, output_dir=args.output_dir)
