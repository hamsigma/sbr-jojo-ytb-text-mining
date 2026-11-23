"""
Script: Generate Publication-Grade Pipeline Flowchart Diagram
Output: output/pipeline_flowchart.png (High Resolution, 300 DPI)
Flow: START -> Scraping (with loop) -> Preprocessing -> Word Cloud Visualization -> FINISH
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart(output_path="output/pipeline_flowchart.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Kanvas proporsional (16:11)
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Judul Flowchart
    ax.text(50, 97, "DIAGRAM ALIR (FLOWCHART) PIPELINE TEXT MINING & WORD CLOUD", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1a237e")
    ax.text(50, 94, "Studi Kasus: Komentar Trailer Resmi JoJo's Bizarre Adventure - Steel Ball Run (Netflix)", 
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#37474f")

    # Helper function kotak proses
    def add_box(x, y, w, h, text, subtitle="", box_color="#e3f2fd", border_color="#1976d2", text_color="#0d47a1"):
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.2",
            facecolor=box_color, edgecolor=border_color, linewidth=2, zorder=2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x, y + h*0.18, text, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color, zorder=3)
            ax.text(x, y - h*0.18, subtitle, ha="center", va="center", fontsize=8.5, color="#263238", zorder=3)
        else:
            ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color, zorder=3)

    # Helper function diamond keputusan
    def add_decision(x, y, size, text, box_color="#fff9c4", border_color="#fbc02d", text_color="#f57f17"):
        diamond = patches.Polygon([
            [x, y + size],
            [x + size * 1.5, y],
            [x, y - size],
            [x - size * 1.5, y]
        ], facecolor=box_color, edgecolor=border_color, linewidth=2, zorder=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color, zorder=3)

    # Helper function panah
    def add_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#37474f", edgecolor="#37474f", width=1.5, headwidth=7, headlength=7, shrink=0.05),
            zorder=1
        )
        if label:
            ax.text((x1+x2)/2 + 2, (y1+y2)/2, label, fontsize=8.5, fontweight="bold", color="#d32f2f", zorder=4)

    # ==========================================
    # 1. FASE 1: PENGUMPULAN DATA (KOLOM KIRI)
    # ==========================================
    add_box(20, 88, 26, 5, "START", "Inisialisasi Sumber Data Video SBR", "#e8f5e9", "#43a047", "#1b5e20")
    
    add_arrow(20, 85.5, 20, 79.5)
    add_box(20, 76, 30, 6, "1. SCRAPING KOMENTAR", "Target: https://youtu.be/k4mcxk8IZ2Y\nModule: src/01_scrape.py", "#e1f5fe", "#0288d1", "#01579b")
    
    add_arrow(20, 73, 20, 66)
    add_decision(20, 62, 3.8, "Jumlah Komentar\n>= 2.500?", "#fff9c4", "#fbc02d", "#f57f17")
    
    # Cabang Decision (Tidak: Loop Kembali ke Scraping)
    ax.annotate("", xy=(30, 76), xytext=(26, 62),
                arrowprops=dict(facecolor="#d32f2f", edgecolor="#d32f2f", width=1.2, headwidth=6, headlength=6), zorder=1)
    ax.text(30.5, 69, "Tidak (< 2.500)\n(Sort Popular)", fontsize=8, fontweight="bold", color="#d32f2f")

    # Cabang Decision (Ya)
    add_arrow(20, 58.2, 20, 51.5, "Ya")
    add_box(20, 48, 30, 6, "DATA MENTAH TERSIMPAN", "data/raw/sbr_comments_raw.csv\n(2.538 Baris Mentah Terkumpul)", "#f3e5f5", "#8e24aa", "#4a148c")

    # Transisi ke Kolom Tengah (Preprocessing)
    ax.annotate("", xy=(50, 88), xytext=(35, 48),
                arrowprops=dict(facecolor="#37474f", edgecolor="#37474f", width=1.5, headwidth=7, headlength=7), zorder=1)

    # ==========================================
    # 2. FASE 2: PREPROCESSING (KOLOM TENGAH)
    # ==========================================
    add_box(50, 88, 30, 5.5, "2. TEXT PREPROCESSING", "Pembersihan Teks: Case Folding\nRegex: Hapus URL, @mention, Angka, Emoji", "#ede7f6", "#5e35b1", "#311b92")

    add_arrow(50, 85.25, 50, 78.75)
    add_box(50, 75.5, 30, 5.5, "TOKENISASI & STOPWORDS", "NLTK Tokenizer + English Stopwords\n+ Custom Domain (jojo, trailer, dll.)", "#ede7f6", "#5e35b1", "#311b92")

    add_arrow(50, 72.75, 50, 66.25)
    add_box(50, 63, 30, 5.5, "LEMMATISASI MORFOLOGIS", "NLTK WordNetLemmatizer + POS Tag\nFilter Panjang Kata (len >= 3)", "#ede7f6", "#5e35b1", "#311b92")

    add_arrow(50, 60.25, 50, 53.75)
    add_box(50, 50.5, 30, 5.5, "FILTER DERAU / BARIS KOSONG", "206 Baris Kosong Dieliminasi\nTingkat Retensi Data: 91,88%", "#ede7f6", "#5e35b1", "#311b92")

    add_arrow(50, 47.75, 50, 41.25)
    add_box(50, 38, 30, 5.5, "DATA BERSIH TERSIMPAN", "data/processed/sbr_comments_clean.csv\n(2.332 Baris Komentar Bersih)", "#e8f5e9", "#43a047", "#1b5e20")

    # Transisi ke Kolom Kanan (Visualisasi Word Cloud)
    ax.annotate("", xy=(80, 88), xytext=(65, 38),
                arrowprops=dict(facecolor="#37474f", edgecolor="#37474f", width=1.5, headwidth=7, headlength=7), zorder=1)

    # ==========================================
    # 3. FASE 3: ANALISIS & WORD CLOUD (KOLOM KANAN)
    # ==========================================
    add_box(80, 88, 28, 5.5, "3. EKSTRAKSI FREKUENSI", "Korpus: 9.996 Kata (2.954 Vocab)\nHitung Distribusi Kemunculan Kata", "#fff3e0", "#fb8c00", "#e65100")

    add_arrow(80, 85.25, 80, 76.75)
    add_box(80, 73.5, 28, 5.5, "OUTPUT TABEL FREKUENSI", "output/top_words.csv\nTop-20 Kata Kunci Dominan", "#fff3e0", "#fb8c00", "#e65100")

    add_arrow(80, 70.75, 80, 62.25)
    add_box(80, 59, 28, 5.5, "VISUALISASI WORD CLOUD", "1600x900 px @ 300 DPI (Viridis)\nsbr_wordcloud.png & chart.png", "#fff3e0", "#fb8c00", "#e65100")

    add_arrow(80, 56.25, 80, 47.75)
    add_box(80, 44.5, 24, 5, "FINISH", "Visualisasi Word Cloud Selesai", "#e8f5e9", "#43a047", "#1b5e20")

    # Container Box / Grouping Visual
    def add_group_box(x, y, w, h, title, bg_color="#fafafa", border_color="#cfd8dc"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.0,rounding_size=1.5",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5, linestyle="--", zorder=0
        )
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 2.5, title, fontsize=9.5, fontweight="bold", color="#455a64", zorder=1)

    add_group_box(4, 40, 32, 53, "FASE 1: PENGUMPULAN DATA")
    add_group_box(33.5, 30, 33, 63, "FASE 2: NLP PREPROCESSING")
    add_group_box(64.5, 36.5, 31, 56.5, "FASE 3: ANALISIS & VISUALISASI WORD CLOUD")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUKSES] Flowchart resolusi tinggi berhasil diperbarui di: {output_path}")

if __name__ == "__main__":
    draw_flowchart()
