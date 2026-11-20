r"""
Module 02: NLP Text Preprocessing & Cleaning
Project: Steel Ball Run Trailer Text Mining & Word Cloud
Standar Teknis:
  1. Regex Cleaning: Hapus URL (http\S+|www\S+), mention (@\w+), angka (\d+), emoji, karakter non-alfabet ([^a-zA-Z\s])
  2. NLTK Resources: Pengecekan otomatis punkt, punkt_tab, stopwords, wordnet, omw-1.4
  3. Domain Stopwords: jojo, bizarre, adventure, trailer, netflix, anime, video, watch, watching, part, dll.
  4. Custom Stopwords: Penghapusan kata generik/filler non-spesifik SBR (like, get, one, really, even, still, make, go, see, know, dll.)
  5. Filter Panjang Kata: Minimal 3 huruf (len(word) >= 3) setelah lemmatization
"""

import os
import sys
import re
import argparse
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Pastikan seluruh dependensi NLTK terunduh otomatis
def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/omw-1.4", "omw-1.4")
    ]
    for res_path, res_name in resources:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(res_name, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

# Daftar Custom & Domain Stopwords Terkurasi untuk Steel Ball Run
CUSTOM_STOPWORDS = {
    # 1. Domain trailer & platform kata umum
    "jojo", "bizarre", "adventure", "part", "video", "trailer", "watch", 
    "watching", "netflix", "anime", "steel", "ball", "run", "sbr", "david", 
    "production", "episode", "series", "youtube", "comment", "channel",
    
    # 2. Kata umum & filler bahasa Inggris (evaluasi stopwords iteratif)
    "like", "get", "one", "really", "even", "still", "make", "go", 
    "see", "know", "think", "look", "time", "people", "thing", "say", 
    "tell", "come", "take", "give", "back", "much", "way", "day", 
    "never", "ever", "well", "want", "need", "let", "call", "show", 
    "man", "bro", "guy", "yes", "please", "would", "could", "also", 
    "first", "year", "good", "bad", "new", "feel", "try", "use",
    
    # 3. Slang, kontraksi tanpa tanda baca, & kata kasar filler
    "gon", "gonna", "cant", "dont", "wont", "didnt", "aint", "weeky",
    "wan", "wanna", "gotta", "fuck", "fucking", "shit", "damn", "omg", "lol"
}

def get_wordnet_pos(treebank_tag: str):
    """Konversi tag POS Treebank ke format WordNet untuk lemmatization presisi."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class TextPreprocessor:
    def __init__(self, additional_stopwords=None):
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            english_stops = set(stopwords.words("english"))
        except Exception:
            english_stops = set()
            
        self.stop_words = english_stops.union(CUSTOM_STOPWORDS)
        if additional_stopwords:
            self.stop_words = self.stop_words.union(set(additional_stopwords))

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Case folding
        text = text.lower()

        # 2. Hapus URL (http://, https://, www.*)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 3. Hapus mention pengguna (@username)
        text = re.sub(r"@\w+", " ", text)

        # 4. Hapus timestamps video (misal 01:23, 2:45)
        text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)

        # 5. Hapus entitas HTML dan escape characters
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\\[nrt]", " ", text)

        # 6. Hapus emoji & karakter non-ASCII
        text = text.encode("ascii", "ignore").decode("ascii")

        # 7. Hapus angka dan digit
        text = re.sub(r"\d+", " ", text)

        # 8. Hapus tanda baca dan karakter non-alfabet (hanya sisakan a-z dan spasi)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # 9. Tokenization
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        if not tokens:
            return ""

        # 10. POS Tagging & Lemmatization
        try:
            pos_tags = pos_tag(tokens)
        except Exception:
            pos_tags = [(t, 'N') for t in tokens]

        cleaned_tokens = []
        for word, tag in pos_tags:
            word = word.strip()
            # Filter awal kata pendek (< 3 karakter) dan stopwords
            if len(word) >= 3 and word not in self.stop_words:
                wn_pos = get_wordnet_pos(tag)
                try:
                    lemma = self.lemmatizer.lemmatize(word, pos=wn_pos)
                except Exception:
                    lemma = word
                
                # Verifikasi kembali panjang kata & status stopword setelah lemmatization
                if len(lemma) >= 3 and lemma not in self.stop_words:
                    cleaned_tokens.append(lemma)

        return " ".join(cleaned_tokens)

def run_preprocessing(
    input_path: str = "data/raw/sbr_comments_raw.csv",
    output_path: str = "data/processed/sbr_comments_clean.csv"
):
    print("=" * 65)
    print(f"[02_PREPROCESS] Memulai Preprocessing Teks NLP (Terkurasi & Tajam)")
    print(f"Input Data  : {input_path}")
    print(f"Output Data : {output_path}")
    print("=" * 65)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File '{input_path}' tidak ditemukan. Jalankan 01_scrape.py terlebih dahulu.")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    initial_count = len(df)
    print(f"[INFO] Total data mentah yang dimuat: {initial_count:,} baris")

    # Drop duplikasi
    if "cid" in df.columns:
        df = df.drop_duplicates(subset=["cid"])
    else:
        df = df.drop_duplicates(subset=["text"])
    
    unique_count = len(df)
    print(f"[INFO] Data setelah hapus duplikasi: {unique_count:,} baris")

    if "text" not in df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan dalam dataset raw!")

    # Inisialisasi Preprocessor
    preprocessor = TextPreprocessor()
    print("[INFO] Menjalankan pembersihan, tokenizing, stopword removal, dan lemmatization...")
    
    df["clean_text"] = df["text"].astype(str).apply(preprocessor.clean_text)

    # Filter data kosong setelah pembersihan
    df_clean = df[df["clean_text"].str.strip().str.len() > 0].copy()
    final_count = len(df_clean)
    dropped_empty = unique_count - final_count

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("-" * 65)
    print(f"[VERIFIKASI TAHAP 2] Preprocessing Selesai!")
    print(f"Jumlah baris awal            : {initial_count:,}")
    print(f"Komentar non-huruf/kosong    : {dropped_empty:,} baris dibuang")
    print(f"Jumlah baris bersih akhir    : {final_count:,} baris tersimpan")
    print(f"Persentase data valid        : {(final_count / initial_count) * 100:.2f}%")
    print(f"Lokasi file                  : {output_path}")
    print("-" * 65)

    return output_path, final_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Preprocessing for SBR Comments")
    parser.add_argument("--input", type=str, default="data/raw/sbr_comments_raw.csv", help="Input raw CSV path")
    parser.add_argument("--output", type=str, default="data/processed/sbr_comments_clean.csv", help="Output clean CSV path")

    args = parser.parse_args()
    run_preprocessing(input_path=args.input, output_path=args.output)
