"""
Module 01: YouTube Comment Scraper
Project: Steel Ball Run Trailer Text Mining & Word Cloud
Target Video: https://youtu.be/k4mcxk8IZ2Y
Target Rows: Minimal 2,500 - 3,000 comments
"""

import os
import sys
import argparse
import time
import pandas as pd
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

def scrape_youtube_comments(
    url: str = "https://youtu.be/k4mcxk8IZ2Y",
    limit: int = 3000,
    output_path: str = "data/raw/sbr_comments_raw.csv",
    sort_mode: int = SORT_BY_RECENT
):
    print("=" * 60)
    print(f"[01_SCRAPE] Memulai proses scraping komentar YouTube")
    print(f"Target Video : {url}")
    print(f"Target Limit : {limit} komentar")
    print(f"Output File  : {output_path}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    downloader = YoutubeCommentDownloader()
    
    comments = []
    start_time = time.time()
    
    try:
        # Fetch comments
        comment_generator = downloader.get_comments_from_url(url, sort_by=sort_mode)
        
        count = 0
        for item in comment_generator:
            comments.append({
                "cid": item.get("cid"),
                "text": item.get("text"),
                "time": item.get("time"),
                "author": item.get("author"),
                "votes": item.get("votes", 0),
                "photo": item.get("photo", ""),
                "heart": item.get("heart", False),
                "reply": item.get("reply", False)
            })
            count += 1
            if count % 250 == 0:
                print(f"[SCRAPER] Terunduh: {count:,} komentar ({time.time() - start_time:.1f} detik)...")
                
            if count >= limit:
                print(f"[SCRAPER] Target limit {limit:,} komentar tercapai.")
                break
                
    except Exception as e:
        print(f"[ERROR] Terjadi kendala saat scraping: {e}", file=sys.stderr)

    # Fallback to popular sorting if recent comments are insufficient
    if len(comments) < 2500 and sort_mode == SORT_BY_RECENT:
        print(f"[INFO] Komentar terbaru kurang dari 2.500 ({len(comments)}). Menggabungkan dengan komentar populer...")
        try:
            seen_cids = {c["cid"] for c in comments}
            pop_generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
            for item in pop_generator:
                cid = item.get("cid")
                if cid and cid not in seen_cids:
                    seen_cids.add(cid)
                    comments.append({
                        "cid": cid,
                        "text": item.get("text"),
                        "time": item.get("time"),
                        "author": item.get("author"),
                        "votes": item.get("votes", 0),
                        "photo": item.get("photo", ""),
                        "heart": item.get("heart", False),
                        "reply": item.get("reply", False)
                    })
                    if len(comments) >= limit:
                        break
        except Exception as e:
            print(f"[WARNING] Gagal mengambil komentar populer tambahan: {e}")

    df = pd.DataFrame(comments)
    total_scraped = len(df)
    elapsed = time.time() - start_time

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("-" * 60)
    print(f"[VERIFIKASI TAHAP 1] Scraping Selesai dalam {elapsed:.2f} detik")
    print(f"Total baris mentah tersimpan : {total_scraped:,} baris")
    print(f"Lokasi file                  : {output_path}")

    if total_scraped < 2500:
        print(f"[WARNING] Peringatan: Total baris ({total_scraped}) kurang dari target minimal 2.500!")
    else:
        print(f"[SUKSES] Target terpenuhi (> 2.500 komentar).")
    print("-" * 60)
    
    return output_path, total_scraped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Comments Scraper for Steel Ball Run Trailer")
    parser.add_argument("--url", type=str, default="https://youtu.be/k4mcxk8IZ2Y", help="YouTube video URL")
    parser.add_argument("--limit", type=int, default=3000, help="Target minimum comment count")
    parser.add_argument("--output", type=str, default="data/raw/sbr_comments_raw.csv", help="Output CSV path")
    parser.add_argument("--sort", type=int, default=1, choices=[0, 1], help="0: Popular, 1: Recent (Default)")

    args = parser.parse_args()
    scrape_youtube_comments(url=args.url, limit=args.limit, output_path=args.output, sort_mode=args.sort)

