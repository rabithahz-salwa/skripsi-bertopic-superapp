"""
Scrape Google Play Store reviews for wondr by BNI and BYOND by BSI
untuk penelitian temporal topic modeling.

Strategi: Ambil SEMUA review dalam window koleksi (tanpa filter rating/bahasa).
Filtering akan dilakukan di tahap preprocessing terpisah.

Usage:
    pip install google-play-scraper pandas tqdm pytz
    python scrape_superapp_reviews.py

Author: Salwa (thesis, ITS Surabaya)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from google_play_scraper import Sort, reviews
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

WIB = pytz.timezone("Asia/Jakarta")


@dataclass(frozen=True)
class AppTarget:
    app_id: str
    app_name: str
    launch_date: datetime   # WIB-localized, inclusive (awal window)
    window_end: datetime    # WIB-localized, inclusive (akhir window)


TARGETS = [
    AppTarget(
        app_id="id.bni.wondr",
        app_name="wondr by BNI",
        launch_date=WIB.localize(datetime(2024, 7, 5, 0, 0, 0)),
        window_end=WIB.localize(datetime(2025, 7, 4, 23, 59, 59)),
    ),
    AppTarget(
        app_id="co.id.bankbsi.superapp",
        app_name="BYOND by BSI",
        launch_date=WIB.localize(datetime(2024, 11, 9, 0, 0, 0)),
        window_end=WIB.localize(datetime(2025, 11, 8, 23, 59, 59)),
    ),
]

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 200               # Maksimum per panggilan google-play-scraper
MAX_BATCHES_PER_APP = 10     # Safety ceiling: 1000 * 200 = 200k review
SLEEP_BETWEEN_BATCHES = 1.5    # Jeda antar batch (detik) agar tidak di-rate-limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def hash_user(name: str) -> str:
    """Hash nama user untuk privacy — tetap bisa melacak user yang sama
    tanpa menyimpan PII mentah."""
    if not name:
        return ""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def count_tokens(text: str) -> int:
    """Hitung jumlah kata secara kasar (whitespace-split setelah strip tanda baca)."""
    if not text:
        return 0
    stripped = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return len(stripped.split())


def to_wib(dt: datetime) -> datetime:
    """Konversi datetime (naive UTC dari API atau yang sudah aware) ke WIB."""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(WIB)


# ---------------------------------------------------------------------------
# Logika scraping
# ---------------------------------------------------------------------------

def scrape_app(target: AppTarget) -> tuple[pd.DataFrame, dict]:
    """
    Paginasi review (sort by NEWEST) sampai melewati tanggal launch.
    Return: (dataframe review dalam window, metadata scraping).
    """
    log.info("=" * 70)
    log.info("Scraping %s (%s)", target.app_name, target.app_id)
    log.info("Window: %s s/d %s (WIB)",
             target.launch_date.date(), target.window_end.date())
    log.info("=" * 70)

    all_rows = []
    continuation_token = None
    batches_fetched = 0
    total_raw_fetched = 0
    oldest_seen_wib = None
    newest_seen_wib = None
    stop_reason = "unknown"

    pbar = tqdm(total=MAX_BATCHES_PER_APP, desc=target.app_name, unit="batch")

    while batches_fetched < MAX_BATCHES_PER_APP:
        try:
            result, continuation_token = reviews(
                target.app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=BATCH_SIZE,
                continuation_token=continuation_token,
            )
        except Exception as e:
            log.error("Batch %d gagal: %s. Sleep 10 detik lalu retry.",
                      batches_fetched, e)
            time.sleep(10)
            try:
                result, continuation_token = reviews(
                    target.app_id,
                    lang="id",
                    country="id",
                    sort=Sort.NEWEST,
                    count=BATCH_SIZE,
                    continuation_token=continuation_token,
                )
            except Exception as e2:
                log.error("Retry juga gagal: %s. Berhenti untuk app ini.", e2)
                stop_reason = f"fetch_error: {e2}"
                break

        batches_fetched += 1
        pbar.update(1)

        if not result:
            stop_reason = "empty_batch"
            break

        total_raw_fetched += len(result)

        batch_times_wib = [to_wib(r["at"]) for r in result]
        batch_oldest = min(batch_times_wib)
        batch_newest = max(batch_times_wib)

        if oldest_seen_wib is None or batch_oldest < oldest_seen_wib:
            oldest_seen_wib = batch_oldest
        if newest_seen_wib is None or batch_newest > newest_seen_wib:
            newest_seen_wib = batch_newest

        for r in result:
            review_dt_wib = to_wib(r["at"])

            # Hanya simpan review dalam window koleksi
            if not (target.launch_date <= review_dt_wib <= target.window_end):
                continue

            text = r.get("content") or ""

            # Handle replyContent dan repliedAt yang bisa None
            reply_date = r.get("repliedAt")
            if reply_date is not None:
                if reply_date.tzinfo is None:
                    reply_date = pytz.utc.localize(reply_date)
                reply_date_iso = reply_date.isoformat()
            else:
                reply_date_iso = None

            # Normalisasi date_utc (API biasanya return naive UTC)
            raw_dt = r["at"]
            if raw_dt.tzinfo is None:
                raw_dt = pytz.utc.localize(raw_dt)

            all_rows.append({
                "review_id": r.get("reviewId", ""),
                "app_id": target.app_id,
                "app_name": target.app_name,
                "user_name_hash": hash_user(r.get("userName", "")),
                "rating": r["score"],
                "review_text": text,
                "thumbs_up_count": r.get("thumbsUpCount", 0),
                "review_created_version": r.get("reviewCreatedVersion"),
                "reply_content": r.get("replyContent"),
                "reply_date": reply_date_iso,
                "date_utc": raw_dt.isoformat(),
                "date_wib": review_dt_wib.isoformat(),
                "date_local": review_dt_wib.date().isoformat(),
                "word_count": count_tokens(text),
                "char_count": len(text),
                "scrape_timestamp": datetime.now(WIB).isoformat(),
            })

        # Berhenti kalau sudah melewati tanggal launch
        if batch_oldest < target.launch_date:
            stop_reason = "passed_launch_date"
            break

        if continuation_token is None:
            stop_reason = "no_continuation_token"
            break

        time.sleep(SLEEP_BETWEEN_BATCHES)

    pbar.close()

    metadata = {
        "app_name": target.app_name,
        "app_id": target.app_id,
        "stop_reason": stop_reason,
        "batches_fetched": batches_fetched,
        "total_raw_fetched": total_raw_fetched,
        "total_in_window": len(all_rows),
        "oldest_review_seen_wib": oldest_seen_wib.isoformat() if oldest_seen_wib else None,
        "newest_review_seen_wib": newest_seen_wib.isoformat() if newest_seen_wib else None,
        "launch_date_wib": target.launch_date.isoformat(),
        "window_end_wib": target.window_end.isoformat(),
        "reached_launch_date": (
            oldest_seen_wib is not None and oldest_seen_wib <= target.launch_date
        ),
    }

    log.info("Alasan berhenti: %s", stop_reason)
    log.info("Total batch diambil: %d", batches_fetched)
    log.info("Total review mentah (semua tanggal): %d", total_raw_fetched)
    log.info("Total review dalam window: %d", len(all_rows))
    if oldest_seen_wib:
        log.info("Review terlama yang terlihat: %s", oldest_seen_wib.date())
    if not metadata["reached_launch_date"]:
        log.warning(
            "API BERHENTI SEBELUM MENCAPAI TANGGAL LAUNCH. "
            "Review terlama %s, target launch %s. "
            "Data mungkin tidak lengkap untuk periode awal.",
            oldest_seen_wib.date() if oldest_seen_wib else "N/A",
            target.launch_date.date()
        )

    return pd.DataFrame(all_rows), metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_metadata = []

    for target in TARGETS:
        df, metadata = scrape_app(target)
        all_metadata.append(metadata)

        if df.empty:
            log.warning("Tidak ada review terkumpul untuk %s — cek app_id.",
                        target.app_name)
            continue

        safe_name = target.app_name.replace(" ", "_").replace("/", "-")
        output_path = OUTPUT_DIR / f"{safe_name}_raw.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        log.info("Tersimpan: %s (%d baris)", output_path, len(df))

        # Ringkasan cepat untuk verifikasi
        log.info("  Distribusi rating: %s",
                 df["rating"].value_counts().sort_index().to_dict())
        log.info("  Rentang tanggal review: %s s/d %s",
                 df["date_local"].min(), df["date_local"].max())
        log.info("  Review dengan <5 kata: %d",
                 int((df["word_count"] < 5).sum()))
        log.info("  Review dengan <10 kata: %d",
                 int((df["word_count"] < 10).sum()))

    # Simpan metadata scraping — berguna untuk dokumentasi metodologi
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = OUTPUT_DIR / "scrape_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8")
    log.info("Metadata scraping tersimpan: %s", metadata_path)

    log.info("Selesai.")


if __name__ == "__main__":
    main()
