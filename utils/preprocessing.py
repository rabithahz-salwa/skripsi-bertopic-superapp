"""
Preprocessing utilities for thesis: Temporal Topic Modeling on Indonesian
banking superapp reviews (wondr by BNI vs BYOND by BSI).

Stage 1-2: Load, filter rating 1-2, extract relative_month & relative_week.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_raw_reviews(filepath: str, date_col: str = "date_wib") -> pd.DataFrame:
    """
    Load raw scraped reviews CSV and parse the date column to datetime.

    Parameters
    ----------
    filepath : str
        Path to raw CSV file (e.g., 'data/raw/wondr_by_BNI_raw.csv').
    date_col : str, default 'date_wib'
        Name of the date column to parse. We use 'date_wib' (UTC+7) by default
        because launch dates are conceptualized in Indonesian local time.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with `date_col` parsed as datetime.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Strip timezone info if present. The scraped data stores `date_wib`
    # as tz-aware (UTC+07:00), but downstream operations (e.g., comparing
    # with launch_date) expect tz-naive datetimes. We strip without
    # converting because the values are already in WIB.
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)

    n_unparsed = df[date_col].isna().sum()
    if n_unparsed > 0:
        print(f"⚠️  Warning: {n_unparsed} rows have unparseable {date_col}.")

    print(f"✅ Loaded {len(df):,} reviews from {filepath}")
    return df


def filter_negative_ratings(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """
    Keep only reviews with rating 1 or 2 (used as proxy for technical complaints).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a `rating` column.
    rating_col : str, default 'rating'
        Name of the rating column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only rating ∈ {1, 2}.
    """
    n_before = len(df)
    df_filtered = df[df[rating_col].isin([1, 2])].copy()
    n_after = len(df_filtered)
    n_dropped = n_before - n_after

    print(f"✅ Rating filter: kept {n_after:,} / {n_before:,} reviews "
          f"(dropped {n_dropped:,} non-negative reviews).")
    print(f"   Distribution: {df_filtered[rating_col].value_counts().to_dict()}")
    return df_filtered


def add_relative_time_columns(
    df: pd.DataFrame,
    launch_date: datetime,
    date_col: str = "date_wib",
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Add `relative_month` (1-12) and `relative_week` (1-52) columns based on
    the time difference from the app's launch date. Reviews outside the
    `window_months` window are dropped (with logging).

    Logic:
    - relative_month uses calendar-based difference via dateutil.relativedelta.
      Example for launch=5 July 2024:
        * 5 Jul – 4 Aug → month 1
        * 5 Aug – 4 Sep → month 2
        * 5 Jun – 4 Jul 2025 → month 12
        * 5 Jul 2025 onwards → dropped
    - relative_week uses simple day difference // 7.
      Example: day 0–6 → week 1, day 7–13 → week 2.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with parsed `date_col`.
    launch_date : datetime
        App launch date (Indonesian local time).
    date_col : str, default 'date_wib'
        Name of the datetime column.
    window_months : int, default 12
        Window length in months. Reviews with relative_month > window_months
        or < 1 are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with new `relative_month` and `relative_week` columns,
        filtered to within the window.
    """
    df = df.copy()

    # Drop rows with missing date (can't compute relative time)
    n_missing = df[date_col].isna().sum()
    if n_missing > 0:
        print(f"⚠️  Dropping {n_missing} rows with missing {date_col}.")
        df = df.dropna(subset=[date_col])

    # Compute relative_month using calendar-based difference
    def compute_relative_month(review_date):
        diff = relativedelta(review_date, launch_date)
        return diff.years * 12 + diff.months + 1

    df["relative_month"] = df[date_col].apply(compute_relative_month)

    # Compute relative_week using simple day difference, capped at 52.
    # Day 364 (last day of 12-month window) would naturally fall into
    # "week 53" with only 1-2 days of data — we collapse it into week 52
    # to keep weekly bins consistent (52 weeks = 1 year mental model).
    raw_week = ((df[date_col] - launch_date).dt.days // 7) + 1
    df["relative_week"] = raw_week.clip(upper=52)

    # Log out-of-window reviews before dropping
    n_before = len(df)
    before_launch = (df["relative_month"] < 1).sum()
    after_window = (df["relative_month"] > window_months).sum()

    df_in_window = df[
        (df["relative_month"] >= 1) & (df["relative_month"] <= window_months)
    ].copy()
    n_after = len(df_in_window)

    print(f"✅ Relative time extraction (launch={launch_date.date()}, "
          f"window={window_months} months):")
    print(f"   Kept {n_after:,} / {n_before:,} reviews in window.")
    if before_launch > 0:
        print(f"   Dropped {before_launch:,} reviews dated BEFORE launch.")
    if after_window > 0:
        print(f"   Dropped {after_window:,} reviews dated AFTER {window_months}-month window.")
    print(f"   Distribution per relative_month: "
          f"{df_in_window['relative_month'].value_counts().sort_index().to_dict()}")

    return df_in_window


def drop_exact_duplicates(
    df: pd.DataFrame,
    text_col: str = "review_text",
    show_top_n: int = 5,
) -> pd.DataFrame:
    """
    Drop exact duplicate reviews based on `text_col`, keeping the first
    (earliest) occurrence. Should be applied BEFORE text normalization
    so that case/punctuation differences are still distinguishable.

    Logs total duplicates removed and the top N most frequently
    duplicated review texts as a sanity check (these are usually
    boilerplate complaints, but unexpected anomalies should be inspected).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Should be sorted chronologically (or by relative
        time) so that `keep='first'` retains the earliest occurrence.
    text_col : str, default 'review_text'
        Name of the text column to dedupe by.
    show_top_n : int, default 5
        Number of most-frequent duplicate texts to display in the log.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    df = df.copy()

    # Sort by date_wib if present, so keep='first' gives earliest occurrence.
    if "date_wib" in df.columns:
        df = df.sort_values("date_wib").reset_index(drop=True)

    n_before = len(df)

    # Identify duplicates BEFORE dropping (for logging top-N)
    dup_mask = df.duplicated(subset=[text_col], keep="first")
    n_dropped = dup_mask.sum()

    # Top-N most frequently duplicated texts
    if n_dropped > 0:
        top_dupes = (
            df[df.duplicated(subset=[text_col], keep=False)]
            .groupby(text_col)
            .size()
            .sort_values(ascending=False)
            .head(show_top_n)
        )

    df_dedup = df[~dup_mask].reset_index(drop=True)
    n_after = len(df_dedup)

    print(f"✅ Exact duplicate removal (by '{text_col}', keep first):")
    print(f"   Kept {n_after:,} / {n_before:,} reviews "
          f"(dropped {n_dropped:,} duplicates).")

    if n_dropped > 0:
        print(f"\n   Top {show_top_n} most-duplicated review texts:")
        for i, (text, count) in enumerate(top_dupes.items(), 1):
            # Truncate long texts for readable log
            preview = text if len(str(text)) <= 80 else str(text)[:77] + "..."
            print(f"     {i}. [{count}x] {preview!r}")

    return df_dedup
