"""
Preprocessing utilities for thesis: Temporal Topic Modeling on Indonesian
banking superapp reviews (wondr by BNI vs BYOND by BSI).

Stage 1-2: Load, filter rating 1-2, extract relative_month & relative_week.
"""

from __future__ import annotations

import re
import pandas as pd
import emoji
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


# ==============================================================================
# Stage 4: Text Normalization
# ==============================================================================
# Pre-compiled regex patterns (compiled once at import for efficiency).
# Order of operations matters — see normalize_text() docstring.

_URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|\b\w+\.(?:com|co\.id|id|net|org|io)\S*',
    flags=re.IGNORECASE,
)

# ASCII emoticons: covers :) :( :D :P :v xD <3 ^_^ etc.
# Designed to match common Indonesian-typed emoticons without being greedy.
_EMOTICON_PATTERN = re.compile(
    r'(?:'
    r'[:;=8][\-o\*\']?[\)\]\(\[dDpPvV/\\:\}\{@\|]'  # :) :( :D :P :v :/ etc.
    r'|<3'                                            # heart
    r'|\^[_\-\.]?\^'                                  # ^_^ ^^ ^.^
    r'|xD|XD'                                         # laughing
    r')',
)

_NUMBER_PATTERN = re.compile(r'\d+')

# Repeated chars: 3+ same chars in a row → collapse to 2.
# Example: "errorrrr" → "errorr", "halooo" → "haloo"
_REPEAT_PATTERN = re.compile(r'(.)\1{2,}')

# Punctuation/symbols: keep only letters and whitespace.
# This runs AFTER URL/emoji/number removal, so we don't lose info.
_PUNCT_PATTERN = re.compile(r'[^a-z\s]')

_WHITESPACE_PATTERN = re.compile(r'\s+')

# Single-letter tokens left over from removing numbers/punctuation
# (e.g., 'v1.3.1' → 'v', ':v' → 'v'). These are noise for topic modeling.
# Matches a single letter surrounded by word boundaries.
_SINGLE_CHAR_PATTERN = re.compile(r'\b[a-z]\b')


def lowercase_text(text: str) -> str:
    """Convert text to lowercase. Returns empty string for non-string input."""
    if not isinstance(text, str):
        return ""
    return text.lower()


def remove_urls(text: str) -> str:
    """
    Remove URLs and domain-like patterns from text.
    Handles: http://..., https://..., www...., bri.co.id, example.com, etc.
    """
    return _URL_PATTERN.sub(' ', text)


def remove_emojis(text: str) -> str:
    """
    Remove both Unicode emojis (via `emoji` library) and ASCII emoticons
    (via regex). Replaces with single space to avoid word concatenation.
    """
    text = emoji.replace_emoji(text, replace=' ')
    text = _EMOTICON_PATTERN.sub(' ', text)
    return text


def remove_numbers(text: str) -> str:
    """Remove all digits. Standalone numbers and digits inside words both go."""
    return _NUMBER_PATTERN.sub(' ', text)


def collapse_repeated_chars(text: str) -> str:
    """
    Collapse 3+ consecutive repeated characters down to 2.
    Conservative approach: preserves natural Indonesian double letters
    (e.g., 'saat', 'lebih', 'maaf') while normalizing exaggerated repetition.

    Examples
    --------
    'errorrrr' → 'errorr'
    'halooo'   → 'haloo'
    'saat'     → 'saat'   (unchanged)
    """
    return _REPEAT_PATTERN.sub(r'\1\1', text)


def remove_punctuation(text: str) -> str:
    """
    Remove all characters except lowercase letters and whitespace.
    Should be called AFTER lowercase_text(), remove_urls(), remove_emojis(),
    and remove_numbers() — those steps need punctuation/digits to work.
    """
    return _PUNCT_PATTERN.sub(' ', text)


def remove_single_chars(text: str) -> str:
    """
    Remove single-letter tokens left as orphans by previous cleaning steps.

    Examples
    --------
    'v1.3.1'    after remove_numbers + remove_punctuation → 'v'    → ''
    ':v'        after remove_punctuation                  → 'v'    → ''
    'error 504' (numbers stripped first, no orphan)       → unchanged

    Indonesian has no meaningful single-letter words, so this is safe.
    """
    return _SINGLE_CHAR_PATTERN.sub(' ', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space and strip."""
    return _WHITESPACE_PATTERN.sub(' ', text).strip()


def normalize_text(text: str) -> str:
    """
    Full text normalization pipeline (Stage 4). Applies all cleaning
    operations in optimal order.

    Pipeline order (each step's output feeds the next):
        1. lowercase            → simplifies regex matching
        2. remove_urls          → before punctuation strip (URLs use . and /)
        3. remove_emojis        → before punctuation strip (emoticons use punct)
        4. remove_numbers       → digits removed
        5. collapse_repeated    → 'errorrrr' → 'errorr' (still has letters)
        6. remove_punctuation   → keep only letters + whitespace
        7. remove_single_chars  → strip orphan letters from steps 4 & 6
        8. normalize_whitespace → single spaces, trimmed

    NOTE: Stopword removal is intentionally NOT part of this pipeline.
    BERTopic best practice: keep stopwords for IndoBERT embedding (needs
    context), then filter them at the c-TF-IDF stage via CountVectorizer.

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    str
        Cleaned text ready for slang normalization (Stage 5).
    """
    text = lowercase_text(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_numbers(text)
    text = collapse_repeated_chars(text)
    text = remove_punctuation(text)
    text = remove_single_chars(text)
    text = normalize_whitespace(text)
    return text


def apply_normalization(
    df: pd.DataFrame,
    text_col: str = "review_text",
    output_col: str = "review_text_cleaned",
) -> pd.DataFrame:
    """
    Apply normalize_text() to a DataFrame column. Adds two new columns:
    `output_col` (cleaned text) and `word_count_after` (word count post-clean).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw text in `text_col`.
    text_col : str, default 'review_text'
        Source column with raw text.
    output_col : str, default 'review_text_cleaned'
        Destination column for cleaned text.

    Returns
    -------
    pd.DataFrame
        DataFrame with new cleaned-text and word-count columns.
    """
    df = df.copy()
    df[output_col] = df[text_col].apply(normalize_text)
    df["word_count_after"] = df[output_col].str.split().str.len()

    n_empty = (df[output_col] == "").sum()
    print(f"✅ Text normalization applied to {len(df):,} reviews.")
    print(f"   Empty after cleaning: {n_empty:,} reviews "
          f"(will be dropped at Stage 7 short-review filter).")
    print(f"   Word count (after) — mean: {df['word_count_after'].mean():.1f}, "
          f"median: {df['word_count_after'].median():.0f}, "
          f"max: {df['word_count_after'].max()}")
    return df
