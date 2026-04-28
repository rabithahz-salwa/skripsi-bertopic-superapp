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


# ==============================================================================
# Stage 5: Slang Normalization
# ==============================================================================

# Indonesian standard words protected from Salsabila mis-replacement.
# Identified via data-driven inspection of most-triggered slang mappings
# in wondr by BNI review data (Top 30 frequency analysis). Each entry
# below is a Salsabila mapping that corrupts standard Indonesian semantics
# after slang-column preprocessing collapses repeated characters.
#
# Without this blocklist, e.g. "aplikasinya error" would become
# "aplikasi dua duanya error" (catastrophic semantic corruption).
INDONESIAN_PROTECTED_WORDS: set[str] = {
    'apa',      # Salsabila: 'apaa' → 'diapa' (after collapse: 'apa' → 'diapa')
    'nya',      # Salsabila: 'nyaa' → 'dua-duanya' → catastrophic
    'sekali',   # Salsabila: 'sekalii' → 'sekali kali' (semantic shift)
    'bener',    # Salsabila: 'benerr' → 'benar benar' (mis-reduplikasi)
    'minta',    # Salsabila: 'mintaa' → 'meminta' (POS shift verb)
    'bukan',    # Salsabila: 'bukann' → 'bukannya' (semantic shift negation)
}


def load_slang_dict(
    salsabila_path: str,
    banking_ext_path: str | None = None,
    blocklist: set[str] | None = None,
) -> dict:
    """
    Load Salsabila colloquial Indonesian lexicon and merge with banking
    extension dictionary. Returns a slang→formal lookup dict.

    Processing applied:
    1. Pre-process slang keys with the same normalize_text() pipeline used
       on review text. This ensures lookups match (e.g., 'eeeehhhh' in dict
       becomes 'eehh', matching the collapsed form in cleaned reviews).
    2. Replace hyphens in formal values with spaces (e.g., 'kata-katanya'
       → 'kata katanya') since punctuation has been stripped from text.
    3. Drop entries where slang or formal becomes empty after cleaning.
    4. Drop "degrading reduplication mappings" where formal = slang × N
       (e.g., 'baru' → 'baru baru'), which are artifacts of preprocessing.
    5. Drop entries whose slang key is in `blocklist` — these are
       Indonesian standard words that get mis-replaced after preprocessing
       (e.g., 'apa' → 'diapa' from Salsabila entry 'apaa' → 'diapa').
       Identified via data-driven inspection of most-triggered replacements.
    6. Handle duplicate slang keys: prioritize Salsabila entries with
       In-dictionary == 1 (formal exists in KBBI), fallback to first.
    7. Banking extension entries override Salsabila entries on conflict
       (more domain-relevant for banking app reviews).

    Parameters
    ----------
    salsabila_path : str
        Path to Salsabila CSV (e.g., 'dictionaries/salsabila.csv').
    banking_ext_path : str, optional
        Path to banking extension CSV (e.g., 'dictionaries/banking_extension.csv').
    blocklist : set of str, optional
        Set of slang keys to exclude from the final dict. If None, uses
        the module-level INDONESIAN_PROTECTED_WORDS as the default. Pass
        an empty set ({}) to disable blocklisting entirely.

    Returns
    -------
    dict
        Mapping {slang_word: formal_word} ready for token-level lookup.
    """
    # Use module-level default if not provided. None vs empty set distinction:
    # None → use default protection; set() → explicitly disable blocklisting.
    if blocklist is None:
        blocklist = INDONESIAN_PROTECTED_WORDS

    # --- Load Salsabila ---
    df_sal = pd.read_csv(salsabila_path)
    df_sal = df_sal[["slang", "formal", "In-dictionary"]].copy()

    # Drop rows with NaN in slang or formal
    df_sal = df_sal.dropna(subset=["slang", "formal"])

    # Pre-process slang and formal columns
    df_sal["slang"] = df_sal["slang"].astype(str).apply(normalize_text)
    df_sal["formal"] = df_sal["formal"].astype(str).str.replace("-", " ", regex=False).apply(normalize_text)

    # Drop rows where cleaning made slang or formal empty
    df_sal = df_sal[(df_sal["slang"] != "") & (df_sal["formal"] != "")]

    # Drop self-mappings (slang == formal, no normalization needed)
    df_sal = df_sal[df_sal["slang"] != df_sal["formal"]]

    # Drop "degrading mappings" — where formal is the slang word repeated.
    # These arise as artifacts of our preprocessing: a Salsabila entry like
    # 'baruu' → 'baru baru' becomes 'baru' → 'baru baru' after collapsing
    # repeated chars in the slang column. This would corrupt clean text.
    def is_degrading(row):
        slang_word = row["slang"]
        formal_words = row["formal"].split()
        # Drop if formal is just slang repeated (e.g. 'baru baru', 'baru baru baru')
        return all(w == slang_word for w in formal_words) and len(formal_words) > 1

    n_before_filter = len(df_sal)
    df_sal = df_sal[~df_sal.apply(is_degrading, axis=1)]
    n_degrading = n_before_filter - len(df_sal)
    if n_degrading > 0:
        print(f"   Dropped {n_degrading} degrading reduplication mappings "
              f"(e.g., 'baru' → 'baru baru').")

    # Drop entries whose slang key is in blocklist. These protect Indonesian
    # standard words from being mis-replaced (identified via data-driven
    # inspection — see Bab 3 methodology section).
    if blocklist:
        n_before_block = len(df_sal)
        df_sal = df_sal[~df_sal["slang"].isin(blocklist)]
        n_blocked = n_before_block - len(df_sal)
        if n_blocked > 0:
            print(f"   Dropped {n_blocked} blocklisted entries (Indonesian "
                  f"standard words protected from mis-replacement).")

    # Handle duplicate keys: sort by In-dictionary descending, keep first.
    # This prioritizes entries where formal is in KBBI (In-dictionary == 1).
    df_sal = df_sal.sort_values("In-dictionary", ascending=False, na_position="last")
    df_sal = df_sal.drop_duplicates(subset=["slang"], keep="first")

    salsabila_dict = dict(zip(df_sal["slang"], df_sal["formal"]))
    print(f"✅ Loaded Salsabila lexicon: {len(salsabila_dict):,} entries "
          f"(after preprocessing & dedup).")

    # --- Load banking extension (optional) ---
    if banking_ext_path is None:
        return salsabila_dict

    df_ext = pd.read_csv(banking_ext_path)
    df_ext = df_ext.dropna(subset=["slang", "formal"])

    df_ext["slang"] = df_ext["slang"].astype(str).apply(normalize_text)
    df_ext["formal"] = df_ext["formal"].astype(str).str.replace("-", " ", regex=False).apply(normalize_text)

    df_ext = df_ext[(df_ext["slang"] != "") & (df_ext["formal"] != "")]
    df_ext = df_ext.drop_duplicates(subset=["slang"], keep="first")

    banking_dict = dict(zip(df_ext["slang"], df_ext["formal"]))
    print(f"✅ Loaded banking extension: {len(banking_dict):,} entries.")

    # Merge: banking overrides Salsabila on key conflict
    n_overrides = len(set(banking_dict.keys()) & set(salsabila_dict.keys()))
    merged = {**salsabila_dict, **banking_dict}
    print(f"✅ Merged dictionary: {len(merged):,} total entries "
          f"({n_overrides} banking entries override Salsabila).")

    return merged


def normalize_slang(text: str, slang_dict: dict) -> str:
    """
    Replace slang tokens with their formal equivalents. Tokenization is
    simple whitespace split — relies on Stage 4 normalization having
    already stripped punctuation.

    Parameters
    ----------
    text : str
        Cleaned review text (output of normalize_text()).
    slang_dict : dict
        Mapping {slang_word: formal_word}.

    Returns
    -------
    str
        Text with slang tokens replaced by formal forms.
    """
    if not text:
        return text

    tokens = text.split()
    normalized = [slang_dict.get(token, token) for token in tokens]
    return " ".join(normalized)


def apply_slang_normalization(
    df: pd.DataFrame,
    slang_dict: dict,
    text_col: str = "review_text_cleaned",
    output_col: str = "review_text_cleaned",
) -> pd.DataFrame:
    """
    Apply slang normalization to a DataFrame column. By default, overwrites
    `review_text_cleaned` in-place (i.e., text now reflects both Stage 4
    and Stage 5). Updates `word_count_after` to match.

    Logs how many reviews had at least one slang replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cleaned text in `text_col`.
    slang_dict : dict
        Slang→formal lookup dict from load_slang_dict().
    text_col : str, default 'review_text_cleaned'
        Source column with cleaned text.
    output_col : str, default 'review_text_cleaned'
        Destination column. Defaults to overwriting source.

    Returns
    -------
    pd.DataFrame
        DataFrame with slang-normalized text.
    """
    df = df.copy()

    original = df[text_col].copy()
    df[output_col] = original.apply(lambda t: normalize_slang(t, slang_dict))

    n_changed = (original != df[output_col]).sum()
    df["word_count_after"] = df[output_col].str.split().str.len()

    print(f"✅ Slang normalization applied to {len(df):,} reviews.")
    print(f"   Reviews with at least one slang replacement: {n_changed:,} "
          f"({n_changed / len(df) * 100:.1f}%).")
    print(f"   Word count (after) — mean: {df['word_count_after'].mean():.1f}, "
          f"median: {df['word_count_after'].median():.0f}, "
          f"max: {df['word_count_after'].max()}")
    return df


# ==============================================================================
# Stage 6: Short Review Filter
# ==============================================================================
# Drops reviews with fewer than min_words tokens after Stage 4-5 normalization.
# Placed AFTER slang normalization (Stage 5) because slang normalization can
# expand short forms (e.g., 'mbanking' → 'mobile banking', 'cs' → 'customer
# service'), changing word counts. Placed BEFORE language filtering (Stage 7)
# because fasttext language detection has low accuracy on short text.

def filter_short_reviews(
    df: pd.DataFrame,
    text_col: str = "review_text_cleaned",
    min_words: int = 5,
) -> pd.DataFrame:
    """
    Drop reviews with fewer than `min_words` tokens (whitespace-split) in
    `text_col`. Recomputes `word_count_after` to reflect current state.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cleaned text in `text_col`.
    text_col : str, default 'review_text_cleaned'
        Column to count words from.
    min_words : int, default 5
        Minimum word count to keep. Reviews with < min_words are dropped.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df = df.copy()

    # Recompute word count to ensure accuracy (defensive — in case slang
    # normalization changed word counts and word_count_after wasn't updated)
    df["word_count_after"] = df[text_col].fillna("").str.split().str.len()

    n_before = len(df)
    df_kept = df[df["word_count_after"] >= min_words].copy()
    n_after = len(df_kept)
    n_dropped = n_before - n_after

    print(f"✅ Short review filter (min_words={min_words}):")
    print(f"   Kept {n_after:,} / {n_before:,} reviews "
          f"(dropped {n_dropped:,} reviews with < {min_words} words).")
    print(f"   Word count (after) — mean: {df_kept['word_count_after'].mean():.1f}, "
          f"median: {df_kept['word_count_after'].median():.0f}, "
          f"max: {df_kept['word_count_after'].max()}")

    return df_kept


# ==============================================================================
# Stage 7 (DEPRECATED): Language Filtering with fasttext lid.176
# ==============================================================================
# DECISION: Language filter NOT used in final preprocessing pipeline.
#
# Rationale (data-driven decision based on audit):
#   1. Initial audit (with Stage 6 short filter): 471 reviews flagged Tier 3
#      (4.6%). Sample inspection showed 5/5 were valid Indonesian misclassified
#      due to fasttext low accuracy on short text.
#   2. After reordering (Stage 6 short filter BEFORE language filter): Tier 3
#      dropped to 123 (1.4%) — improvement, but...
#   3. Deeper audit of remaining 123 Tier 3:
#        - 84% had low confidence (<0.5) → uncertain predictions
#        - 9/10 high-confidence Tier 3 samples were FALSE POSITIVES
#          (valid Indonesian misclassified as ms/en due to banking loanwords,
#          typos, regional slang, and Indonesian-Malay similarity)
#   4. Conclusion: fasttext lid.176 has systemic bias on Indonesian banking
#      domain. False positive rate (drop valid Indonesian) > false negative
#      rate (keep non-Indonesian).
#
# Since dataset is pre-filtered to rating 1-2 of Indonesian banking apps
# (where >95% is naturally Indonesian), and BERTopic clusters non-Indonesian
# outliers naturally, automated language filtering is omitted.
#
# Functions BELOW are KEPT for:
#   - Reproducibility of the audit that led to this decision
#   - Reference for future research on different domains
#   - Documentation in thesis Bab 3 (methodology rationale)
#
# Strategy used in audit: 3-tier filtering, lenient for code-switching
#   TIER 1 (KEEP): top_lang == 'id' AND confidence >= 0.5
#   TIER 2 (KEEP but FLAG):
#       Case A: top_lang == 'id' but confidence < 0.5
#       Case B: top_lang != 'id' but 'id' in top 3 with confidence >= 0.2
#   TIER 3 (DROP): top_lang != 'id' and 'id' not in top 3

# Confidence thresholds (centralized constants for transparency)
_TIER1_CONF_THRESHOLD = 0.5
_TIER2B_CONF_THRESHOLD = 0.2
_TOP_K_PREDICTIONS = 3


def load_lang_detector(model_path: str = "lid.176.bin"):
    """
    Load the fasttext language identification model. This is an expensive
    operation (~5-10 seconds for 126MB model), so call once and reuse.

    Also patches a NumPy 2.x compatibility issue: fasttext-wheel 0.9.2 uses
    the deprecated `np.array(..., copy=False)` pattern internally, which
    raises ValueError on NumPy >= 2.0. We monkey-patch model.predict() to
    use np.asarray() instead.

    Parameters
    ----------
    model_path : str, default 'lid.176.bin'
        Path to the fasttext lid.176.bin model file.

    Returns
    -------
    fasttext._FastText
        Loaded fasttext model with .predict() method (NumPy 2.x compatible).
    """
    import fasttext
    import numpy as np

    # Suppress the deprecation warning from fasttext's load_model
    fasttext.FastText.eprint = lambda *args, **kwargs: None

    model = fasttext.load_model(model_path)

    # Patch for NumPy 2.x compatibility:
    # The high-level model.predict() formats raw output via
    #   np.array(probs, copy=False)
    # which raises ValueError on NumPy >= 2.0. We replace it with a
    # wrapper that uses np.asarray() instead.
    #
    # Note: model.f.predict() (low-level C++ binding) returns a
    # list of (prob, label) tuples — verified empirically:
    #   [(0.987, '__label__id'), (0.003, '__label__eu'), ...]
    # We unpack and reformat to match the original predict() API:
    #   (labels: tuple of str, probs: np.ndarray of float)
    _f_predict = model.f.predict

    def patched_predict(text, k=1, threshold=0.0, on_unicode_error="strict"):
        if isinstance(text, list):
            # Batch input
            all_labels, all_probs = [], []
            for t in text:
                raw = _f_predict(t, k, threshold, on_unicode_error)
                probs_t = np.asarray([item[0] for item in raw])
                labels_t = tuple(item[1] for item in raw)
                all_labels.append(labels_t)
                all_probs.append(probs_t)
            return all_labels, all_probs
        else:
            raw = _f_predict(text, k, threshold, on_unicode_error)
            probs = np.asarray([item[0] for item in raw])
            labels = tuple(item[1] for item in raw)
            return labels, probs

    # Replace predict on the model object
    model.predict = patched_predict

    print(f"✅ Loaded fasttext language detector from {model_path}")
    return model


def detect_language_tier(text: str, model) -> tuple[str, float, int]:
    """
    Run fasttext language detection on a single text and assign a tier
    based on the 3-tier strategy (see module-level documentation).

    Parameters
    ----------
    text : str
        Cleaned review text (output of Stage 5).
    model : fasttext._FastText
        Loaded fasttext model from load_lang_detector().

    Returns
    -------
    tuple of (str, float, int)
        (top_lang, top_confidence, tier) where:
        - top_lang: ISO 639-1 code (e.g. 'id', 'en', 'ms') without
          the '__label__' prefix
        - top_confidence: float in [0, 1]
        - tier: 1 (keep, confident id), 2 (keep, flagged), or 3 (drop)
    """
    # Empty text: cannot detect language, treat as Tier 3 (will be dropped)
    if not text or not text.strip():
        return ("", 0.0, 3)

    # fasttext doesn't accept newlines in input — defensive cleanup
    text_clean = text.replace("\n", " ").strip()

    # Get top K predictions
    labels, probs = model.predict(text_clean, k=_TOP_K_PREDICTIONS)
    # Strip '__label__' prefix from labels
    langs = [lbl.replace("__label__", "") for lbl in labels]
    top_lang, top_conf = langs[0], float(probs[0])

    # Tier 1: confident Indonesian
    if top_lang == "id" and top_conf >= _TIER1_CONF_THRESHOLD:
        return (top_lang, top_conf, 1)

    # Tier 2 Case A: top is Indonesian but low confidence
    if top_lang == "id" and top_conf < _TIER1_CONF_THRESHOLD:
        return (top_lang, top_conf, 2)

    # Tier 2 Case B: top is not Indonesian, but Indonesian is in top K
    # with sufficient confidence
    for lang, prob in zip(langs[1:], probs[1:]):
        if lang == "id" and float(prob) >= _TIER2B_CONF_THRESHOLD:
            return (top_lang, top_conf, 2)

    # Tier 3: clearly non-Indonesian
    return (top_lang, top_conf, 3)


def apply_language_detection(
    df: pd.DataFrame,
    model,
    text_col: str = "review_text_cleaned",
) -> pd.DataFrame:
    """
    Apply language detection to all rows in DataFrame. Adds three columns:
    `lang_top`, `lang_top_conf`, `lang_tier`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cleaned text in `text_col`.
    model : fasttext._FastText
        Loaded fasttext model.
    text_col : str, default 'review_text_cleaned'
        Source column with text for detection.

    Returns
    -------
    pd.DataFrame
        DataFrame with new lang_top, lang_top_conf, lang_tier columns.
    """
    df = df.copy()

    # Apply row-by-row. tqdm progress bar would be nice but adds complexity;
    # 10K rows finishes in a few seconds anyway.
    results = df[text_col].apply(lambda t: detect_language_tier(t, model))

    df["lang_top"] = results.apply(lambda r: r[0])
    df["lang_top_conf"] = results.apply(lambda r: r[1])
    df["lang_tier"] = results.apply(lambda r: r[2])

    print(f"✅ Language detection applied to {len(df):,} reviews.")
    print(f"   Tier distribution:")
    tier_counts = df["lang_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        label = {1: "KEEP confident-id", 2: "KEEP flagged", 3: "DROP non-id"}[tier]
        print(f"     Tier {tier} ({label}): {count:,} ({pct:.1f}%)")

    # Top-5 non-id languages detected (for audit insight)
    non_id = df[df["lang_top"] != "id"]
    if len(non_id) > 0:
        top_other_langs = non_id["lang_top"].value_counts().head(5)
        print(f"   Top 5 non-Indonesian languages detected:")
        for lang, count in top_other_langs.items():
            print(f"     '{lang}': {count:,}")

    return df


def filter_by_language_tier(
    df: pd.DataFrame,
    audit_path: str | None = None,
    keep_tiers: tuple = (1, 2),
) -> pd.DataFrame:
    """
    Filter DataFrame by language tier. Tier 3 reviews are dropped, optionally
    saved to an audit CSV for manual inspection of false negatives.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with `lang_tier` column from apply_language_detection().
    audit_path : str, optional
        If provided, saves dropped (Tier 3) reviews to this CSV path.
        Recommended: 'data/processed/wondr_tier3_dropped.csv'.
    keep_tiers : tuple, default (1, 2)
        Tiers to keep. Tier 3 is dropped by default.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (Tier 1 + Tier 2 by default).
    """
    n_before = len(df)
    df_dropped = df[~df["lang_tier"].isin(keep_tiers)].copy()
    df_kept = df[df["lang_tier"].isin(keep_tiers)].copy()
    n_dropped = len(df_dropped)

    print(f"✅ Language tier filter (keep tiers {keep_tiers}):")
    print(f"   Kept {len(df_kept):,} / {n_before:,} reviews "
          f"(dropped {n_dropped:,} Tier 3).")

    # Save Tier 3 audit file
    if audit_path is not None and n_dropped > 0:
        audit_cols = [
            "review_id", "review_text", "review_text_cleaned",
            "lang_top", "lang_top_conf",
        ]
        # Only keep columns that exist (defensive)
        audit_cols = [c for c in audit_cols if c in df_dropped.columns]
        df_dropped[audit_cols].to_csv(audit_path, index=False)
        print(f"   Tier 3 reviews saved to: {audit_path}")

    return df_kept


# ==============================================================================
# Stage 8: Save Outputs (BERTopic-ready & Full Audit)
# ==============================================================================

def save_preprocessed_outputs(
    df: pd.DataFrame,
    bertopic_path: str,
    full_path: str,
    bertopic_cols: list | None = None,
) -> None:
    """
    Save final preprocessed DataFrame to two CSV files:

    1. BERTopic-ready (slim) — minimal columns for topic modeling pipeline.
       Default columns: review_id, review_text_cleaned, relative_month,
       relative_week, date_wib, rating.

    2. Full audit — all columns retained, useful for inspection and
       validation in later analysis stages.

    Parameters
    ----------
    df : pd.DataFrame
        Final preprocessed DataFrame (output of Stage 6 short filter).
    bertopic_path : str
        Output path for BERTopic-ready CSV (e.g., 'data/processed/wondr_bertopic.csv').
    full_path : str
        Output path for full audit CSV (e.g., 'data/processed/wondr_full.csv').
    bertopic_cols : list, optional
        Override default column list for BERTopic version.

    Returns
    -------
    None
        Saves two files. Prints save confirmation and shape.
    """
    if bertopic_cols is None:
        bertopic_cols = [
            "review_id",
            "review_text_cleaned",
            "relative_month",
            "relative_week",
            "date_wib",
            "rating",
        ]

    # Defensive: only keep columns that actually exist in the df
    available_cols = [c for c in bertopic_cols if c in df.columns]
    missing_cols = set(bertopic_cols) - set(available_cols)
    if missing_cols:
        print(f"⚠️  Missing columns in df, skipping: {missing_cols}")

    # Save slim BERTopic version
    df_bertopic = df[available_cols].copy()
    df_bertopic.to_csv(bertopic_path, index=False)
    print(f"✅ BERTopic-ready saved: {bertopic_path}")
    print(f"   Shape: {df_bertopic.shape}, Columns: {list(df_bertopic.columns)}")

    # Save full audit version
    df.to_csv(full_path, index=False)
    print(f"\n✅ Full audit saved: {full_path}")
    print(f"   Shape: {df.shape}, Columns: {len(df.columns)} columns")
