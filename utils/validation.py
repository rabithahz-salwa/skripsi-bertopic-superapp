"""
Validation utilities for thesis: Temporal Topic Modeling on Indonesian
banking superapp reviews (wondr by BNI vs BYOND by BSI).

Stage 4: Manual validation of rating-based complaint filter using stratified
sampling and weighted proportion estimation (Cochran, 1977). Supports the
classification framework adapted from Maalej et al. (2016) with one extension
category for non-technical complaints in the banking domain.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# Categories used for manual labeling, adapted from Maalej et al. (2016)
# with one extension for the banking superapp domain.
TECHNICAL_CATEGORIES = [
    "Bug Report",
    "Feature Request",
    "User Experience",
]

NON_TECHNICAL_CATEGORIES = [
    "Rating (Praise/Dispraise)",
    "Non-Technical Complaint",
]

ALL_CATEGORIES = TECHNICAL_CATEGORIES + NON_TECHNICAL_CATEGORIES


def stratified_sample_by_month(
    df: pd.DataFrame,
    n_per_stratum: int = 25,
    stratum_col: str = "relative_month",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draw an equal-allocation stratified random sample by relative month.

    Implements equal allocation rather than proportional allocation to ensure
    that every relative month (1-12 post-launch) is represented with the same
    sample size. This supports per-stratum analysis of the validity rate
    across the 12-month observation window. Note that point estimates of
    population proportions must be computed using a weighted estimator
    (see compute_weighted_proportion) because the sample is non-proportional.

    Parameters
    ----------
    df : pd.DataFrame
        Full filtered DataFrame (rating 1-2) with a relative_month column
        ranging from 1 to 12.
    n_per_stratum : int, default 25
        Number of reviews to sample from each relative month. The default of
        25 yields 25 x 12 = 300 total samples, matching the proposal.
    stratum_col : str, default 'relative_month'
        Name of the column used as the stratification variable.
    random_state : int, default 42
        Random seed for reproducibility. Convention: 42 (de facto standard
        in the scikit-learn / pandas ecosystem).

    Returns
    -------
    pd.DataFrame
        Stratified sample of shape (n_per_stratum * n_strata, n_columns).
        Rows are shuffled (not ordered by stratum) to prevent calibration
        drift during manual labeling.

    Raises
    ------
    ValueError
        If any stratum has fewer rows than n_per_stratum, sampling without
        replacement is infeasible.

    Notes
    -----
    Uses sampling without replacement (replace=False). This is appropriate
    for finite-population inference: each review can be drawn at most once.

    References
    ----------
    Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley. Ch. 5.
    """
    # Pre-flight check: every stratum must have enough rows.
    stratum_sizes = df.groupby(stratum_col).size()
    too_small = stratum_sizes[stratum_sizes < n_per_stratum]
    if len(too_small) > 0:
        raise ValueError(
            f"The following strata have fewer than {n_per_stratum} rows and "
            f"cannot be sampled without replacement:\n{too_small.to_string()}"
        )

    # Sample n_per_stratum rows from each stratum independently.
    # We sample stratum-by-stratum in a comprehension instead of using
    # groupby().apply() because newer pandas versions (>= 2.2) drop the
    # grouping column from the result of apply with non-trivial behavior;
    # this pattern is more explicit and version-stable.
    sample = pd.concat(
        [
            group.sample(n=n_per_stratum, random_state=random_state)
            for _, group in df.groupby(stratum_col)
        ],
        ignore_index=True,
    )

    # Shuffle the entire sample so that rows from different months are
    # interleaved. This avoids calibration drift during sequential labeling
    # (e.g., judgment becoming stricter as the labeler progresses).
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sample


def prepare_for_sheets(
    sample_df: pd.DataFrame,
    id_col: str = "review_id",
    rating_col: str = "rating",
    text_col: str = "review_text",
    stratum_col: str = "relative_month",
) -> pd.DataFrame:
    """
    Reformat a stratified sample for upload to Google Sheets manual labeling.

    Reorders columns so the labeler sees the most relevant context first
    (relative month, rating, review text) and appends two empty columns
    (category, notes) for manual annotation. Auxiliary columns are kept at
    the end for traceability but can be hidden in the Sheets UI.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Output of stratified_sample_by_month.
    id_col : str, default 'review_id'
        Name of the unique review identifier column. Required for merging
        the labeled CSV back to the full dataset.
    rating_col : str, default 'rating'
        Name of the star rating column (expected values: 1 or 2 only).
    text_col : str, default 'review_text'
        Name of the cleaned review text column to be labeled.
    stratum_col : str, default 'relative_month'
        Name of the relative month column (1-12 post-launch).

    Returns
    -------
    pd.DataFrame
        Sample with columns ordered as:
        [stratum_col, rating_col, text_col, 'category', 'notes', id_col, ...]
        where 'category' and 'notes' are empty strings ready for manual entry.
    """
    # Defensive copy so we never mutate the caller's DataFrame.
    df = sample_df.copy()

    # Add empty annotation columns. Use empty string (not NaN) so that
    # Google Sheets renders them as truly empty cells without the
    # spurious "NaN" string artifact that pandas-to-CSV sometimes produces.
    df["category"] = ""
    df["notes"] = ""

    # Define the preferred column order: labeler-facing columns first,
    # then annotation columns, then the ID for traceability, then the rest.
    primary_cols = [stratum_col, rating_col, text_col, "category", "notes", id_col]
    other_cols = [c for c in df.columns if c not in primary_cols]
    ordered_cols = primary_cols + other_cols

    # Validate that all primary columns actually exist (defensive against
    # typos or schema drift in the input data).
    missing = [c for c in primary_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns missing from sample_df: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[ordered_cols]


def compute_weighted_proportion(
    labeled_df: pd.DataFrame,
    population_df: pd.DataFrame,
    category_col: str = "category",
    stratum_col: str = "relative_month",
) -> pd.DataFrame:
    """
    Compute weighted population proportions for each label category.

    For an equal-allocation stratified sample, the simple sample proportion
    p_hat = count_c / n_total is a biased estimator of the true population
    proportion when stratum sizes differ. This function applies the
    Cochran (1977) weighted estimator:

        p_hat_weighted_c = sum_m (N_m / N) * (count_c_in_m / n_m)

    where N_m is the population size in stratum m, N is total population,
    count_c_in_m is the labeled count of category c in stratum m, and n_m
    is the sample size in stratum m.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Sample DataFrame after manual labeling. Must contain at minimum
        the category_col and stratum_col columns.
    population_df : pd.DataFrame
        Full filtered population DataFrame (rating 1-2). Used to compute
        N_m for each stratum.
    category_col : str, default 'category'
        Name of the manual label column.
    stratum_col : str, default 'relative_month'
        Name of the relative month column (must match in both DataFrames).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by category, with columns:
        - raw_count : count of category in the sample
        - raw_pct   : simple sample proportion (count / sample_total)
        - weighted_pct : Cochran-weighted population proportion estimate
        - is_technical : bool, whether the category counts as technical

    Notes
    -----
    The raw and weighted percentages will differ when stratum populations
    are heterogeneous. The weighted estimate should be reported as the
    primary result; the gap between raw and weighted reveals how much
    the equal-allocation design corrected stratum imbalance.

    References
    ----------
    Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley. Eq. 5.6.
    """
    # Compute stratum population sizes N_m and total population N.
    stratum_pop = population_df.groupby(stratum_col).size().rename("N_m")
    total_pop = stratum_pop.sum()
    weights = (stratum_pop / total_pop).rename("weight")  # N_m / N for each m

    # For each (stratum, category) pair, count the sample occurrences,
    # then divide by total sample size in that stratum to get within-stratum
    # proportion p_c_in_m = count_c_in_m / n_m.
    counts_by_stratum = (
        labeled_df.groupby([stratum_col, category_col]).size().unstack(fill_value=0)
    )
    sample_size_per_stratum = counts_by_stratum.sum(axis=1)  # n_m for each m
    within_stratum_props = counts_by_stratum.div(sample_size_per_stratum, axis=0)

    # Weighted sum across strata: sum_m (N_m / N) * p_c_in_m for each c.
    # weights and within_stratum_props share the stratum index; align then sum.
    weighted_pct = (within_stratum_props.mul(weights, axis=0)).sum(axis=0) * 100

    # Also compute the (biased) simple proportion for comparison.
    raw_count = labeled_df[category_col].value_counts()
    raw_pct = (raw_count / len(labeled_df)) * 100

    # Assemble the summary table.
    summary = pd.DataFrame({
        "raw_count": raw_count,
        "raw_pct": raw_pct,
        "weighted_pct": weighted_pct,
    }).fillna(0)

    summary["is_technical"] = summary.index.isin(TECHNICAL_CATEGORIES)

    # Sort: technical categories first, then by weighted_pct descending.
    summary = summary.sort_values(
        by=["is_technical", "weighted_pct"], ascending=[False, False]
    )

    return summary


def validation_summary(
    labeled_df: pd.DataFrame,
    population_df: pd.DataFrame,
    category_col: str = "category",
    stratum_col: str = "relative_month",
) -> dict:
    """
    Produce a complete validation summary for reporting in Bab IV.

    Wraps compute_weighted_proportion with additional aggregate statistics:
    overall validity rate (proportion of technical categories) and per-stratum
    validity rate (to assess temporal stability of the rating filter).

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Sample DataFrame after manual labeling.
    population_df : pd.DataFrame
        Full filtered population DataFrame (rating 1-2).
    category_col : str, default 'category'
        Name of the manual label column.
    stratum_col : str, default 'relative_month'
        Name of the relative month column.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'per_category' : DataFrame from compute_weighted_proportion
        - 'overall_validity_pct' : float, weighted % of technical categories
        - 'per_month_validity_pct' : Series indexed by relative_month,
          giving the simple proportion of technical reviews per stratum
        - 'sample_total' : int, total number of labeled reviews
        - 'population_total' : int, total population size
    """
    per_category = compute_weighted_proportion(
        labeled_df, population_df, category_col, stratum_col
    )

    # Overall validity rate: sum of weighted_pct across technical categories.
    overall_validity = per_category.loc[
        per_category["is_technical"], "weighted_pct"
    ].sum()

    # Per-month validity rate: simple proportion within each stratum.
    # This is OK to report as a simple proportion because within a stratum
    # the design is a simple random sample.
    is_tech_mask = labeled_df[category_col].isin(TECHNICAL_CATEGORIES)
    per_month_validity = (
        is_tech_mask.groupby(labeled_df[stratum_col]).mean() * 100
    ).rename("validity_pct")

    return {
        "per_category": per_category,
        "overall_validity_pct": float(overall_validity),
        "per_month_validity_pct": per_month_validity,
        "sample_total": int(len(labeled_df)),
        "population_total": int(len(population_df)),
    }
