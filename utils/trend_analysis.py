"""
Trend analysis module untuk temporal topic modeling.

Mengimplementasikan uji tren Mann-Kendall (modified Hamed-Rao) + Sen's slope
pada prevalensi topik mingguan, sesuai BAB 3 metodologi:
- Granularity: mingguan (7-day blocks dari launch date, cap 52 minggu)
- Test: Hamed-Rao modified Mann-Kendall (handle autokorelasi temporal)
- Klasifikasi: emerging (slope > 0 + significant), declining (slope < 0 + significant),
                stable (not significant)
- Significance level: alpha = 0.05 (Bonferroni correction acknowledged sebagai limitation)

Workflow:
1. compute_weekly_series_per_topic() -> dict berisi 'frequency' & 'proportion' DataFrames
2. apply_mann_kendall_per_topic() -> DataFrame hasil test per topic
3. plot_trend_summary() -> forest plot Sen's slope
4. plot_significant_trends_lines() -> lineplot trend yang significant

References:
- Mann (1945) Econometrica 13(3)
- Kendall (1975) Rank Correlation Methods
- Sen (1968) JASA 63
- Hamed & Rao (1998) J. Hydrol. 204 — modified MK untuk autokorelasi
- Yue, Pilon, Cavadias (2002) J. Hydrol. 259
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
from pathlib import Path


# ============================================================
# 1. WEEKLY AGGREGATION
# ============================================================

def compute_weekly_series_per_topic(
    df_clean,
    launch_date,
    n_weeks=52,
    date_col='date'
):
    """
    Aggregate df_clean ke series mingguan per topic (frequency + proportion).
    
    Mengikuti BAB 3 Langkah 5a: prevalensi topik mingguan, ~52 titik data.
    Minggu didefinisikan sebagai 7-day blocks dari launch_date:
    - Minggu 1: hari 0-6 post-launch
    - Minggu 2: hari 7-13
    - ...
    - Minggu 52: hari 357-363
    Hari 364+ di-drop (cap di 52 minggu).
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Output filter rating 1-2 dan topic != -1.
        Wajib ada kolom 'topic' dan kolom date (default 'date').
    launch_date : str atau pd.Timestamp
        Tanggal peluncuran app (format 'YYYY-MM-DD' atau Timestamp).
        wondr: '2024-07-05', BYOND: '2024-11-09'
    n_weeks : int, default 52
        Jumlah minggu yang di-cover (cap window).
    date_col : str, default 'date'
        Nama kolom date di df_clean.
    
    Returns
    -------
    dict
        {
            'frequency': DataFrame (n_weeks x n_topics),
                index = relative_week (1..52), columns = topic_id, values = count
            'proportion': DataFrame (n_weeks x n_topics),
                index = relative_week (1..52), columns = topic_id,
                values = count / total_complaints_that_week
            'weekly_totals': Series (n_weeks,),
                total complaints per week (untuk validasi/sanity check)
        }
    """
    # Validasi input
    if date_col not in df_clean.columns:
        raise ValueError(f"Kolom '{date_col}' tidak ada di df_clean. "
                         f"Available: {list(df_clean.columns)}")
    if 'topic' not in df_clean.columns:
        raise ValueError("Kolom 'topic' tidak ada di df_clean.")
    
    # Convert launch_date ke Timestamp
    launch = pd.Timestamp(launch_date)
    
    # Copy supaya tidak modify original
    df = df_clean.copy()
    
    # Pastikan date column bertipe datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Hitung relative_week: floor((date - launch_date).days / 7) + 1
    # +1 supaya minggu pertama = 1, bukan 0 (lebih intuitif untuk laporan)
    days_since_launch = (df[date_col] - launch).dt.days
    df['relative_week'] = (days_since_launch // 7) + 1
    
    # Filter ke window: 1 <= relative_week <= n_weeks
    # Drop minggu < 1 (kalau ada review pre-launch karena timezone) atau > n_weeks
    df = df[(df['relative_week'] >= 1) & (df['relative_week'] <= n_weeks)].copy()
    
    # Aggregate frequency: count per (topic, relative_week)
    freq = (df.groupby(['relative_week', 'topic'])
              .size()
              .unstack(fill_value=0))
    
    # Reindex supaya semua minggu 1..n_weeks ada (fill 0 untuk minggu kosong)
    freq = freq.reindex(range(1, n_weeks + 1), fill_value=0)
    
    # Pastikan column order konsisten (sorted by topic_id)
    freq = freq.reindex(sorted(freq.columns), axis=1)
    
    # Compute weekly totals (total keluhan per minggu, semua topic)
    weekly_totals = freq.sum(axis=1)
    
    # Compute proportion: count / weekly_total
    # Handle division by zero: minggu dengan 0 keluhan total -> proportion = 0 (bukan NaN)
    # Ini keputusan: 0 lebih informatif dari NaN untuk MK test (tidak akan di-skip)
    prop = freq.div(weekly_totals.replace(0, np.nan), axis=0).fillna(0)
    
    return {
        'frequency': freq,
        'proportion': prop,
        'weekly_totals': weekly_totals
    }

def compute_weekly_series_from_relative_week(
    df_clean,
    n_weeks=52,
    week_col='relative_week'
):
    """
    Aggregate ke series mingguan menggunakan kolom relative_week yang sudah
    precomputed di preprocessing (lebih aman, konsisten dengan Phase 1-4).
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Wajib ada kolom 'topic' dan 'relative_week'.
    n_weeks : int, default 52
    week_col : str, default 'relative_week'
    
    Returns
    -------
    dict : {'frequency', 'proportion', 'weekly_totals'}
        Sama struktur dengan compute_weekly_series_per_topic().
    """
    if week_col not in df_clean.columns:
        raise ValueError(f"Kolom '{week_col}' tidak ada di df_clean.")
    if 'topic' not in df_clean.columns:
        raise ValueError("Kolom 'topic' tidak ada di df_clean.")
    
    df = df_clean.copy()
    
    # Filter ke window: 1 <= relative_week <= n_weeks
    df = df[(df[week_col] >= 1) & (df[week_col] <= n_weeks)].copy()
    
    # Aggregate frequency
    freq = (df.groupby([week_col, 'topic'])
              .size()
              .unstack(fill_value=0))
    
    freq = freq.reindex(range(1, n_weeks + 1), fill_value=0)
    freq = freq.reindex(sorted(freq.columns), axis=1)
    
    weekly_totals = freq.sum(axis=1)
    prop = freq.div(weekly_totals.replace(0, np.nan), axis=0).fillna(0)
    
    return {
        'frequency': freq,
        'proportion': prop,
        'weekly_totals': weekly_totals
    }


def apply_fdr_correction(mk_results, alpha=0.05, method='fdr_bh'):
    """
    Apply FDR-BH (Benjamini-Hochberg, 1995) correction ke hasil MK test.
    Reject null hypothesis berdasarkan adjusted p-value.
    
    Parameters
    ----------
    mk_results : pd.DataFrame
        Output dari apply_mann_kendall_per_topic().
    alpha : float, default 0.05
        FDR threshold (q-value).
    method : str, default 'fdr_bh'
        Method untuk multipletests. 'fdr_bh' = Benjamini-Hochberg standard.
    
    Returns
    -------
    pd.DataFrame
        Copy dari mk_results dengan kolom tambahan:
        - MK_p_value_adj: BH-adjusted p-value
        - Significant_FDR: boolean significant setelah FDR correction
        - Classification_FDR: emerging/declining/stable berdasarkan FDR
    """
    from statsmodels.stats.multitest import multipletests
    
    df = mk_results.copy()
    
    # Handle NaN p-values (dari topic dengan insufficient_variance)
    valid_mask = df['MK_p_value'].notna()
    
    # Initialize columns
    df['MK_p_value_adj'] = np.nan
    df['Significant_FDR'] = False
    df['Classification_FDR'] = 'stable'
    
    if valid_mask.sum() > 0:
        # Apply FDR-BH ke valid p-values saja
        valid_pvals = df.loc[valid_mask, 'MK_p_value'].values
        
        rejected, p_adj, _, _ = multipletests(
            valid_pvals,
            alpha=alpha,
            method=method
        )
        
        df.loc[valid_mask, 'MK_p_value_adj'] = p_adj
        df.loc[valid_mask, 'Significant_FDR'] = rejected
        
        # Re-classify berdasarkan FDR significance
        for idx in df[valid_mask].index:
            if df.at[idx, 'Significant_FDR']:
                if df.at[idx, 'Sen_slope'] > 0:
                    df.at[idx, 'Classification_FDR'] = 'emerging'
                else:
                    df.at[idx, 'Classification_FDR'] = 'declining'
            else:
                df.at[idx, 'Classification_FDR'] = 'stable'
    
    return df

# ============================================================
# 2. MANN-KENDALL TEST
# ============================================================

def classify_trend(trend, slope, alpha=0.05, p_value=None):
    """
    Klasifikasi trend ke 'emerging' / 'declining' / 'stable' sesuai BAB 3.
    
    Parameters
    ----------
    trend : str
        Output dari pymannkendall ('increasing', 'decreasing', 'no trend').
    slope : float
        Sen's slope.
    alpha : float, default 0.05
    p_value : float, optional
        Untuk override klasifikasi (kalau mau strict berdasarkan p < alpha).
    
    Returns
    -------
    str : 'emerging', 'declining', atau 'stable'
    """
    # pymannkendall sudah return 'increasing'/'decreasing'/'no trend' berdasarkan alpha-nya sendiri,
    # tapi kita re-check pakai p_value supaya konsisten dengan alpha kita
    if p_value is not None:
        if p_value >= alpha:
            return 'stable'
        elif slope > 0:
            return 'emerging'
        else:
            return 'declining'
    
    # Fallback: pakai output trend langsung dari pymannkendall
    if trend == 'increasing':
        return 'emerging'
    elif trend == 'decreasing':
        return 'declining'
    else:
        return 'stable'


def apply_mann_kendall_per_topic(
    series_df,
    topic_labels,
    series_type='proportion',
    alpha=0.05
):
    """
    Apply Hamed-Rao modified Mann-Kendall + Sen's slope per topic.
    
    Loop semua topic di kolom series_df, jalankan test, return tabel hasil.
    
    Parameters
    ----------
    series_df : pd.DataFrame
        Output dari compute_weekly_series_per_topic()['proportion'] atau ['frequency'].
        Index = relative_week, columns = topic_id.
    topic_labels : dict
        Mapping topic_id -> label string. Misal {0: 'Mobile Banking', 1: 'Verifikasi wajah', ...}
    series_type : str, default 'proportion'
        'proportion' atau 'frequency'. Untuk dokumentasi di output table.
    alpha : float, default 0.05
        Significance level.
    
    Returns
    -------
    pd.DataFrame
        Kolom: Topic, Label, Series_Type, MK_trend, MK_p_value, MK_z, 
               Sen_slope, Sen_intercept, Tau, Significant, Classification
        n_row = jumlah topic di series_df.
    """
    results = []
    
    for topic_id in series_df.columns:
        series = series_df[topic_id].values  # numpy array, length = n_weeks
        
        # Edge case: kalau series semua 0 atau semua sama (no variance),
        # Mann-Kendall akan return 'no trend' dengan p=1.0. Aman, tidak perlu skip.
        # Tapi check minimal: ada minimal 4 unique values (rule of thumb).
        n_unique = len(np.unique(series))
        if n_unique < 3:
            # Topic terlalu sparse, MK tidak bermakna
            results.append({
                'Topic': topic_id,
                'Label': topic_labels.get(topic_id, f'Topic {topic_id}'),
                'Series_Type': series_type,
                'MK_trend': 'insufficient_variance',
                'MK_p_value': np.nan,
                'MK_z': np.nan,
                'Sen_slope': np.nan,
                'Sen_intercept': np.nan,
                'Tau': np.nan,
                'Significant': False,
                'Classification': 'stable',  # default ke stable
            })
            continue
        
        # Apply Hamed-Rao modified MK
        # Output: trend, h (boolean significant), p, z, Tau, s, var_s, slope, intercept
        result = mk.hamed_rao_modification_test(series, alpha=alpha)
        
        # Klasifikasi
        classification = classify_trend(
            trend=result.trend,
            slope=result.slope,
            alpha=alpha,
            p_value=result.p
        )
        
        results.append({
            'Topic': topic_id,
            'Label': topic_labels.get(topic_id, f'Topic {topic_id}'),
            'Series_Type': series_type,
            'MK_trend': result.trend,
            'MK_p_value': result.p,
            'MK_z': result.z,
            'Sen_slope': result.slope,
            'Sen_intercept': result.intercept,
            'Tau': result.Tau,
            'Significant': result.h,  # boolean
            'Classification': classification,
        })
    
    return pd.DataFrame(results)


# ============================================================
# 3. VISUALIZATION
# ============================================================

def plot_trend_summary(
    mk_results,
    fig_dir,
    app_name,
    series_type='proportion',
    alpha=0.05,
    figsize=(10, 8),
    save=True
):
    """
    Forest plot Sen's slope per topic.
    
    - x-axis: Sen's slope (centered di 0)
    - y-axis: topic label (sorted by slope)
    - Filled marker = significant (p < alpha)
    - Hollow marker = not significant
    - Color: red (declining/negative) vs green (emerging/positive) vs gray (stable)
    
    Parameters
    ----------
    mk_results : pd.DataFrame
        Output dari apply_mann_kendall_per_topic().
    fig_dir : Path atau str
        Directory untuk save figure.
    app_name : str
        'wondr' atau 'byond' (untuk filename).
    series_type : str, default 'proportion'
        Untuk title dan filename.
    alpha : float, default 0.05
    figsize : tuple, default (10, 8)
    save : bool, default True
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by Sen_slope (declining ke emerging, kiri ke kanan)
    df_sorted = mk_results.sort_values('Sen_slope', ascending=True).reset_index(drop=True)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Y positions
    y_pos = np.arange(len(df_sorted))
    
    # Color mapping berdasarkan Classification
    color_map = {
        'emerging': '#2ca02c',   # green
        'declining': '#d62728',  # red
        'stable': '#7f7f7f',     # gray
    }
    colors = [color_map.get(c, '#7f7f7f') for c in df_sorted['Classification']]
    
    # Marker style: filled untuk significant, hollow untuk not significant
    for i, row in df_sorted.iterrows():
        marker_style = 'o' if row['Significant'] else 'o'
        facecolor = colors[i] if row['Significant'] else 'white'
        edgecolor = colors[i]
        
        ax.scatter(
            row['Sen_slope'], y_pos[i],
            s=150, marker=marker_style,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2,
            zorder=3
        )
    
    # Reference line di slope = 0
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Y labels
    ax.set_yticks(y_pos)
    labels_with_topic = [f"T{row['Topic']}: {row['Label']}" for _, row in df_sorted.iterrows()]
    ax.set_yticklabels(labels_with_topic)
    
    # X label tergantung series_type
    if series_type == 'proportion':
        ax.set_xlabel("Sen's slope (perubahan proporsi per minggu)")
    else:
        ax.set_xlabel("Sen's slope (perubahan frekuensi per minggu)")
    
    ax.set_title(
        f"Mann-Kendall Trend Analysis ({app_name.upper()}) — {series_type.capitalize()}\n"
        f"α = {alpha}, Hamed-Rao modification, n_weeks = 52",
        fontsize=11
    )
    
    # Legend manual
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=10, label='Emerging (significant ↑)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=10, label='Declining (significant ↓)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#7f7f7f', markeredgewidth=2,
               markersize=10, label='Not significant (stable)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save:
        filename = f"10_mk_forest_{series_type}.png"
        filepath = fig_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    
    return fig


def plot_significant_trends_lines(
    series_df,
    mk_results,
    topic_labels,
    fig_dir,
    app_name,
    series_type='proportion',
    figsize=(14, 8),
    save=True
):
    """
    Lineplot khusus topic yang significant trend-nya.
    Overlay dengan Sen's slope line untuk visual confirmation.
    
    Layout:
    - Small multiples (subplot grid)
    - 1 panel per topic significant
    - Setiap panel: data minggu (line) + Sen's slope line (dashed)
    
    Parameters
    ----------
    series_df : pd.DataFrame
        Output compute_weekly_series_per_topic().
    mk_results : pd.DataFrame
        Output apply_mann_kendall_per_topic().
    topic_labels : dict
    fig_dir : Path
    app_name : str
    series_type : str
    figsize : tuple
    save : bool
    
    Returns
    -------
    fig : matplotlib Figure atau None kalau tidak ada significant topic.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter ke topic significant
    sig_results = mk_results[mk_results['Significant']].copy()
    
    if len(sig_results) == 0:
        print(f"⚠ Tidak ada topic dengan significant trend ({series_type}). Skip plot.")
        return None
    
    n_sig = len(sig_results)
    # Layout: max 3 kolom
    ncols = min(3, n_sig)
    nrows = int(np.ceil(n_sig / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    weeks = series_df.index.values  # 1..52
    
    for ax_idx, (_, row) in enumerate(sig_results.iterrows()):
        ax = axes_flat[ax_idx]
        topic_id = row['Topic']
        slope = row['Sen_slope']
        intercept = row['Sen_intercept']
        p_val = row['MK_p_value']
        classification = row['Classification']
        
        # Plot data
        series = series_df[topic_id].values
        color = '#2ca02c' if classification == 'emerging' else '#d62728'
        ax.plot(weeks, series, marker='o', markersize=3, linewidth=1.2,
                color=color, alpha=0.7, label='Observed')
        
        # Plot Sen's slope line: y = intercept + slope * (x - 1)
        # Note: pymannkendall return intercept di x=0, jadi adjust untuk x=1..52
        sen_line = intercept + slope * np.arange(len(weeks))
        ax.plot(weeks, sen_line, linestyle='--', color='black', linewidth=1.5,
                label=f"Sen's slope = {slope:.4f}")
        
        # Title
        label_short = topic_labels.get(topic_id, f'Topic {topic_id}')
        ax.set_title(
            f"T{topic_id}: {label_short}\n"
            f"{classification.capitalize()} (p = {p_val:.4f})",
            fontsize=9
        )
        ax.set_xlabel('Minggu pasca-launch', fontsize=8)
        ax.set_ylabel(series_type.capitalize(), fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for ax_idx in range(n_sig, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)
    
    fig.suptitle(
        f"Significant Trends — {app_name.upper()} ({series_type.capitalize()})",
        fontsize=12, y=1.00
    )
    plt.tight_layout()
    
    if save:
        filename = f"10_mk_significant_lines_{series_type}.png"
        filepath = fig_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    
    return fig
