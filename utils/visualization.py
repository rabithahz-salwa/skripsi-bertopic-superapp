"""
Visualization utilities for BERTopic DTM analysis.

Provides plotting functions for Dynamic Topic Modeling outputs:
- Small multiples grid (one panel per topic)
- Proportion plot (normalized for base rate effects)
- Proportion plot highlighted (presentation-ready, focused topics + annotations)

Plus shared topic label constants for wondr and BYOND.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# Topic labels (centralized — single source of truth)
# ============================================================

TOPIC_LABELS_WONDR = {
    0:  "Mobile Banking comparison",
    1:  "Verifikasi wajah gagal",
    2:  "Saldo terpotong / transaksi gagal",
    3:  "Gangguan & maintenance",
    4:  "Tarik tunai tanpa kartu",
    5:  "Aplikasi gak bisa dibuka",
    6:  "OTP / kode verifikasi email",
    7:  "Error 'kendala tim perbaiki'",
    8:  "Password / PIN salah",
    9:  "Sesi berakhir saat login",
    10: "Premature launch complaint",
    11: "Permintaan perbaikan bug",
    12: "Limit top-up wallet",
}

# Placeholder — akan diisi setelah BERTopic fit di notebook 06
TOPIC_LABELS_BYOND = {}


# ============================================================
# Data transformation
# ============================================================

def compute_proportion(tot_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-month proportion for each topic.

    Normalizes Frequency by total in-cluster docs per Timestamp,
    handling base-rate effects (e.g., bulan 5 spike in wondr).

    Parameters
    ----------
    tot_clean : pd.DataFrame
        Filtered topics_over_time (outlier -1 removed). Required columns:
        Topic, Timestamp, Frequency, Words.

    Returns
    -------
    pd.DataFrame
        Same as tot_clean plus columns: TotalFreq, Proportion.
        Proportion sums to 1.0 per Timestamp.
    """
    total_per_month = (
        tot_clean.groupby('Timestamp')['Frequency']
        .sum()
        .reset_index()
        .rename(columns={'Frequency': 'TotalFreq'})
    )
    out = tot_clean.merge(total_per_month, on='Timestamp')
    out['Proportion'] = out['Frequency'] / out['TotalFreq']
    return out


# ============================================================
# Plot: Small multiples grid
# ============================================================

def plot_dtm_smallmultiples(
    tot_clean: pd.DataFrame,
    topic_labels: dict,
    fig_dir: Path,
    app_name: str,
    n_cols: int = 4,
    figsize_per_panel: tuple = (3.5, 2.5),
    dpi: int = 300,
    filename: str = "08_dtm_smallmultiples.png",
) -> Path:
    """
    Plot one frequency-over-time panel per topic in a grid.

    Parameters
    ----------
    tot_clean : pd.DataFrame
        Filtered topics_over_time (outlier -1 removed).
    topic_labels : dict
        Mapping {topic_id: label_string}.
    fig_dir : Path
        Directory to save figure (created if missing).
    app_name : str
        For figure suptitle (e.g., "wondr by BNI").
    n_cols : int
        Grid columns (default 4 -> 4x4 for 13-16 topics).
    figsize_per_panel : tuple
        (width, height) per panel in inches.
    dpi : int
        Save DPI (300 = publication-grade).
    filename : str
        Output filename in fig_dir.

    Returns
    -------
    Path to saved figure.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    topic_ids = sorted(topic_labels.keys())
    n_topics = len(topic_ids)
    n_rows = int(np.ceil(n_topics / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
    )
    axes_flat = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, max(n_topics, 20)))

    for idx, topic_id in enumerate(topic_ids):
        ax = axes_flat[idx]
        topic_data = (
            tot_clean[tot_clean['Topic'] == topic_id]
            .sort_values('Timestamp')
        )
        total_n = int(topic_data['Frequency'].sum())
        label = topic_labels.get(topic_id, f"Topic {topic_id}")

        ax.plot(
            topic_data['Timestamp'],
            topic_data['Frequency'],
            marker='o',
            markersize=4,
            linewidth=1.5,
            color=colors[idx],
        )
        ax.set_title(f"T{topic_id}: {label}\n(N={total_n})", fontsize=9)
        ax.set_xticks(range(1, 13))
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(alpha=0.3)

    # Hide unused panels
    for j in range(n_topics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Topic Frequency Evolution Across 12 Relative Months — {app_name}",
        fontsize=12,
        fontweight='bold',
        y=1.00,
    )
    fig.supxlabel("Relative Month (post-launch)", fontsize=10)
    fig.supylabel("Frequency", fontsize=10)
    fig.tight_layout()

    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ============================================================
# Plot: Proportion (normalized)
# ============================================================

def plot_dtm_proportion(
    tot_clean: pd.DataFrame,
    topic_labels: dict,
    fig_dir: Path,
    app_name: str,
    figsize: tuple = (12, 6),
    dpi: int = 300,
    filename: str = "08_dtm_proportion.png",
) -> Path:
    """
    Plot topic proportions over time on a single panel.

    Each line = one topic. Y-axis = proportion of in-cluster docs that
    month assigned to that topic (sums to 1.0 per month).

    Parameters
    ----------
    tot_clean : pd.DataFrame
        Filtered topics_over_time (outlier -1 removed).
    topic_labels : dict
        Mapping {topic_id: label_string}.
    fig_dir : Path
        Directory to save figure (created if missing).
    app_name : str
        For figure title.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Save DPI.
    filename : str
        Output filename in fig_dir.

    Returns
    -------
    Path to saved figure.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tot_proportion = compute_proportion(tot_clean)

    topic_ids = sorted(topic_labels.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(topic_ids), 20)))

    fig, ax = plt.subplots(figsize=figsize)

    for idx, topic_id in enumerate(topic_ids):
        topic_data = (
            tot_proportion[tot_proportion['Topic'] == topic_id]
            .sort_values('Timestamp')
        )
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        ax.plot(
            topic_data['Timestamp'],
            topic_data['Proportion'],
            marker='o',
            markersize=4,
            linewidth=1.5,
            color=colors[idx],
            label=f"T{topic_id}: {label}",
        )

    ax.set_title(
        f"Topic Proportion Evolution — {app_name}\n(Normalized by total in-cluster docs per month)",
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel("Relative Month", fontsize=10)
    ax.set_ylabel("Topic Proportion (% of in-cluster docs that month)", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.3)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ============================================================
# Plot: Proportion highlighted (presentation-ready)
# ============================================================

def plot_dtm_proportion_highlighted(
    tot_clean: pd.DataFrame,
    topic_labels: dict,
    fig_dir: Path,
    app_name: str,
    highlight_topics: list,
    annotations: list = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
    filename: str = "08_dtm_proportion_highlighted.png",
) -> Path:
    """
    Presentation-ready proportion plot with focused topic highlighting + annotations.

    Selected topics rendered with full color + bold lines; non-selected topics
    rendered as gray background context. Optional annotations (vertical lines,
    arrow callouts) overlay key narrative events.

    Parameters
    ----------
    tot_clean : pd.DataFrame
        Filtered topics_over_time (outlier -1 removed).
    topic_labels : dict
        Mapping {topic_id: label_string}.
    fig_dir : Path
        Directory to save figure (created if missing).
    app_name : str
        For figure title.
    highlight_topics : list[int]
        Topic IDs to highlight with full color + bold line.
    annotations : list[dict], optional
        Annotation specs. Supported types:
        - {'type': 'vline', 'x': float, 'text': str, 'text_y': float (0-1, axes coord)}
        - {'type': 'arrow', 'text': str, 'xytext': (x, y), 'xy': (x, y)}
        vline x is data coord; arrow xy/xytext are data coords.
    figsize : tuple
    dpi : int
    filename : str

    Returns
    -------
    Path to saved figure.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tot_proportion = compute_proportion(tot_clean)

    topic_ids = sorted(topic_labels.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(topic_ids), 20)))

    fig, ax = plt.subplots(figsize=figsize)

    # ---- Pass 1: render non-highlighted topics as gray background ----
    for idx, topic_id in enumerate(topic_ids):
        if topic_id in highlight_topics:
            continue
        topic_data = (
            tot_proportion[tot_proportion['Topic'] == topic_id]
            .sort_values('Timestamp')
        )
        ax.plot(
            topic_data['Timestamp'],
            topic_data['Proportion'],
            marker='o',
            markersize=3,
            linewidth=1.0,
            color='gray',
            alpha=0.25,
            zorder=1,  # behind highlighted lines
        )

    # ---- Pass 2: render highlighted topics with full color + bold ----
    for idx, topic_id in enumerate(topic_ids):
        if topic_id not in highlight_topics:
            continue
        topic_data = (
            tot_proportion[tot_proportion['Topic'] == topic_id]
            .sort_values('Timestamp')
        )
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        ax.plot(
            topic_data['Timestamp'],
            topic_data['Proportion'],
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=colors[idx],
            label=f"T{topic_id}: {label}",
            zorder=3,  # in front
        )

    # ---- Annotations (optional) ----
    if annotations:
        for ann in annotations:
            if ann['type'] == 'vline':
                ax.axvline(
                    x=ann['x'],
                    color='red',
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=2,
                )
                if 'text' in ann:
                    ax.text(
                        ann['x'] + 0.1,
                        ann.get('text_y', 0.95),
                        ann['text'],
                        transform=ax.get_xaxis_transform(),
                        fontsize=9,
                        color='red',
                        fontweight='bold',
                        verticalalignment='top',
                    )
            elif ann['type'] == 'arrow':
                ax.annotate(
                    ann['text'],
                    xy=ann['xy'],
                    xytext=ann['xytext'],
                    fontsize=9,
                    fontweight='bold',
                    arrowprops=dict(
                        arrowstyle='->',
                        color='black',
                        lw=1.2,
                    ),
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.85,
                    ),
                )

    ax.set_title(
        f"Topic Proportion Evolution — {app_name}\n(Highlighted: key narrative topics)",
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel("Relative Month", fontsize=10)
    ax.set_ylabel("Topic Proportion (% of in-cluster docs that month)", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.3)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        frameon=False,
        title='Highlighted topics',
        title_fontsize=9,
    )

    fig.tight_layout()
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path
