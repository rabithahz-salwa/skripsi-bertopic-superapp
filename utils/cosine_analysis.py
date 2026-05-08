"""
cosine_analysis.py
Cosine similarity analysis untuk validasi konsistensi temporal topic.

Section 9 dual-approach:
- D1: Embedding-level (semantic drift detection)
- D2: c-TF-IDF level (lexical drift detection)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# D1: EMBEDDING-LEVEL COSINE SIMILARITY
# =============================================================================

def compute_topic_centroids_per_month(df_clean, embeddings_clean, n_months=12):
    """
    Compute mean embedding (centroid) per (topic, month) pair.

    Setiap centroid merepresentasikan "semantic fingerprint" dari topic
    di bulan tertentu. Cosine similarity antar centroid antar bulan
    menunjukkan apakah substansi topic stabil sepanjang waktu.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Harus punya kolom: 'topic', 'relative_month'.
        Posisi baris harus align dengan baris di embeddings_clean.
    embeddings_clean : np.ndarray
        Shape (n_docs, embedding_dim). Index-aligned dengan df_clean.
    n_months : int, default 12
        Window analisis (12 bulan post-launch).

    Returns
    -------
    centroids : dict
        Nested dict: {topic_id: {month: np.ndarray (embedding_dim,) or None}}
        None jika tidak ada dokumen untuk kombinasi (topic, month) tsb.
    """
    # Reset index biar posisi baris di df sama dengan posisi di embeddings.
    # Ini krusial: kalau df_clean.index masih bekas filter sebelumnya
    # (misal 0, 5, 8, 12, ...), akses embeddings_clean[idx] akan salah.
    df_reset = df_clean.reset_index(drop=True)

    topics = sorted(df_reset['topic'].unique())
    months = list(range(1, n_months + 1))  # 1-indexed: bulan 1 s/d 12

    centroids = {}
    for topic in topics:
        centroids[topic] = {}
        for month in months:
            # Boolean mask untuk filter docs di (topic, month) ini
            mask = (df_reset['topic'] == topic) & (df_reset['relative_month'] == month)
            positions = df_reset[mask].index.to_numpy()

            if len(positions) == 0:
                # Tidak ada dokumen untuk kombinasi ini
                centroids[topic][month] = None
            else:
                # Mean pooling: rata-rata vektor sepanjang axis=0 (per-dimensi)
                # Hasilnya 1D array shape (embedding_dim,)
                centroids[topic][month] = embeddings_clean[positions].mean(axis=0)

    return centroids


def compute_cosine_sim_matrix_embedding(centroids, topic_id, n_months=12):
    """
    Compute matrix cosine similarity 12x12 antar bulan untuk satu topic.

    Vectorized approach: stack semua centroid jadi matrix (n_months, embedding_dim),
    lalu hitung pairwise similarity dalam 1 panggilan sklearn. Bulan tanpa
    centroid (None) diisi zero vector sebagai placeholder, lalu hasilnya
    di-mask jadi NaN supaya tidak menyesatkan saat plot/agregasi.

    Parameters
    ----------
    centroids : dict
        Output dari compute_topic_centroids_per_month().
        Struktur: {topic_id: {month: np.ndarray or None}}
    topic_id : int
        Topic yang ingin dianalisis.
    n_months : int, default 12

    Returns
    -------
    sim_matrix : pd.DataFrame
        Shape (n_months, n_months). Index & columns = bulan 1..n_months.
        Diagonal = 1.0 (kecuali bulan invalid → NaN).
        Simetris: sim_matrix.loc[i, j] == sim_matrix.loc[j, i].
    """
    months = list(range(1, n_months + 1))

    # Cari embedding_dim dari centroid pertama yang valid.
    # Pakai generator + next() biar gak perlu loop manual.
    # StopIteration kalau SEMUA bulan None — edge case tapi bagusnya di-handle.
    try:
        embedding_dim = next(
            v.shape[0] for v in centroids[topic_id].values() if v is not None
        )
    except StopIteration:
        # Topic ini gak ada data sama sekali di semua bulan — return all-NaN
        return pd.DataFrame(np.nan, index=months, columns=months)

    # Build stacked matrix (n_months, embedding_dim) + valid_mask
    stacked = np.zeros((n_months, embedding_dim))
    valid_mask = np.zeros(n_months, dtype=bool)

    for idx, month in enumerate(months):
        v = centroids[topic_id].get(month)
        if v is not None:
            stacked[idx] = v
            valid_mask[idx] = True
        # else: stacked[idx] tetap zero (placeholder), valid_mask[idx] tetap False

    # Vectorized: 1 call → matrix (n_months, n_months) of pairwise similarities
    sim_full = cosine_similarity(stacked)

    # Wrap ke DataFrame dengan label bulan yang manusiawi
    sim_matrix = pd.DataFrame(sim_full, index=months, columns=months)

    # Mask invalid rows & columns ke NaN.
    # ~valid_mask = negasi boolean ([T,F,T] → [F,T,F])
    # .loc[~valid_mask, :] = baris yang invalid, semua kolom
    # .loc[:, ~valid_mask] = semua baris, kolom yang invalid
    # Setelah dua operasi ini, NaN muncul di seluruh row/col untuk bulan invalid.
    sim_matrix.loc[~valid_mask, :] = np.nan
    sim_matrix.loc[:, ~valid_mask] = np.nan

    return sim_matrix


def compute_consecutive_sim_embedding(centroids, topic_id, n_months=12):
    """
    Compute consecutive cosine similarity: sim(t, t+1) untuk t = 1..n_months-1.

    Berguna untuk line plot yang menunjukkan kelancaran transisi antar bulan.
    Drop tajam di sim(t, t+1) tertentu mengindikasikan disruption / shift
    substantif di topic pada periode tersebut.

    Parameters
    ----------
    centroids : dict
        Output dari compute_topic_centroids_per_month().
    topic_id : int
    n_months : int, default 12

    Returns
    -------
    df_consecutive : pd.DataFrame
        Shape (n_months - 1, 3). Kolom: ['month_from', 'month_to', 'cosine_sim'].
        cosine_sim = NaN jika salah satu bulan tidak punya centroid.
    """
    records = []
    for t in range(1, n_months):
        v_from = centroids[topic_id].get(t)
        v_to = centroids[topic_id].get(t + 1)

        if v_from is not None and v_to is not None:
            # cosine_similarity expects 2D input; reshape (768,) → (1, 768)
            # Output shape (1, 1), ambil scalar dengan [0, 0]
            sim = cosine_similarity(
                v_from.reshape(1, -1),
                v_to.reshape(1, -1)
            )[0, 0]
        else:
            sim = np.nan

        records.append({
            'month_from': t,
            'month_to': t + 1,
            'cosine_sim': sim
        })

    return pd.DataFrame(records)


# =============================================================================
# D2: c-TF-IDF LEVEL COSINE SIMILARITY
# =============================================================================

def build_ctfidf_vectors_per_month(tot_raw, vocab=None):
    """
    Build binary word-presence vectors dari topics_over_time output BERTopic.

    Setiap (topic, month) di-encode sebagai vector binary di vocabulary space:
    cell = 1 kalau kata muncul di top-words BERTopic untuk (topic, month) tsb,
    0 kalau tidak. Cosine similarity antar vector menunjukkan overlap leksikal
    antara representasi topic di bulan-bulan berbeda.

    Approach: binary presence (sesuai keputusan Section 9 design).
    Tradeoff: kehilangan informasi rank/score, tapi defensible & simple.

    Parameters
    ----------
    tot_raw : pd.DataFrame
        Output BERTopic.topics_over_time(). Kolom wajib:
        - 'Topic' (int): topic ID, termasuk -1 (outlier)
        - 'Words' (str): comma-separated string top words, e.g. "a, b, c, ..."
        - 'Timestamp' (int): relative month (1-12)
        - 'Frequency' (int): jumlah dokumen [tidak digunakan di sini]
    vocab : list of str or None, default None
        Vocabulary global. Jika None, dibangun otomatis dari union semua words.
        Pass eksplisit kalau ingin share vocab antar app (wondr & BYOND).

    Returns
    -------
    vectors : dict
        Nested: {topic_id: {month: np.ndarray (vocab_size,) or None}}
        None jika tidak ada row tot_raw untuk (topic, month) tsb.
    vocab : list of str
        Sorted vocabulary. Berguna untuk inspect kata apa di dimensi mana,
        dan untuk re-use lintas app.
    """
    # Filter outlier topic -1 dulu — ini noise dari BERTopic, bukan topic real.
    df = tot_raw[tot_raw['Topic'] != -1].copy()

    # Parse 'Words' string jadi list. Pakai .strip() untuk buang whitespace
    # (kalau pakai split(',') aja, " mobile" punya space di depan).
    df['words_list'] = df['Words'].apply(
        lambda s: [w.strip() for w in s.split(',') if w.strip()]
    )

    # Build vocab kalau belum dikasih. Union semua kata, lalu sort.
    # Sort penting: vocab order menentukan dimensi vector. Kalau gak sorted,
    # run berikutnya bisa beda urutan (vocab dari set tidak deterministic),
    # dan cosine sim hasilnya sama (sim tidak bergantung urutan dimensi),
    # tapi inspect manual jadi confusing.
    if vocab is None:
        all_words = set()
        for words in df['words_list']:
            all_words.update(words)
        vocab = sorted(all_words)

    # Mapping word → index untuk lookup O(1) saat encode
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # Build {topic: {month: vector or None}}
    topics = sorted(df['Topic'].unique())
    months = sorted(df['Timestamp'].unique())

    vectors = {}
    for topic in topics:
        vectors[topic] = {}
        for month in months:
            row = df[(df['Topic'] == topic) & (df['Timestamp'] == month)]

            if len(row) == 0:
                # Tidak ada entry untuk kombinasi ini di tot_raw
                vectors[topic][month] = None
            else:
                # Ambil words_list dari row pertama (pasti unik per (topic, month))
                words = row.iloc[0]['words_list']
                vec = np.zeros(vocab_size)
                for w in words:
                    # Kata di luar vocab di-skip silently.
                    # Kasus ini hanya relevan kalau user pass custom vocab
                    # yang lebih kecil dari union actual.
                    if w in word_to_idx:
                        vec[word_to_idx[w]] = 1
                vectors[topic][month] = vec

    return vectors, vocab


def compute_cosine_sim_matrix_ctfidf(vectors, topic_id, n_months=12):
    """
    Compute matrix cosine similarity 12x12 antar bulan dari c-TF-IDF vectors.

    Vectorized approach: stack semua vector jadi matrix (n_months, vocab_size),
    hitung pairwise similarity dalam 1 panggilan sklearn. Bulan tanpa vector
    (None) ATAU dengan all-zero vector di-mask jadi NaN.

    Parameters
    ----------
    vectors : dict
        Output dari build_ctfidf_vectors_per_month().
        Struktur: {topic_id: {month: np.ndarray or None}}
    topic_id : int
    n_months : int, default 12

    Returns
    -------
    sim_matrix : pd.DataFrame
        Shape (n_months, n_months). Index & columns = bulan 1..n_months.
        Diagonal = 1.0 untuk bulan valid, NaN untuk bulan invalid.
    """
    months = list(range(1, n_months + 1))

    # Detect vocab_size dari vector pertama yang valid (non-None, non-zero).
    # Tambahan check `.sum() > 0` dibanding D1: di D2, vector bisa not-None
    # tapi all-zero kalau pass custom vocab yang tidak cover words actual.
    try:
        vocab_size = next(
            v.shape[0] for v in vectors[topic_id].values()
            if v is not None and v.sum() > 0
        )
    except StopIteration:
        # Topic ini gak ada data valid sama sekali
        return pd.DataFrame(np.nan, index=months, columns=months)

    # Build stacked matrix + valid_mask (sama pattern dengan function #2)
    stacked = np.zeros((n_months, vocab_size))
    valid_mask = np.zeros(n_months, dtype=bool)

    for idx, month in enumerate(months):
        v = vectors[topic_id].get(month)
        # Valid kalau not None DAN punya minimal 1 kata (sum > 0)
        if v is not None and v.sum() > 0:
            stacked[idx] = v
            valid_mask[idx] = True

    # Vectorized pairwise similarity
    sim_full = cosine_similarity(stacked)

    sim_matrix = pd.DataFrame(sim_full, index=months, columns=months)

    # Mask invalid rows & columns ke NaN
    sim_matrix.loc[~valid_mask, :] = np.nan
    sim_matrix.loc[:, ~valid_mask] = np.nan

    return sim_matrix


def compute_consecutive_sim_ctfidf(vectors, topic_id, n_months=12):
    """
    Compute consecutive cosine similarity dari c-TF-IDF vectors:
    sim(t, t+1) untuk t = 1..n_months-1.

    Output format identik dengan compute_consecutive_sim_embedding(),
    sehingga line plot D1 vs D2 bisa pakai code yang sama.

    Parameters
    ----------
    vectors : dict
        Output dari build_ctfidf_vectors_per_month().
    topic_id : int
    n_months : int, default 12

    Returns
    -------
    df_consecutive : pd.DataFrame
        Shape (n_months - 1, 3). Kolom: ['month_from', 'month_to', 'cosine_sim'].
        cosine_sim = NaN jika salah satu bulan tidak punya vector valid.
    """
    records = []
    for t in range(1, n_months):
        v_from = vectors[topic_id].get(t)
        v_to = vectors[topic_id].get(t + 1)

        # Valid kalau dua-duanya not None DAN punya minimal 1 kata
        both_valid = (
            v_from is not None and v_to is not None
            and v_from.sum() > 0 and v_to.sum() > 0
        )

        if both_valid:
            sim = cosine_similarity(
                v_from.reshape(1, -1),
                v_to.reshape(1, -1)
            )[0, 0]
        else:
            sim = np.nan

        records.append({
            'month_from': t,
            'month_to': t + 1,
            'cosine_sim': sim
        })

    return pd.DataFrame(records)

# =============================================================================
# PLOTTING (D1 & D2 — dual-purpose dengan flag `approach`)
# =============================================================================

def plot_cosine_heatmaps(
    sim_matrices,
    topic_labels,
    fig_dir,
    app_name,
    approach="embedding",
    cmap="viridis",
    figsize=(16, 14),
    save=True,
):
    """
    Plot small multiples 4x4 heatmap cosine similarity per topic (13 topic).

    Slot ke-14 sampai 16 di-hide karena cuma ada 13 topic. Color range
    di-share antar panel (auto-detect dari min/max semua sim_matrices)
    supaya komparasi visual antar topic valid. NaN cell auto-render
    sebagai cell kosong oleh seaborn (cell tanpa warna).

    Parameters
    ----------
    sim_matrices : dict
        {topic_id: pd.DataFrame (12x12)} — output dari
        compute_cosine_sim_matrix_embedding() atau _ctfidf() per topic.
    topic_labels : dict
        {topic_id: str} mapping ke human-readable label, e.g.
        {0: "MB comparison", 1: "Verifikasi wajah", ...}
    fig_dir : pathlib.Path
        Direktori output figure (e.g., outputs/figures/wondr/).
    app_name : str
        "wondr" atau "byond" — dipakai di filename + title.
    approach : str, default "embedding"
        "embedding" untuk D1, "ctfidf" untuk D2. Affecting filename + title.
    cmap : str, default "viridis"
        Matplotlib colormap name.
    figsize : tuple, default (16, 14)
        Figure size dalam inch.
    save : bool, default True
        Kalau True, simpan ke fig_dir / "09_cosine_{approach}_heatmap.png".

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object — caller bisa modify lebih lanjut atau plt.show() manual.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.ravel()

    # Auto-detect shared color range dari semua sim_matrices, abaikan NaN.
    # Pakai nanmin/nanmax biar NaN di-skip otomatis.
    all_values = np.concatenate([
        m.values.flatten() for m in sim_matrices.values()
    ])
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    # Plot per topic
    topics_sorted = sorted(sim_matrices.keys())
    for idx, topic_id in enumerate(topics_sorted):
        ax = axes_flat[idx]
        sim = sim_matrices[topic_id]

        # cbar=False per panel — colorbar shared di-add di akhir
        sns.heatmap(
            sim,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
            linewidths=0.3,
            linecolor='white',
            xticklabels=True,
            yticklabels=True,
        )

        # Title: label topic. Truncate kalau terlalu panjang biar gak overflow.
        label = topic_labels.get(topic_id, f"T{topic_id}")
        if len(label) > 25:
            label = label[:22] + "..."
        ax.set_title(f"T{topic_id}: {label}", fontsize=10)

        ax.set_xlabel("Bulan", fontsize=8)
        ax.set_ylabel("Bulan", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes (slot ke-14, 15, 16 dari grid 4x4)
    for idx in range(len(topics_sorted), n_rows * n_cols):
        axes_flat[idx].axis('off')

    # Shared colorbar di kanan, span seluruh figure tinggi.
    # Caranya: bikin scalar mappable terpisah karena tiap heatmap punya
    # mappable sendiri — pakai cmap + Normalize untuk shared bar.
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])  # required by older matplotlib versions
    cbar = fig.colorbar(sm, ax=axes_flat.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=10)

    # Suptitle
    approach_label = "Embedding-level (IndoBERT centroid)" if approach == "embedding" \
        else "c-TF-IDF level (top-words binary)"
    fig.suptitle(
        f"Cosine Similarity Antar Bulan per Topic — {app_name.upper()}\n{approach_label}",
        fontsize=13,
        y=0.995,
    )

    # Save
    if save:
        from pathlib import Path
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / f"09_cosine_{approach}_heatmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    return fig


def plot_cosine_lineplot_highlighted(
    consecutive_sims,
    topic_labels,
    fig_dir,
    app_name,
    highlight_topics,
    approach="embedding",
    figsize=(12, 6),
    save=True,
):
    """
    Plot consecutive cosine similarity sebagai line plot dengan
    sebagian topic di-highlight (sesuai pattern plot_dtm_proportion_highlighted).

    Topic non-highlighted di-render abu-abu transparan sebagai background;
    topic highlighted pakai warna distinct dengan legend. Plotting order:
    non-highlighted dulu (belakang), highlighted setelahnya (depan).

    Parameters
    ----------
    consecutive_sims : dict
        {topic_id: pd.DataFrame} dengan kolom
        ['month_from', 'month_to', 'cosine_sim'] — output dari
        compute_consecutive_sim_*() per topic.
    topic_labels : dict
        {topic_id: str} mapping label.
    fig_dir : pathlib.Path
    app_name : str
        "wondr" atau "byond".
    highlight_topics : list of int
        Topic IDs yang di-highlight, e.g. [0, 1, 2, 3, 6].
    approach : str, default "embedding"
        "embedding" atau "ctfidf".
    figsize : tuple, default (12, 6)
    save : bool, default True

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette untuk highlighted: ambil dari tab10 (qualitative, distinct)
    highlight_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(highlight_topics)]
    color_map = dict(zip(highlight_topics, highlight_colors))

    # === Step 1: Plot non-highlighted dulu (di belakang) ===
    for topic_id, df_consec in consecutive_sims.items():
        if topic_id in highlight_topics:
            continue
        ax.plot(
            df_consec['month_from'],
            df_consec['cosine_sim'],
            color='lightgray',
            alpha=0.5,
            linewidth=1.0,
            zorder=1,  # paling belakang
        )

    # === Step 2: Plot highlighted di atas (foreground) ===
    for topic_id in highlight_topics:
        if topic_id not in consecutive_sims:
            continue  # safeguard kalau highlight_topics include ID yang gak ada
        df_consec = consecutive_sims[topic_id]
        label = topic_labels.get(topic_id, f"T{topic_id}")
        ax.plot(
            df_consec['month_from'],
            df_consec['cosine_sim'],
            color=color_map[topic_id],
            linewidth=2.2,
            marker='o',
            markersize=5,
            label=f"T{topic_id}: {label}",
            zorder=2,  # depan
        )

    # X-axis: month_from values 1-11. Label sebagai "M1→M2", "M2→M3", dst.
    # Pakai value pertama dari salah satu df sebagai reference.
    sample_df = next(iter(consecutive_sims.values()))
    x_ticks = sample_df['month_from'].values
    x_labels = [f"M{int(r['month_from'])}→M{int(r['month_to'])}"
                for _, r in sample_df.iterrows()]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel("Transisi bulan", fontsize=11)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(
        loc='best',
        fontsize=9,
        framealpha=0.9,
        title="Highlighted topics",
    )

    # Suptitle
    approach_label = "Embedding-level (IndoBERT centroid)" if approach == "embedding" \
        else "c-TF-IDF level (top-words binary)"
    fig.suptitle(
        f"Consecutive Cosine Similarity — {app_name.upper()}\n{approach_label}",
        fontsize=12,
    )

    plt.tight_layout()

    # Save
    if save:
        from pathlib import Path
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / f"09_cosine_{approach}_lineplot.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    return fig
