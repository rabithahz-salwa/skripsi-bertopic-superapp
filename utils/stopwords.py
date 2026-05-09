"""
Stopword utilities for c-TF-IDF vectorizer in BERTopic pipeline.

Phase A: Sastrawi standard (~123 unique words after dedup).
Phase B: Sastrawi + custom domain extension per app (populated after
empirical analysis of Phase A topic representations).

Note: Stopwords are applied at c-TF-IDF stage only, NOT during text
preprocessing. This follows BERTopic best practice — embedding model
(IndoBERT) benefits from full sentence context including stopwords.

Sastrawi quirk: the raw stopword list contains 3 duplicate entries
('jika', 'sementara', 'juga' each appear twice). We dedupe via set
conversion since duplicates have no effect on c-TF-IDF vocabulary
filtering (set membership semantics).

App-specific custom stopwords:
- Wondr (BNI): populated in v1 of Phase B (notebook 05).
- BYOND (BSI): to be populated after Phase A inspection notebook 06.
"""
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Custom domain stopwords per app for Indonesian banking superapp reviews.
# Each app has its own list to avoid cross-contamination — vocabulary
# specific to BNI shouldn't be filtered when analyzing BSI reviews.
#
# Wondr Tier 1: Brand & generic app terms (high-frequency, non-discriminative
# across topics — e.g., "wondr" appears in T7, T9, T12 and many others).
# Wondr Tier 2: Discourse markers, particles, and filler words common in
# Indonesian informal review text (e.g., "kalau", "deh", "sih").
CUSTOM_DOMAIN_STOPWORDS = {
    'wondr': [
        # Tier 1: Brand & generic app terms
        "wondr",      # brand name app
        "bni",        # brand name bank
        "aplikasi",   # generic app term, dominant in T7/T9/T12
        "nya",        # suffix particle ("aplikasi nya", "wondr nya")
        "kok",        # discourse marker (T9, T12)
        # Tier 2: Discourse markers & filler words
        "kalau",      # conditional marker (T3, T11)
        "kalo",       # informal variant of "kalau"
        "deh",        # particle (T11)
        "dulu",       # temporal marker (T11)
        "mending",    # comparison marker (T5, T11)
        "banget",     # intensifier (T0, T2)
        "sih",        # particle
        "ya",         # particle
    ],
    'byond': [
        # Empty placeholder. Populate after Phase A inspection di notebook 06
        # Section 6.2-6.3. Likely candidates: 'bsi', 'byond', 'aplikasi',
        # plus discourse markers (probably similar to wondr Tier 2).
    ],
}


def get_sastrawi_stopwords():
    """
    Return Sastrawi standard Indonesian stopword list (deduplicated).

    Sastrawi's raw list contains 3 duplicate entries ('jika', 'sementara',
    'juga'); we dedupe via set conversion. This has no effect on c-TF-IDF
    behavior (set membership semantics) but keeps the API consistent.

    Returns
    -------
    list of str
        Sastrawi default stopwords, deduplicated (~123 unique words).

    Examples
    --------
    >>> stopwords = get_sastrawi_stopwords()
    >>> len(stopwords) > 100
    True
    >>> 'yang' in stopwords
    True
    """
    factory = StopWordRemoverFactory()
    return list(set(factory.get_stop_words()))


def get_extended_stopwords(app_name='wondr'):
    """
    Return Sastrawi stopwords extended with app-specific custom domain words.

    Used in Phase B after empirical analysis of Phase A topics to remove
    generic high-frequency words that reduce topic interpretability
    (e.g., 'aplikasi', 'wondr', 'bni' for wondr; 'bsi', 'byond' for BYOND).

    Parameters
    ----------
    app_name : str, default 'wondr'
        Which app's custom stopwords to include. Must be a key in
        CUSTOM_DOMAIN_STOPWORDS dict ('wondr' or 'byond'). Default
        'wondr' for backwards compatibility with existing notebook 05.

    Returns
    -------
    list of str
        Sastrawi stopwords + CUSTOM_DOMAIN_STOPWORDS[app_name], deduplicated.

    Raises
    ------
    KeyError
        If app_name is not a recognized app key.

    Examples
    --------
    >>> wondr_stopwords = get_extended_stopwords('wondr')
    >>> 'wondr' in wondr_stopwords
    True
    >>> byond_stopwords = get_extended_stopwords('byond')
    >>> 'wondr' in byond_stopwords  # wondr-specific not in BYOND
    False
    """
    if app_name not in CUSTOM_DOMAIN_STOPWORDS:
        raise KeyError(
            f"Unknown app_name '{app_name}'. "
            f"Valid options: {list(CUSTOM_DOMAIN_STOPWORDS.keys())}"
        )
    base = get_sastrawi_stopwords()
    custom = CUSTOM_DOMAIN_STOPWORDS[app_name]
    # Use set union to dedupe in case custom words overlap with Sastrawi
    return list(set(base) | set(custom))
