"""
Stopword utilities for c-TF-IDF vectorizer in BERTopic pipeline.

Phase A: Sastrawi standard (~123 unique words after dedup).
Phase B: Sastrawi + custom domain extension (populated after empirical
analysis of Phase A topic representations).

Note: Stopwords are applied at c-TF-IDF stage only, NOT during text
preprocessing. This follows BERTopic best practice — embedding model
(IndoBERT) benefits from full sentence context including stopwords.

Sastrawi quirk: the raw stopword list contains 3 duplicate entries
('jika', 'sementara', 'juga' each appear twice). We dedupe via set
conversion since duplicates have no effect on c-TF-IDF vocabulary
filtering (set membership semantics).
"""
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Custom domain stopwords for Indonesian banking superapp reviews.
# Populated in Phase B (round 1) based on empirical analysis of Phase A
# topic representations (see notebook 05_bertopic_wondr.ipynb Section 6).
#
# Tier 1: Brand & generic app terms (high-frequency, non-discriminative
# across topics — e.g., "wondr" appears in T7, T9, T12 and many others).
# Tier 2: Discourse markers, particles, and filler words common in
# Indonesian informal review text (e.g., "kalau", "deh", "sih").
CUSTOM_DOMAIN_STOPWORDS = [
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
]


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


def get_extended_stopwords():
    """
    Return Sastrawi stopwords extended with custom domain words.

    Used in Phase B after empirical analysis of Phase A topics to
    remove generic high-frequency words that reduce topic
    interpretability (e.g., 'aplikasi', 'wondr', 'bni').

    Returns
    -------
    list of str
        Sastrawi stopwords + CUSTOM_DOMAIN_STOPWORDS, deduplicated.
    """
    base = get_sastrawi_stopwords()
    # Use set union to dedupe in case custom words overlap with Sastrawi
    return list(set(base) | set(CUSTOM_DOMAIN_STOPWORDS))
