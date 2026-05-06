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
# Initially empty — to be populated in Phase B based on Phase A
# topic representation analysis (e.g., generic words like "aplikasi",
# "bank" yang muncul di hampir semua topik dan gak diskriminatif).
CUSTOM_DOMAIN_STOPWORDS = []


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
    interpretability (e.g., 'aplikasi', 'bank', 'bni', 'bsi').

    Returns
    -------
    list of str
        Sastrawi stopwords + CUSTOM_DOMAIN_STOPWORDS, deduplicated.
    """
    base = get_sastrawi_stopwords()
    # Use set union to dedupe in case custom words overlap with Sastrawi
    return list(set(base) | set(CUSTOM_DOMAIN_STOPWORDS))
