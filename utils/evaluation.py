"""
Coherence evaluation wrappers for BERTopic pipeline.

Provides functions to compute topic coherence (c_v primary, c_npmi secondary)
using gensim's CoherenceModel. Used for model selection in sensitivity
analysis grid search.

Methodology reference: bertopic_session_handoff_v2.md
- Primary metric: c_v (Roder et al., 2015)
- Secondary metric: c_npmi (validation of c_v winner)
"""

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def _extract_topic_words(topic_model, top_n: int = 10) -> list[list[str]]:
    """
    Extract top-N words per topic from a fitted BERTopic model.

    Excludes the noise topic (-1) since it does not represent a coherent topic.

    Parameters
    ----------
    topic_model : BERTopic
        A fitted BERTopic instance.
    top_n : int, default=10
        Number of top words to extract per topic.

    Returns
    -------
    list[list[str]]
        List of topic word lists, e.g., [['transfer', 'gagal', ...], [...]].
    """
    topic_words = []
    for topic_id in topic_model.get_topics():
        if topic_id == -1:  # skip noise
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]
        topic_words.append(words)
    return topic_words


def _tokenize_docs(docs: list[str]) -> list[list[str]]:
    """
    Simple whitespace tokenization for coherence computation.

    Note: gensim's CoherenceModel needs tokenized docs to build a co-occurrence
    reference. We use whitespace split since our preprocessing already produced
    clean, lowercased text.

    Parameters
    ----------
    docs : list[str]
        List of preprocessed document strings.

    Returns
    -------
    list[list[str]]
        List of token lists.
    """
    return [doc.split() for doc in docs]


def compute_coherence(
    topic_model,
    docs: list[str],
    coherence: str = "c_v",
    top_n: int = 10,
) -> float:
    """
    Compute topic coherence for a fitted BERTopic model.

    Parameters
    ----------
    topic_model : BERTopic
        A fitted BERTopic instance.
    docs : list[str]
        Preprocessed documents used to fit the BERTopic model. Used here to
        build the gensim co-occurrence reference corpus.
    coherence : str, default='c_v'
        Coherence measure. Common choices: 'c_v', 'c_npmi', 'u_mass'.
    top_n : int, default=10
        Number of top words per topic to consider in coherence calculation.

    Returns
    -------
    float
        Coherence score. Higher is better for c_v and c_npmi.

    Examples
    --------
    >>> from bertopic import BERTopic
    >>> topic_model = BERTopic().fit(docs)
    >>> score = compute_coherence(topic_model, docs, coherence='c_v')
    """
    topic_words = _extract_topic_words(topic_model, top_n=top_n)
    tokenized_docs = _tokenize_docs(docs)
    dictionary = Dictionary(tokenized_docs)

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=coherence,
    )
    return coherence_model.get_coherence()


def compute_coherence_both(
    topic_model,
    docs: list[str],
    top_n: int = 10,
) -> dict:
    """
    Compute both c_v and c_npmi coherence in one call.

    Convenience function for the validation step where we check whether
    c_v winner agrees with c_npmi winner.

    Parameters
    ----------
    topic_model : BERTopic
        A fitted BERTopic instance.
    docs : list[str]
        Preprocessed documents used to fit the BERTopic model.
    top_n : int, default=10
        Number of top words per topic to consider.

    Returns
    -------
    dict
        Dictionary with keys 'c_v' and 'c_npmi' mapping to float scores.

    Examples
    --------
    >>> scores = compute_coherence_both(topic_model, docs)
    >>> print(scores)
    {'c_v': 0.62, 'c_npmi': 0.18}
    """
    # Build dictionary once, reuse for both metrics (efficiency)
    topic_words = _extract_topic_words(topic_model, top_n=top_n)
    tokenized_docs = _tokenize_docs(docs)
    dictionary = Dictionary(tokenized_docs)

    scores = {}
    for measure in ["c_v", "c_npmi"]:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence=measure,
        )
        scores[measure] = coherence_model.get_coherence()

    return scores
