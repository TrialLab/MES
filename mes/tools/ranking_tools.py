"""
Ranking — shared sentence encoder and hybrid ranking of evidence passages.

Passage score = weighted sum of the dense similarity (sentence encoder, cosine)
and the lexical similarity (TF-IDF, cosine), both min-max normalized within the
candidate set. Item-level priors (study design, recency, source rank) are added
by the sources. ``mmr_select`` removes near-duplicate items.
"""

import logging
import re
import threading

import numpy as np

from mes import config

logger = logging.getLogger(__name__)

_encoder = None
_encoder_lock = threading.Lock()
_encode_lock = threading.Lock()


def get_encoder():
    """Sentence encoder singleton (``MES_ENCODER``); loads from the local cache first."""
    global _encoder
    with _encoder_lock:
        if _encoder is None:
            from sentence_transformers import SentenceTransformer
            try:
                _encoder = SentenceTransformer(config.ENCODER, local_files_only=True)
            except Exception:
                _encoder = SentenceTransformer(config.ENCODER)
            logger.info("Loaded encoder %s on %s", config.ENCODER, _encoder.device)
    return _encoder


def encode(texts, is_query=False, batch_size=64):
    """L2-normalized embeddings (float32) of ``texts``."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = get_encoder()
    if is_query and hasattr(model, "encode_query"):
        fn = model.encode_query
    elif not is_query and hasattr(model, "encode_document"):
        fn = model.encode_document
    else:
        fn = model.encode
    with _encode_lock:  # one encode at a time; parallel sources share the model
        vectors = fn(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def lexical_scores(query, texts):
    """TF-IDF cosine similarity of ``query`` to each text (unigrams + bigrams, English stop words)."""
    if not texts:
        return np.zeros(0, dtype=np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        matrix = vectorizer.fit_transform(list(texts) + [query])
    except ValueError:  # empty vocabulary
        return np.zeros(len(texts), dtype=np.float32)
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    return scores.astype(np.float32)


def hybrid_scores(query, texts, dense_weight=0.65, query_vector=None, lexical_query=None):
    """
    Normalized hybrid relevance of each text to ``query`` (0..1) and the text embeddings.
    ``lexical_query`` (e.g. the question plus synonyms and drug codes) is used for the TF-IDF part.
    """
    if not texts:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    q = query_vector if query_vector is not None else encode([query], is_query=True)[0]
    vectors = encode(texts)
    dense = vectors @ q
    lexical = lexical_scores(lexical_query or query, texts)
    return dense_weight * _minmax(dense) + (1 - dense_weight) * _minmax(lexical), vectors


def expanded_query(plan):
    """Question plus every concept label and synonym (for lexical matching)."""
    terms = [t for e in plan.entities for t in [e.get("name", "")] + list(e.get("synonyms") or [])]
    return " ".join([plan.query] + [t for t in dict.fromkeys(terms) if t])


def mmr_select(vectors, scores, k, diversity=0.3):
    """Maximal marginal relevance: pick ``k`` indices balancing score and novelty."""
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    scores = _minmax(scores)
    selected, candidates = [], list(range(n))
    sims = vectors @ vectors.T if len(vectors) == n else np.zeros((n, n))
    while candidates and len(selected) < k:
        if not selected:
            best = max(candidates, key=lambda i: scores[i])
        else:
            best = max(candidates, key=lambda i: (1 - diversity) * scores[i] - diversity * max(sims[i, j] for j in selected))
        selected.append(best)
        candidates.remove(best)
    return selected


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])")


def chunk_text(text, size=900, overlap=150):
    """Split ``text`` into passages of about ``size`` characters at sentence boundaries."""
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if len(text) <= size:
        return [text] if text else []
    sentences = _SENTENCE_END.split(text)
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:  # very long sentence or table row block
            head, sentence = sentence[:size], sentence[size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1] + " " + sentence if overlap else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
