"""
Lexical diversity metrics.

Sources:
  - self_bleu, wdistinct_n: MAS_Diversity (Chen et al., ACL 2026)
    compute_vendi_extended.py
  - distinct_n, mean_bigram_repetition, bigram_repetition: shrooms-llm benchmark_metrics.py
  - lexical_jaccard_distance, verbosity_ratio, generation_fails_coherence_fast,
    distinct_1, distinct_2, distinct_*_for_evaluation: ported from shrooms-llm benchmark_metrics.py
"""

from __future__ import annotations
import re
import math
from collections import Counter


# Stopwords + boilerplate academic terms (from MAS_Diversity)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "it", "its", "we", "our", "i", "my", "you", "your", "they", "their",
    "as", "if", "not", "no",
}
_BOILERPLATE = {
    "propose", "hypothesize", "investigate", "study", "explore",
    "examine", "analyze", "present", "demonstrate",
}
_FILTER = _STOPWORDS | _BOILERPLATE


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _FILTER]


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Self-BLEU (MAS_Diversity)
# ---------------------------------------------------------------------------

def self_bleu(texts: list[str]) -> float:
    """
    Average sentence-BLEU of each text using all other texts as references.
    Lower = more lexically diverse.
    Falls back to unigram overlap if nltk is unavailable.
    """
    if len(texts) < 2:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
        smoother = SmoothingFunction().method1
        scores = []
        tokenized = [t.lower().split() for t in texts]
        for i, hyp in enumerate(tokenized):
            refs = [t for j, t in enumerate(tokenized) if j != i]
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smoother))
        return float(sum(scores) / len(scores))
    except ImportError:
        return _self_bleu_fallback(texts)


def _self_bleu_fallback(texts: list[str]) -> float:
    """Unigram overlap fallback when nltk is not installed."""
    scores = []
    bags = [set(t.lower().split()) for t in texts]
    for i, b in enumerate(bags):
        others = [bags[j] for j in range(len(bags)) if j != i]
        if not others:
            continue
        union = set().union(*others)
        if not union:
            continue
        scores.append(len(b & union) / len(b | union) if b | union else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# WDistinct-n (MAS_Diversity)
# ---------------------------------------------------------------------------

def wdistinct_n(texts: list[str], n: int = 3) -> float:
    """
    IDF-weighted n-gram distinctness across all texts.
    Higher = more diverse content vocabulary.

    IDF is computed from the batch itself (no external corpus needed).
    """
    if not texts:
        return 0.0

    tokenized = [_tokenize(t) for t in texts]
    all_ngrams = [_ngrams(toks, n) for toks in tokenized]

    # Build IDF from this batch
    N = len(texts)
    df: Counter = Counter()
    for doc_ngrams in all_ngrams:
        for ng in set(doc_ngrams):
            df[ng] += 1
    idf = {ng: math.log((1 + N) / (1 + count)) for ng, count in df.items()}

    scores = []
    for doc_ngrams in all_ngrams:
        if not doc_ngrams:
            continue
        unique = set(doc_ngrams)
        total_weight = sum(idf.get(ng, 0.0) for ng in doc_ngrams)
        unique_weight = sum(idf.get(ng, 0.0) for ng in unique)
        if total_weight > 0:
            scores.append(unique_weight / total_weight)

    return float(sum(scores) / len(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Distinct-n (shrooms-llm benchmark_metrics.py)
# ---------------------------------------------------------------------------

def distinct_n(texts: list[str], n: int) -> float:
    """
    Ratio of unique n-grams to total n-grams across all texts.
    Higher = more diverse. Same formulation as shrooms-llm distinct_1 / distinct_2.
    """
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(t.lower().split())

    total = [tuple(all_tokens[i : i + n]) for i in range(len(all_tokens) - n + 1)]
    if not total:
        return 0.0
    return len(set(total)) / len(total)


# ---------------------------------------------------------------------------
# Bigram repetition (shrooms-llm benchmark_metrics.py)
# ---------------------------------------------------------------------------

def bigram_repetition(text: str) -> float:
    """
    Within-text bigram repetition: 1 - |unique_bigrams| / |total_bigrams|.
    0 = perfectly novel, 1 = fully repetitive.
    """
    tokens = text.lower().split()
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def mean_bigram_repetition(texts: list[str]) -> float:
    """Mean bigram_repetition across a list of texts."""
    if not texts:
        return 0.0
    return sum(bigram_repetition(t) for t in texts) / len(texts)


# ---------------------------------------------------------------------------
# Single-text variants (ported from shrooms-llm benchmark_metrics.py)
# These operate on one string, matching the original shrooms-llm API.
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    import re as _re
    return _re.findall(r"\S+", text.lower())


def distinct_1(text: str) -> float:
    """``|unique words| / |words|`` for a single text. Empty string → 0."""
    words = _words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def distinct_2(text: str) -> float:
    """``|unique bigrams| / |bigrams|`` for a single text. Fewer than 2 words → 0."""
    words = _words(text)
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


# Alias matching the original shrooms-llm name
bigram_repetition_index = bigram_repetition


# ---------------------------------------------------------------------------
# Coherence guard (ported from shrooms-llm benchmark_metrics.py)
# ---------------------------------------------------------------------------

_CHAR_RUN_PATTERN = re.compile(r"(.)\1{4,}")  # 5+ identical consecutive chars


def generation_fails_coherence_fast(text: str) -> bool:
    """
    Fast heuristic for gibberish / semantic collapse — no LLM needed.
    Returns True if the text looks degenerate (character runs, extreme repetition,
    very low unigram entropy). Use to zero out diversity scores for bad generations.
    """
    s = text.strip()
    if not s:
        return True
    if _CHAR_RUN_PATTERN.search(s):
        return True
    words = _words(s)
    if len(words) < 2:
        return False
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    top = max(counts.values())
    if top / len(words) >= 0.72:
        return True
    if len(words) >= 12:
        u = len(set(words))
        if u / len(words) < 0.08:
            return True
        probs = [c / len(words) for c in counts.values()]
        h = -sum(p * math.log(p + 1e-12, 2) for p in probs)
        h_norm = h / math.log(len(counts) + 1e-12, 2) if len(counts) > 1 else 0.0
        if h_norm < 0.35:
            return True
    return False


def distinct_1_for_evaluation(text: str) -> float:
    """distinct_1, zeroed out when generation_fails_coherence_fast triggers."""
    if generation_fails_coherence_fast(text):
        return 0.0
    return distinct_1(text)


def distinct_2_for_evaluation(text: str) -> float:
    """distinct_2, zeroed out when generation_fails_coherence_fast triggers."""
    if generation_fails_coherence_fast(text):
        return 0.0
    return distinct_2(text)


# ---------------------------------------------------------------------------
# Comparison utilities (ported from shrooms-llm benchmark_metrics.py)
# ---------------------------------------------------------------------------

def lexical_jaccard_distance(a: str, b: str) -> float:
    """
    ``1 - |A∩B| / |A∪B|`` over whitespace token sets (lowercased).
    Higher = fewer shared words (surface novelty vs a reference).
    """
    sa, sb = set(_words(a)), set(_words(b))
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return 1.0 - (len(sa & sb) / union)


def verbosity_ratio(num_tokens_baseline: int, num_tokens_steered: int) -> float:
    """
    steered / baseline token count ratio.
    Returns nan if baseline is 0 and steered is non-zero; 1.0 if both are 0.
    """
    if num_tokens_baseline <= 0:
        return float("nan") if num_tokens_steered > 0 else 1.0
    return float(num_tokens_steered) / float(num_tokens_baseline)
