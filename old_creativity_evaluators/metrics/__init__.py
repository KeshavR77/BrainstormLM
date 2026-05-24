from .structural import vendi_score, order_parameter, pairwise_cosine_distance
from .lexical import (
    self_bleu,
    wdistinct_n,
    distinct_n,
    distinct_1,
    distinct_2,
    mean_bigram_repetition,
    bigram_repetition,
    bigram_repetition_index,
    generation_fails_coherence_fast,
    distinct_1_for_evaluation,
    distinct_2_for_evaluation,
    lexical_jaccard_distance,
    verbosity_ratio,
)
from .perplexity import perplexity_stats_from_nlls

__all__ = [
    # structural
    "vendi_score",
    "order_parameter",
    "pairwise_cosine_distance",
    # lexical — collection
    "self_bleu",
    "wdistinct_n",
    "distinct_n",
    "mean_bigram_repetition",
    # lexical — single text
    "distinct_1",
    "distinct_2",
    "bigram_repetition",
    "bigram_repetition_index",  # alias for bigram_repetition
    # coherence guard
    "generation_fails_coherence_fast",
    "distinct_1_for_evaluation",
    "distinct_2_for_evaluation",
    # comparison utilities
    "lexical_jaccard_distance",
    "verbosity_ratio",
    "perplexity_stats_from_nlls",
]
