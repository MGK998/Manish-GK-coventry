import math
from typing import Dict, List

def compute_idf(index: Dict[str, Dict[str, int]], n_docs: int) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    for term, postings in index.items():
        df = len(postings)
        idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf

def bm25_score(
    query_terms: List[str],
    index: Dict[str, Dict[str, int]],
    doc_lengths: Dict[str, int],
    idf: Dict[str, float],
    k1: float = 1.2,
    b: float = 0.75
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not doc_lengths:
        return scores
    avgdl = sum(doc_lengths.values()) / float(len(doc_lengths))
    if avgdl <= 0:
        return scores

    for term in query_terms:
        postings = index.get(term)
        if not postings:
            continue
        term_idf = idf.get(term, 0.0)
        for doc_id, tf in postings.items():
            dl = doc_lengths.get(doc_id, 0)
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            s = term_idf * (tf * (k1 + 1)) / (denom if denom else 1.0)
            scores[doc_id] = scores.get(doc_id, 0.0) + s
    return scores
