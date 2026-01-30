from typing import Dict, List
from .preprocess import preprocess
from .bm25 import bm25_score

def search(query: str, payload: Dict, top_k: int = 10, use_stemming: bool = False, org_filter: str = None) -> List[Dict]:
    docs: Dict[str, Dict] = payload.get("docs", {})
    index: Dict[str, Dict[str, int]] = payload.get("index", {})
    doc_lengths: Dict[str, int] = payload.get("doc_lengths", {})
    idf: Dict[str, float] = payload.get("idf", {})

    q_terms = preprocess(query, use_stemming=use_stemming)
    scores = bm25_score(q_terms, index=index, doc_lengths=doc_lengths, idf=idf)

    # Filter by organisation if requested
    if org_filter:
        valid_doc_ids = {
            doc_id for doc_id, d in docs.items() 
            if org_filter in d.get("organisations", [])
        }
        scores = {doc_id: score for doc_id, score in scores.items() if doc_id in valid_doc_ids}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in ranked:
        d = docs.get(doc_id, {})
        results.append({"score": round(float(score), 4), **d})
    return results
