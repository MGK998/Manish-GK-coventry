import hashlib
from typing import Dict, List, Tuple
from .preprocess import preprocess
from .bm25 import compute_idf
from .storage import save_json

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def build_documents(publications: List[Dict]) -> Dict[str, Dict]:
    docs: Dict[str, Dict] = {}
    for p in publications:
        url = p.get("publication_url") or ""
        if not url:
            continue
        doc_id = stable_id(url)
        docs[doc_id] = {
            "id": doc_id,
            "title": p.get("title", ""),
            "year": p.get("year", ""),
            "authors": p.get("authors", []),
            "author_links": p.get("author_links", []),
            "publication_url": url,

            "abstract": p.get("abstract", ""),
            "organisations": p.get("organisations", []),
        }
    return docs

def build_inverted_index(docs: Dict[str, Dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    index: Dict[str, Dict[str, int]] = {}
    doc_lengths: Dict[str, int] = {}

    for doc_id, d in docs.items():
        text = " ".join([
            d.get("title",""),
            d.get("abstract",""),
            " ".join(d.get("authors", [])),
            str(d.get("year","")),
        ])
        terms = preprocess(text)
        doc_lengths[doc_id] = len(terms)

        tf: Dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1

        for term, freq in tf.items():
            index.setdefault(term, {})[doc_id] = freq

    return index, doc_lengths

def save_index(index_path: str, docs: Dict[str, Dict], index: Dict[str, Dict[str, int]], doc_lengths: Dict[str, int]) -> None:
    idf = compute_idf(index, n_docs=len(docs))
    payload = {
        "docs": docs,
        "index": index,
        "doc_lengths": doc_lengths,
        "idf": idf,
    }
    save_json(index_path, payload)
