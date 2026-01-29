import re
from typing import List, Iterable

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","from","into",
    "we","you","they","he","she","i","me","my","our","your","their","them"
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    out = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out

def simple_stem(token: str) -> str:
    """
    A very small, dependency-free stemmer (suffix stripping).
    It is not Porter Stemmer, but helps show "partial query" behaviour.
    """
    for suf in ("ingly","edly","ing","ed","ies","es","s","ly"):
        if token.endswith(suf) and len(token) > len(suf) + 2:
            if suf == "ies":
                return token[:-3] + "y"
            return token[:-len(suf)]
    return token

def preprocess(text: str, use_stemming: bool = False) -> List[str]:
    terms = normalize_tokens(tokenize(text))
    if use_stemming:
        terms = [simple_stem(t) for t in terms]
    return terms
