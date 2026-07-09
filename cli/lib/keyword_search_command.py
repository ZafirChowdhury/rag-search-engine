import sys

from .keyword_search import InvertedIndex, tokenize_single_term, tokenize_text
from .search_utils import DEFAULT_SEARCH_LIMIT


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def load_idx_helper() -> InvertedIndex:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        sys.exit("cache not found!")
    return idx


def bm25_idf_command(term: str) -> str:
    idx = load_idx_helper()
    bm25idf = idx.get_bm25_idf(tokenize_single_term(term))
    return f"BM25 IDF score of '{term}': {bm25idf:.2f}"


def tfidf_command(doc_id: int, tfidf_term: str) -> str:
    idx = load_idx_helper()
    tf = idx.get_tf(doc_id, tfidf_term)
    idf = idx.get_idf(tfidf_term)
    tf_idf = tf * idf
    return f"TF-IDF score of '{tfidf_term}' in document '{doc_id}': {tf_idf:.2f}"


def get_idf_command(term: str) -> float:
    idx = load_idx_helper()
    term = tokenize_single_term(term)
    return idx.get_idf(term)


def get_tf_command(doc_id: str, term: str) -> int:
    idx = load_idx_helper()
    return idx.get_tf(doc_id, tokenize_single_term(term))


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = load_idx_helper()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for id in matching_doc_ids:
            if id in seen:
                continue
            seen.add(id)
            doc = idx.docmap[id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results
