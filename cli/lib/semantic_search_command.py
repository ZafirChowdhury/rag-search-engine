from .semantic_search import SemanticSearch
from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

import re

def semantic_chunk_command(text: str, max_chunk_size=4, overlap=0) -> None:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError("Chunk size must be bigger then overlap")

    pointer = 0
    while pointer < len(sentences):
        chunk = " ".join(sentences[pointer : pointer + max_chunk_size])
        chunks.append(chunk)

        pointer += step

    print(f"Semantically chunking {len(text)} characters")
    for i, line in enumerate(chunks, start=1):
        print(f"{i} {line}")

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def search_command(query: str, limit=DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)
    for i, result in enumerate(results, start=1):
        truncated = result["description"][:100] + "..." if len(result["description"]) > 100 else result["description"]
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(truncated)

def embed_query_text_command(query: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings_command() -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def verify_model_command() -> None:
    search_instance = SemanticSearch()

    print(f"Model loaded: {str(search_instance.model)}")
    print(f"Max sequence length: {search_instance.max_seq_length}")

def embed_text_command(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
