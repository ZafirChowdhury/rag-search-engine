from .semantic_search import SemanticSearch
from .search_utils import load_movies

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
