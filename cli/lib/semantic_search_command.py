from .semantic_search import SemanticSearch

def verify_model_command() -> None:
    ss = SemanticSearch()

    print(f"Model loaded: {str(ss.model)}")
    print(f"Max sequence length: {ss.max_seq_length}")

def embed_text_command(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
