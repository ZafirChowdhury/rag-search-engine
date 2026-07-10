from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_seq_length = 256

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
           raise ValueError("cannot generate embeding for a empty string")

        embedding = self.model.encode([text])
        return embedding[0]
