import os
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
load_dotenv()

class OpenRouterBGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self,model: str = "baai/bge-m3"):
        self.api_key = os.getenv("OPENROUTER_API")
        self.model = model
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def __call__(self, texts: Documents) -> Embeddings:
        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API')}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": list(texts),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # each item in data["data"] is {"embedding": [...]}
        return [d["embedding"] for d in data["data"]]

# Example usage
