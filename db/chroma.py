# filepath: /home/gowtham/COde-recet/db/chroma.py
import chromadb
import os
import re
from utilis import get_markdown_chunks
from dotenv import load_dotenv
from embedding.bgme import OpenRouterBGEEmbeddingFunction
from pathlib import Path

load_dotenv()
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENET"),
    database=os.getenv("CHROMA_DB"),
)

def init_collection(name: str = "your_collection_name2"):
    ef = OpenRouterBGEEmbeddingFunction()
    print("Inside embedding")
    collection = client.get_or_create_collection(name=name,embedding_function=ef)
    return collection

def index_pdfs(source: str, collection_name: str):
    """Index a single PDF/markdown source into the given Chroma collection.

    - Ensures the collection is initialised before use
    - Safely handles empty collections (no existing IDs)
    - Appends new IDs after the current max numeric suffix
    """

    documents, images = get_markdown_chunks(source)

    collection = init_collection(name=collection_name)

    maxtext_id = 0
    maximg_id = 0

    try:
        results = collection.get(include=["ids"])  # type: ignore[arg-type]
        all_ids = results.get("ids", []) if isinstance(results, dict) else []
    except Exception as e:
        print(f"Error fetching existing IDs from collection '{collection_name}': {e}")
        all_ids = []

    if all_ids:
        doc_pattern = re.compile(r"doc_(\d+)")
        image_pattern = re.compile(r"image_(\d+)")
        text_numbers = []
        image_numbers = []

        for doc_id in all_ids:
            if not isinstance(doc_id, str):
                continue
            doc_match = doc_pattern.search(doc_id)
            if doc_match:
                text_numbers.append(int(doc_match.group(1)))
            image_match = image_pattern.search(doc_id)
            if image_match:
                image_numbers.append(int(image_match.group(1)))

        if text_numbers:
            maxtext_id = max(text_numbers)
        if image_numbers:
            maximg_id = max(image_numbers)

    ids = [f"doc_{i + maxtext_id + 1}" for i in range(len(documents))]

    if documents:
        try:
            collection.add(
                ids=ids,
                documents=documents,
            )
            print(
                f"Successfully added {len(documents)} documents to collection '{collection.name}'"
            )
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            return

    image_ids = [f"image_{i + maximg_id + 1}" for i in range(len(images))]
    if images:
        try:
            collection.add(
                ids=image_ids,
                documents=images,
            )
            print(
                f"Successfully added {len(images)} images to collection '{collection.name}'"
            )
        except Exception as e:
            print(f"Error adding images to collection: {e}")
            return
    
    images_folder = Path("images")
    if images_folder.exists() and images_folder.is_dir():
        for image_file in images_folder.iterdir():
            if image_file.is_file():
                try:
                    image_file.unlink()
                    print(f"Deleted image: {image_file}")
                except Exception as e:
                    print(f"Error deleting image {image_file}: {e}")
    
    print(
        f"Indexed {len(documents)} documents and {len(images)} images into collection '{collection.name}'"
    )

if __name__ == "__main__":
    index_pdfs()