# filepath: /home/gowtham/COde-recet/main.py
from utilis import get_markdown_chunks
from db.chroma import index_pdfs

if __name__ == "__main__":
    index_pdfs()
    print("Markdown chunks extracted successfully.")