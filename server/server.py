import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from db.chroma import index_pdfs  # adjust import as needed


app = FastAPI()


@app.get("/doc")
async def read_root():
    return {"message": "Welcome to the FastAPI server!"}


@app.post("/ingest")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Query(
        ..., description="Name of the collection to index the document into"
    ),
):
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename

    try:
        with temp_file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        index_pdfs(source=str(temp_file_path), collection_name=collection_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing file: {e}")

    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "collection_name": collection_name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)