import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llms.captioning import generate_image_caption  # new import

def markdown_chunk(markdown_str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(markdown_str)
    return texts

def load_pdf_doc(source: str):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    doc_result = converter.convert(source)
    document = doc_result.document

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    image_files = []
    picture_counter = 0

    # Iterate over items and save all PictureItem images
    for element, _level in document.iterate_items():
        if isinstance(element, PictureItem):
            picture_counter += 1
            image = element.get_image(document)  # returns PIL.Image
            image_path = images_dir / f"picture_{picture_counter}.png"
            image.save(image_path, format="PNG")
            image_files.append(str(image_path))

    return document, image_files

def get_markdown_chunks(source: str):
    """Extract markdown chunks, generate captions for images, and return them."""
    doc, image_files = load_pdf_doc(source)
    mark_down = doc.export_to_markdown()

    markdown_chunks = markdown_chunk(markdown_str=mark_down)
    image_captions = [generate_image_caption(image_path) for image_path in image_files]

    return markdown_chunks, image_captions