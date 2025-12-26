import os
import yaml
from tqdm.auto import tqdm
from index_creator import LocalIndex
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

from pypdf import PdfReader


class TextDocument:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


def _read_text_file(path: str) -> str:
    # Robust-ish text reading (handles utf-8 and common fallbacks)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            pass
    # last resort: ignore errors
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_pypdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


def load_file(filename: str):
    try:
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = _read_pdf_pypdf(filename)
        else:
            text = _read_text_file(filename)

        # keep your original behavior
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return []


def chunk_text_by_word_count(paragraphs, word_limit=500):
    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if current_word_count + paragraph_word_count > word_limit:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def get_content(root_dir):
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return []

    filenames = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if not name.startswith(".")
    ]

    split_pages_text = []
    for filename in filenames:
        if os.path.isdir(filename):
            continue

        print(f"Processing {filename}...")
        paragraphs = load_file(filename)
        chunks = chunk_text_by_word_count(paragraphs)

        for chunk in chunks:
            processed_text = (
                chunk.replace("\n", " ")
                     .replace("\r", " ")
                     .replace("\t", " ")
                     .strip()
            )
            if processed_text:
                split_pages_text.append(processed_text)

    return split_pages_text


def vectorize_and_upload_pdf_content(collection, textlist, embedding_model):
    batch_size = 100
    print(f"Generating embeddings using HuggingFace ({embedding_model.model_name}) and uploading {len(textlist)} chunks...")

    for i in tqdm(range(0, len(textlist), batch_size)):
        lines_batch = textlist[i: i + batch_size]
        ids_batch = [str(uuid.uuid4()) for _ in lines_batch]
        embeddings_batch = embedding_model.embed_documents(lines_batch)

        collection.add(
            documents=lines_batch,
            embeddings=embeddings_batch,
            ids=ids_batch
        )


def process_pdf_file():
    config_path = "config_local.yaml"
    if not os.path.exists(config_path):
        print("Config file not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    local_index = LocalIndex()
    collection = local_index.get_collection()

    embed_model_name = config["vector_db"]["embedding_model"]
    print(f"Initializing embeddings with model: {embed_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model_name)

    data_path = config["data"]["path"]
    textlist = get_content(data_path)

    if not textlist:
        print("No content found or processed.")
        return

    vectorize_and_upload_pdf_content(collection, textlist, embedding_model)
    print("Ingestion complete.")


if __name__ == "__main__":
    process_pdf_file()