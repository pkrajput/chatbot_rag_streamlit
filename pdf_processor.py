import os
import yaml
import openai
import textract
from tqdm.auto import tqdm
from index_creator import PineconeIndex


class TextDocument:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


def load_file(filename):
    text = textract.process(filename).decode("utf-8")
    return text.split("\n\n")


def chunk_text_by_word_count(paragraphs, word_limit=1000):
    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # If adding this paragraph exceeds the word limit, finalize the current chunk
        if current_word_count + paragraph_word_count > word_limit:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def get_content(root_dir):
    filenames = [os.path.join(root_dir, name) for name in os.listdir(root_dir)]

    split_pages_text = []
    for filename in filenames:
        paragraphs = load_file(filename)
        chunks = chunk_text_by_word_count(paragraphs)
        for chunk in chunks:
            processed_text = (
                chunk.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
            )
            split_pages_text.append(processed_text)
    return split_pages_text


def vectorize_and_upload_pdf_content(
    index, embedding_model_name, textlist, openai_api_key
):
    batch_size = 100
    openai.api_key = openai_api_key
    for i in tqdm(range(0, len(textlist), batch_size)):
        lines_batch = textlist[i : i + batch_size]
        res = openai.Embedding.create(input=lines_batch, engine=embedding_model_name)
        embeddings = [record["embedding"] for record in res["data"]]
        meta = [{"text": line} for line in lines_batch]
        index_id = [str(j) for j in range(i, i + len(lines_batch))]
        to_upsert = zip(index_id, embeddings, meta)
        index.upsert(vectors=list(to_upsert))


def process_pdf_file():
    with open("./vector_db/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    index = PineconeIndex().get_index()
    openai_api_key = config["openai"]["embedding"]["api_key"]
    embedding_model_name = config["openai"]["embedding"]["model_name"]
    textlist = get_content(config["pinecone"]["path"])
    vectorize_and_upload_pdf_content(
        index, embedding_model_name, textlist, openai_api_key
    )


if __name__ == "__main__":
    process_pdf_file()
