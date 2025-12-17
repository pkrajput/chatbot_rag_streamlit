import chromadb
import yaml
import os
from chromadb.config import Settings

class LocalIndex:
    def __init__(self):
        config_path = "config_local.yaml"
        # Fallback to older config if local doesn't exist (or we can insist on local)
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.persist_directory = config["vector_db"]["path"]
        self.collection_name = "pdf_knowledge_base" # Can be configurable
        
        # Initialize Chroma Client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        # We don't need to specify embedding function here if we handle embeddings externally
        # But Chroma can handle them if we passed an embedding function. 
        # For now, let's keep it simple and just get the collection.
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def get_collection(self):
        return self.collection

if __name__ == "__main__":
    idx = LocalIndex()
    print(f"Index initialized at {idx.persist_directory}")
