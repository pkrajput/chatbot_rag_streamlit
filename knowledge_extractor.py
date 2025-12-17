import yaml
from index_creator import LocalIndex
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

class KnowledgeExtractor:
    def __init__(self):
        with open("config_local.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.local_index = LocalIndex()
        self.collection = self.local_index.get_collection()
        
        # Switching back to HuggingFaceEmbeddings
        self.embed_model = HuggingFaceEmbeddings(
            model_name=self.config["vector_db"]["embedding_model"]
        )

    def get_related_knowledge(self, query, top_k=3, passback_gpt=False):
        if len(query) == 0:
            return [""]

        # Generate embedding for the query
        query_embedding = self.embed_model.embed_query(query)

        # Query Chroma using values
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # results['documents'] is a list of lists (batch queries)
        contexts = results['documents'][0] if results['documents'] else []

        if not passback_gpt:
            return contexts

        else:
            joined_context = "\n".join(contexts)
            
            llm = ChatOllama(
                model=self.config["llm"]["model"],
                base_url=self.config["llm"]["base_url"]
            )
            
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that provides accurate information given context. You do not use any external info to frame your answer but only the context. Reply without any precurser to text."
                ),
                HumanMessage(content=f"Context: {joined_context}\n\nQuery: {query}"),
            ]

            response = llm.invoke(messages)
            return response.content
