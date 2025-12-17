# 🤖 Local RAG Chatbot with Streamlit & Ollama

Welcome to the **Local RAG (Retrieval-Augmented Generation) Chatbot**! This project allows you to chat with your own documents (PDFs) privately on your local machine. It uses powerful open-source models to find relevant information from your files and answer your questions accurately.

---

## ✨ Features

*   **完全 Private & Local**: Runs 100% on your machine. No data leaves your computer.
*   **Document Chat**: Upload PDF files and get answers based *only* on their content.
*   **Vector Search**: Uses a high-performance vector database (ChromaDB) to find the most relevant text chunks.
*   **Smart Query Refinement**: The AI rephrases your follow-up questions to understand context from previous messages.
*   **Interactive UI**: Clean, responsive chat interface built with Streamlit.
*   **Reference Citations**: Shows you exactly which parts of the document were used to generate the answer.

---

## 🛠️ Tech Stack & Models

This project combines several state-of-the-art tools:

*   **Interface**: [Streamlit](https://streamlit.io/)
*   **LLM (Large Language Model)**: `qwen3:8b` running via [Ollama](https://ollama.com/)
*   **Embeddings Model**: `all-MiniLM-L6-v2` (via HuggingFace) for converting text into numbers.
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) for storing and searching text.
*   **Orchestration**: [LangChain](https://python.langchain.com/)

---

## ⚙️ Prerequisites

Before you start, make sure you have the following installed:

1.  **Python 3.9+**
2.  **[Ollama](https://ollama.com/)**: This is required to run the LLM locally.
    *   Download and install Ollama.
    *   Pull the specific model used in this project by running this command in your terminal:
        ```bash
        ollama pull qwen3:8b
        ```

---

## 🚀 Installation & Setup

1.  **Clone the Repository** (or download the files)
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ How to Run

### Step 1: Add Your Documents
1.  Place your PDF files into the `data/` folder.
2.  (If the folder doesn't exist, create one named `data` in the root directory).

### Step 2: Ingest the Documents
Before you can chat, the system needs to read and "learn" your documents. Run this script:

```bash
python pdf_processor.py
```

*   **What this does**: It reads all PDFs in the `data/` folder, splits them into small distinct chunks, creates mathematical embeddings for each chunk using `all-MiniLM-L6-v2`, and saves them into the local Chroma database (`chroma_db/`).

### Step 3: Start the Chatbot
Launch the web interface:

```bash
streamlit run main.py
```

*   A new tab should automatically open in your browser (usually at `http://localhost:8501`).
*   Start chatting! Ask questions about the documents you added.

---

## 🔧 Configuration Details

You can customize the behavior of the app by editing `config_local.yaml` or specific python files.

### 1. Changing the LLM Model
By default, this project uses **`qwen3:8b`**. To use a different model (e.g., `llama3` or `mistral`):
1.  Open `config_local.yaml`.
2.  Change the `model` value under `llm`:
    ```yaml
    llm:
      type: "ollama"
      model: "llama3"  # Change this to your preferred model
    ```
3.  Make sure you have pulled that model in Ollama (`ollama pull llama3`).

### 2. Changing the Embedding Model
The default embedding model is **`all-MiniLM-L6-v2`**. It is lightweight and effective. To change it:
1.  Open `config_local.yaml`.
2.  Update the `embedding_model` field:
    ```yaml
    vector_db:
      embedding_model: "sentence-transformers/all-mpnet-base-v2" # Example of a larger model
    ```
3.  **Note**: If you change the embedding model, you **must** delete the `chroma_db` folder and re-run `python pdf_processor.py` to re-generate the database.

### 3. Adjusting Retrieval Count (Advanced)
By default, the bot retrieves the **top 5** most relevant chunks of text to answer your question.
*   **To change this**: Open `main.py` and look for line ~89:
    ```python
    context_list = ke.get_related_knowledge(
        refined_query, top_k=5, passback_gpt=False
    )
    ```
    Change `top_k=5` to any number you prefer (e.g., `top_k=3` for faster, more focused answers, or `top_k=10` for more context).

---

## 📂 Project Structure

*   `main.py`: The frontend application (Streamlit). Handles user input and displays chat.
*   `pdf_processor.py`: The script that reads PDFs and builds the vector database.
*   `knowledge_extractor.py`: Logic for searching the database and generating answers.
*   `utils.py`: Helper functions for query refinement and chat history management.
*   `config_local.yaml`: Central configuration file for model names and paths.
*   `requirements.txt`: List of python libraries required.
*   `data/`: Directory where you put your PDF files.
*   `chroma_db/`: Directory where the vector database stores data (automatically created).

---

## ❓ FAQ

**Q: I get a "Model not found" error.**
A: Make sure you installed Ollama and ran `ollama pull qwen3:8b` (or whichever model is in your config).

**Q: The bot says "I don't know" even though the info is in the PDF.**
A: Try increasing the `top_k` value in `main.py` to give the model more context, or try rephrasing your question to match the document's wording more closely.

**Q: Can I use this without GPU?**
A: Yes! Ollama runs surprisingly well on CPU, though a GPU will make it much faster.
