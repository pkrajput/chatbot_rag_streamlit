import streamlit as st
from knowledge_extractor import KnowledgeExtractor
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_conversation_string, query_refiner
import yaml
import os

# --- Page Config & CSS ---
st.set_page_config(page_title="Qwen Local Chat", page_icon="🤖")

st.markdown("""
<style>
    /* Android-like Bubble Styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    /* User Message - Greenish/Blue, Right-ish feel */
    [data-testid="stChatMessage"][data-test-user-name="user"] {
        background-color: #E3F2FD;
        border-bottom-right-radius: 2px;
    }
    /* Assistant Message - White/Grey */
    [data-testid="stChatMessage"][data-test-user-name="assistant"] {
        background-color: #F5F5F5;
        border-bottom-left-radius: 2px;
    }
    
    .stMarkdown {
        font-family: 'Roboto', sans-serif;
    }
    
    h1 {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Load local config
    config_path = "config_local.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ke = KnowledgeExtractor()

    st.title("Qwen Local Chat Assistant")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I assist you?", "context": None}
        ]

    # Initialize Local LLM (Ollama)
    llm = ChatOllama(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"]
    )

    # --- Display Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander("Context Retrieved"):
                    st.text(msg["context"])

    # --- Input Area ---
    if query := st.chat_input("Type your query here..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Processing
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Get History
                conversation_string = get_conversation_string()
                
                # 2. Refine Query (using history)
                refined_query = query_refiner(conversation_string, query, llm)
                
                # 3. Retrieve Context
                context_list = ke.get_related_knowledge(
                    refined_query, top_k=5, passback_gpt=False
                )
                context_text = "\n".join(context_list)
                
                # 4. Construct Explicit Prompt
                # The user requested explicit "Previous conversation" tracking
                
                final_prompt = f"""
Answer the question as truthfully as possible using the provided context.
If the answer is not contained within the text below, say 'I'm Sorry. This information is not in my database.'

Context:
{context_text}

Previous Conversation:
{conversation_string}

Query:
{query}
"""
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=final_prompt)
                ]

                # 5. Generate Response
                response_message = llm.invoke(messages)
                response = response_message.content
                
                st.markdown(response)
                
                # Show Context (Immediate)
                if context_text:
                    with st.expander("Context Retrieved"):
                        st.text(context_text)

        # Save Assistant Message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "context": context_text
        })
