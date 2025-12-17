import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

def query_refiner(conversation, query, llm):
    
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    
    # Using ChatOllama which expects messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content

def get_conversation_string():
    conversation_string = ""
    # Ensure messages exist
    if "messages" not in st.session_state:
        return ""
        
    messages = st.session_state["messages"]
    
    # Iterate through messages, excluding the last one if it's the current pending user query
    # (The refiner calls this after appending user query but before assistant response)
    # We want past history.
    
    for msg in messages[:-1]:
        if msg["role"] == "user":
            conversation_string += "Human: " + msg["content"] + "\n"
        elif msg["role"] == "assistant":
            conversation_string += "Bot: " + msg["content"] + "\n"
            
    return conversation_string
