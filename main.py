from knowledge_extractor import KnowledgeExtractor
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import streamlit as st
from streamlit_chat import message
from utils import get_conversation_string, query_refiner

import yaml


if __name__ == "__main__":
    with open("./vector_db/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ke = KnowledgeExtractor()

    st.subheader("Redfield RAG chatbot")

    if "responses" not in st.session_state:
        st.session_state["responses"] = ["How can I assist you?"]

    if "requests" not in st.session_state:
        st.session_state["requests"] = []

    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        openai_api_key=config["openai"]["embedding"]["api_key"],
    )

    if "buffer_memory" not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k=10, return_messages=True
        )

    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I'm Sorry. This information is not in my database.'"""
    )

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template,
        ]
    )
    # Use st.session_state["buffer_memory"] instead of st.session_state.buffer_memory
    conversation = ConversationChain(
        memory=st.session_state["buffer_memory"],
        prompt=prompt_template,
        llm=llm,
        verbose=True,
    )

    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = ke.get_related_knowledge(
                    refined_query, top_k=5, passback_gpt=False
                )
                response = conversation.predict(
                    input=f"Context:\n {context} \n\n Query:\n{query}"
                )
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
    with response_container:
        if st.session_state["responses"]:
            for i in range(len(st.session_state["responses"])):
                message(st.session_state["responses"][i], key=str(i))
                if i < len(st.session_state["requests"]):
                    message(
                        st.session_state["requests"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
