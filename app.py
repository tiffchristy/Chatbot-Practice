import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import openai
from PIL import Image
import random
import os
import json
import requests
import base64
from io import BytesIO
from streamlit_extras.let_it_rain import rain


st.set_page_config(page_title="Chatbot Title", page_icon="ðŸ‘©ðŸ»â€ðŸ«", layout="centered", initial_sidebar_state="auto", menu_items=None)

#Context

# Set OpenAI API key
openai.api_key = st.secrets.openai.api_key

# Display centered text
#st.markdown("<p style='text-align: center;'>Welcome to Chatbot Page!</p>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading, please wait... This may take 1-2 minutes."):
        
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(persist_dir="index.vecstore")

        # Load the index
        index = load_index_from_storage(storage_context)

        # Load the finetuned model 
        ft_model_name = "ft:gpt-4.1-mini-2025-04-14:dab07:asa:C1b8U3av"
        ft_context = ServiceContext.from_defaults(llm=OpenAI(model=ft_model_name, temperature=0.3), 
        context_window=2048, 
        
        system_prompt="""Give answers to users' questions.
        """
        )           
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Me questions related to the data set"}
    ]

if prompt := st.chat_input("Ask Me Questions Relating to my data set."):
    # Save the original user question to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.new_question = True

    # Create a detailed prompt for the chat engine
    chat_history = ' '.join([message["content"] for message in st.session_state.messages])
    detailed_prompt = f"{chat_history} {prompt}"

if "new_question" in st.session_state.keys() and st.session_state.new_question:
   for message in st.session_state.messages: # Display the prior chat messages
       with st.chat_message(message["role"]):
           st.write(message["content"])
   st.session_state.new_question = False # Reset new_question to False

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
   with st.chat_message("assistant"):
       with st.spinner("Calculating..."):
           response = chat_engine.chat(detailed_prompt)
           st.write(response.response)
           # Append the assistant's detailed response to the chat history
           st.session_state.messages.append({"role": "assistant", "content": response.response})



