import streamlit as st
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import SimpleVectorStore
import openai
from PIL import Image
import random
import os
import json
import requests
import base64
from io import BytesIO
from streamlit_extras.let_it_rain import rain

st.set_page_config(
    page_title="Chatbot Title", 
    page_icon="👩🏻‍🏫", 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None
)

# Set OpenAI API key
openai.api_key = st.secrets.openai.api_key

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading, please wait... This may take 1-2 minutes."):
        # Load the finetuned model 
        ft_model_name = "ft:gpt-3.5-turbo-0125:dab07:tiffany:C1p3oxW6"
        
        # Configure Settings with the finetuned model (new approach)
        Settings.llm = OpenAI(model=ft_model_name, temperature=0.3)
        Settings.context_window = 2048
        Settings.num_output = 256
        Settings.chunk_size = 512
        
        # Set the system prompt in the LLM
        Settings.llm.system_prompt = "Give answers to users' questions."
        
        try:
            # Try to load with the old file structure first
            # Load vector store directly from the old format
            vector_store = SimpleVectorStore.from_persist_path("index.vecstore/vector_store.json")
            
            # Create storage context with the loaded vector store
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir="index.vecstore"
            )
            
            # Load the index with the storage context
            index = load_index_from_storage(storage_context)
            
        except Exception as e:
            # If that fails, try the standard approach
            try:
                storage_context = StorageContext.from_defaults(persist_dir="index.vecstore")
                index = load_index_from_storage(storage_context)
            except Exception as e2:
                st.error(f"Could not load index. Please ensure the index files exist in 'index.vecstore' directory.")
                st.error(f"Error details: {str(e2)}")
                raise e2
        
        return index

index = load_data()

# Create chat engine with the configured index
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Answer the following questions related to the data set"}
    ]

# Initialize detailed_prompt variable
detailed_prompt = None

# Handle user input
if prompt := st.chat_input("Ask Me Questions Relating to my data set."):
    # Save the original user question to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.new_question = True
    
    # Create a detailed prompt for the chat engine
    chat_history = ' '.join([message["content"] for message in st.session_state.messages])
    detailed_prompt = f"{chat_history} {prompt}"
    
    # Store detailed_prompt in session state for later use
    st.session_state.detailed_prompt = detailed_prompt

# Display chat messages when there's a new question
if "new_question" in st.session_state.keys() and st.session_state.new_question:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    st.session_state.new_question = False

# Generate response if last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            # Use the stored detailed_prompt or the last user message
            if "detailed_prompt" in st.session_state:
                prompt_to_use = st.session_state.detailed_prompt
            else:
                prompt_to_use = st.session_state.messages[-1]["content"]
            
            response = chat_engine.chat(prompt_to_use)
            st.write(response.response)
            
            # Append the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response.response})
