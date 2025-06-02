# Local Multimodal AI Chat - Multimodal chat application with Gemini
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
from chat_api_handler import ChatAPIHandler
from utils import get_timestamp, load_config, get_avatar, list_available_models, command
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from html_templates import css
from database_operations import (
    get_db_manager,
    close_db_manager,
    DEFAULT_CHAT_MEMORY_LENGTH,
    DEFAULT_RETRIEVED_DOCUMENTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
# vectordb_handler.py
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import chromadb
from langchain.embeddings import OpenAIEmbeddings # Assuming you use OpenAI embeddings
# ... rest of your code
import sqlite3
from auth_handler import show_login_page
import os
import shutil # Import shutil for directory operations

config = load_config()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def detoggle_pdf_chat():
    st.session_state.pdf_chat = False

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    db_manager = get_db_manager()
    db_manager.message_repo.delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def list_model_options():
    models = list_available_models()
    if st.session_state.endpoint_to_use == "gemini":
        return models["gemini"]
    elif st.session_state.endpoint_to_use == "openai":
        return models["openai"]
    return []

def update_model_options():
    st.session_state.model_options = list_model_options()

def initialize_session_state():
    # Initialize database manager
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = get_db_manager()

    # Initialize session key for chat history
    if "session_key" not in st.session_state:
        st.session_state.session_key = get_timestamp()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Load messages from database when initializing
        db_manager = get_db_manager()
        stored_messages = db_manager.message_repo.load_messages(st.session_state.session_key)
        if stored_messages:
            st.session_state.messages = [
                {"role": msg["sender_type"], "content": msg["content"]}
                for msg in stored_messages if msg["message_type"] == "text"
            ]

    if "endpoint_to_use" not in st.session_state:
        st.session_state["endpoint_to_use"] = "gemini"
    if "model_to_use" not in st.session_state:
        st.session_state["model_to_use"] = "gemini-pro"
    if "pdf_chat" not in st.session_state:
        st.session_state["pdf_chat"] = False
    if "retrieved_documents" not in st.session_state:
        st.session_state["retrieved_documents"] = 4
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    # PDF processing parameters
    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 1000  # Default chunk size
    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = 200  # Default overlap
    if "chat_memory_length" not in st.session_state:
        st.session_state["chat_memory_length"] = 4  # Default chat memory length

# --- NEW FUNCTION FOR CLEARING PDF DATA ---
def clear_pdf_data():
    """
    Deletes the ChromaDB directory where PDF embeddings are stored.
    Also clears the current chat session and forces a UI rerun.
    """
    chroma_db_path = "./chroma_db"
    if os.path.exists(chroma_db_path):
        try:
            shutil.rmtree(chroma_db_path)
            st.success("PDF knowledge base (ChromaDB) cleared successfully!")
            # Clear current chat messages as they were based on old PDF data
            st.session_state.messages = []
            # Force a new chat session to ensure a clean slate after clearing PDF data
            st.session_state.session_key = get_timestamp()
            st.rerun() # Rerun to refresh the UI and reflect the cleared state
        except Exception as e:
            st.error(f"Error clearing PDF knowledge base: {e}")
    else:
        st.info("No PDF knowledge base found to clear.")

def show_chat_interface():
    st.title(f"AI Chat Assistant - Welcome {st.session_state['username']}!")

    # Sidebar configuration
    with st.sidebar:
        st.title("Settings")

        # Session management
        all_sessions = st.session_state.db_manager.message_repo.get_all_chat_history_ids()
        if all_sessions:
            selected_session = st.selectbox(
                "Select Chat Session",
                ["New Session"] + all_sessions,
                index=0
            )
            if selected_session != "New Session":
                st.session_state.session_key = selected_session
                # Load messages for selected session
                stored_messages = st.session_state.db_manager.message_repo.load_messages(selected_session)
                st.session_state.messages = [
                    {"role": msg["sender_type"], "content": msg["content"]}
                    for msg in stored_messages if msg["message_type"] == "text"
                ]
                st.rerun()

        # Clear chat history button (for current session's text chat)
        if st.button("Clear Chat History"):
            db_manager = get_db_manager()
            db_manager.message_repo.delete_chat_history(st.session_state.session_key)
            st.session_state.messages = []
            st.rerun()

        # --- NEW BUTTON FOR CLEARING PDF DATA ---
        # This button specifically targets the ChromaDB (PDF embeddings)
        if st.button("Clear PDF Knowledge Base"):
            clear_pdf_data()

        # Endpoint selection
        st.session_state["endpoint_to_use"] = st.selectbox(
            "Select Endpoint",
            ["gemini", "openai"],
            index=0 if st.session_state["endpoint_to_use"] == "gemini" else 1
        )

        # Model selection based on endpoint
        if st.session_state["endpoint_to_use"] == "openai":
            models = ["gpt-3.5-turbo", "gpt-4"]
            default_index = 0 if st.session_state["model_to_use"] == "gpt-3.5-turbo" else 1
        else:
            models = ["gemini-pro"]
            default_index = 0

        st.session_state["model_to_use"] = st.selectbox(
            "Select Model",
            models,
            index=default_index
        )

        # PDF chat toggle
        st.session_state["pdf_chat"] = st.toggle("PDF Chat Mode", st.session_state["pdf_chat"])

        if st.session_state["pdf_chat"]:
            st.session_state["retrieved_documents"] = st.slider(
                "Number of documents to retrieve",
                min_value=1,
                max_value=10,
                value=st.session_state["retrieved_documents"]
            )

            pdf_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True
            )

            if pdf_files:
                add_documents_to_db(pdf_files)

        # Logout button
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.rerun()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("What is your question?"):
        # Save user message to database
        db_manager = get_db_manager()
        db_manager.message_repo.save_message(
            st.session_state.session_key,
            "user",
            "text",
            user_input
        )

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            image = None

            # Handle image upload - this part of the code needs review.
            # `st.file_uploader` within `st.chat_input` block won't work as expected.
            # File uploader should generally be outside the chat input loop or handled differently.
            # For multimodal input with chat_input, you typically handle text and then an image separately.
            # If you intend to pass an image with each user_input, reconsider the UI flow.
            # For now, I'm leaving it as is, but flagging it.
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = uploaded_file.read()

            llm_answer = ChatAPIHandler.chat(user_input=user_input, chat_history=st.session_state.messages, image=image)

            # Save assistant message to database
            db_manager.message_repo.save_message(
                st.session_state.session_key,
                "assistant",
                "text",
                llm_answer
            )

            message_placeholder.markdown(llm_answer)
        st.session_state.messages.append({"role": "assistant", "content": llm_answer})

def main():
    initialize_session_state()

    # Show login page if not logged in
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()