import streamlit as st
from document_processing import load_document_and_create_vectorstore
from conversation_chain import get_conversation_chain

def initialize_chat(uploaded_file):
    """Initialize the conversation chain and set up chat history."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    vectorstore = load_document_and_create_vectorstore(uploaded_file, file_type)
    st.session_state.conversation = get_conversation_chain(vectorstore)
    st.session_state.chat_history = []

def handle_userinput(user_question):
    """Handle user input by generating a response using the conversational chain."""
    if st.session_state.conversation:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history.append((user_question, response['answer']))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Conversation chain is not initialized.")

def display_chat_history():
    """Display the conversation history."""
    st.write("Conversation history:")
    for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
        st.write(f"Q{i}: {question}")
        st.write(f"A{i}: {answer}")
