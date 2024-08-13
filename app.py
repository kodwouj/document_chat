import streamlit as st
from user_interface import handle_userinput, initialize_chat, display_chat_history

def main():
    st.title("Chat with Documents")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx"])

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            initialize_chat(uploaded_file)
        st.success("File processed successfully!")
    
    user_question = st.text_input("Ask a question about the document")

    if user_question:
        handle_userinput(user_question)
    
    if st.session_state.get('chat_history'):
        display_chat_history()

if __name__ == "__main__":
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    main()
