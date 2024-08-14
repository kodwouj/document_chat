import streamlit as st
import asyncio
from user_interface import handle_userinput, initialize_chat, display_chat_history, run_async_initialize_chat

async def main():
    """
    Main function to run the Streamlit app. Handles file upload, question input, 
    and displays chat history.

    It initializes the chat processing asynchronously and manages the user 
    interface elements like file upload and text input.

    Returns:
        None
    """
    st.title("Chat with Documents")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx", "pptx"])

    if uploaded_file is not None:
        try:
            with st.spinner("Processing file..."):
                await initialize_chat(uploaded_file)
            st.success("File processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    user_question = st.text_input("Ask a question about the document")

    if user_question:
        try:
            handle_userinput(user_question)
        except Exception as e:
            st.error(f"Error handling user input: {e}")

    if st.session_state.get('chat_history'):
        display_chat_history()

if __name__ == "__main__":
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    asyncio.run(main())
