import tempfile
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DocxLoader, TextLoader, CSVLoader, ExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document_and_create_vectorstore(uploaded_file, file_type):
    """Load a document from an UploadedFile object and create a vector store."""
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if file_type == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_type == "docx":
        loader = DocxLoader(temp_file_path)
    elif file_type == "txt":
        loader = TextLoader(temp_file_path)
    elif file_type == "csv":
        loader = CSVLoader(temp_file_path)
    elif file_type == "xlsx":
        loader = ExcelLoader(temp_file_path)
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore
