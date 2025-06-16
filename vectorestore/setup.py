import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DOCS_DIR = "docs"
PERSIST_DIR = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

def load_documents(directory=DOCS_DIR):
    """Loads .txt documents from the given directory."""
    loader = DirectoryLoader(directory)
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from '{directory}':")
    for doc in docs:
        print(f"  • {os.path.basename(doc.metadata['source'])}")

    return docs

def split_documents(documents):
    """Splits documents into chunks suitable for vector storage."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(documents)

    print(f"Total chunks generated: {len(chunks)}")

    return chunks

def build_vectorstore():
    """Builds or loads a Chroma vector store from local disk."""

    # Remove existing DB if requested
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("Existing vectorstore deleted.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    documents = load_documents()
    chunks = split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("New vectorstore created and saved to disk.")

    return vectordb

def load_vectorstore():
    """Load an existing vectorstore from disk."""
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        raise RuntimeError("Vectorstore not found. Run build_vectorstore() first.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)