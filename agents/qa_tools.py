from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from vectorestore.setup import load_vectorstore
from config import TOP_K
from langchain_core.documents import Document

# Load persistent vector store
_vectordb = load_vectorstore()

# LLaMA 3.2 setup
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Retrieval tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve documents with similarity score for citation."""
    docs_and_scores = _vectordb.similarity_search_with_score(query, k=TOP_K)

    serialized_blocks = []
    retrieved_docs = []

    for doc, score in docs_and_scores:
        doc.metadata["similarity"] = round(score, 3)
        doc.metadata["title"] = doc.metadata.get("source", "No Title")
        serialized_blocks.append(
            f"(Similarity: {score:.3f}) {doc.page_content[:100]}..."
        )
        retrieved_docs.append(doc)

    return "\n\n".join(serialized_blocks), retrieved_docs
