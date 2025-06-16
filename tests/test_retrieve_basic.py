from agents.qa_tools import retrieve

def test_retrieve_returns_documents():
    query = "What are the features of the LensPro 4K Webcam?"
    result = retrieve.invoke({"query": query})

    # Validate structure
    assert isinstance(result, dict)
    assert "content" in result
    assert "artifact" in result
    assert isinstance(result["artifact"], list)

    # Validate at least 1 document was returned
    assert len(result["artifact"]) > 0
    assert all(hasattr(doc, "page_content") for doc in result["artifact"])
