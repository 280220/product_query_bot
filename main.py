from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pipeline import get_langgraph
from langchain_core.messages import HumanMessage
import httpx
import os

# Load your LangGraph
graph = get_langgraph()

# Environment/config
CALLBACK_URL = os.getenv("CALLBACK_URL")

app = FastAPI()

# === API JSON-Based Endpoint ===
class QueryInput(BaseModel):
    user_id: str
    query: str


@app.post("/query")
async def handle_query(input_data: QueryInput):
    state = {
        "messages": [HumanMessage(content=input_data.query)]
    }

    result = graph.invoke(state, config={"configurable": {"thread_id": input_data.user_id}})
    final_message = result["messages"][-1]

    try:
        async with httpx.AsyncClient() as client:
            await client.post(CALLBACK_URL, json={
                "user_id": input_data.user_id,
                "answer": final_message.content
            })
    except Exception as e:
        print(f"[ERROR] Callback failed: {e}")
        print("Answer:", final_message.content)

    return {"status": "processed", "answer": final_message.content}


# === Local Callback Receiver for Debugging ===
@app.post("/callback")
async def callback_receiver(data: dict):
    print("Callback received at /callback:")
    print(data)
    return {"status": "received"}


# === Web UI Form ===
@app.get("/", response_class=HTMLResponse)
async def form():
    return """
    <form action="/query-form" method="post">
      <label>User ID:</label><br>
      <input type="text" name="user_id" value="user123"><br><br>
      <label>Query:</label><br>
      <input type="text" name="query" size="50"><br><br>
      <input type="submit" value="Ask">
    </form>
    """


@app.post("/query-form", response_class=HTMLResponse)
async def handle_form_query(
    user_id: str = Form(...),
    query: str = Form(...)
):
    if not query.strip():
        return """
        <h2>Answer:</h2>
        <p style="color:red;">⚠️ It seems like you forgot to ask a question.</p>
        <a href="/">🔙 Try again</a>
        """

    state = {"messages": [HumanMessage(content=query)]}
    result = graph.invoke(state, config={"configurable": {"thread_id": user_id}})
    final_message = result["messages"][-1]

    # Format response content for HTML (preserve line breaks)
    html_content = final_message.content.replace("\n", "<br>")

    return f"""
    <h2>Answer:</h2>
    <p>{html_content}</p>
    <hr>
    <a href="/">🔙 Ask another question</a>
    """