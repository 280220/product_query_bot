from agents.qa_tools import retrieve, llm
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Generate a tool call or direct response
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define the tool node (retriever)
tools = ToolNode([retrieve])

# Generate final response with citations
def generate(state: MessagesState):
    # Extract recent tool messages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = list(reversed(recent_tool_messages))

    # Build context from tool results
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_prompt = (
        "You are a Q&A assistant. Answer concisely (max 3 sentences), "
        "using only the context below:\n\n" + docs_content
    )

    # Construct message sequence
    conversation = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]
    prompt = [SystemMessage(system_prompt)] + conversation

    # Invoke LLM
    response = llm.invoke(prompt)

    # Get only tool messages related to the most recent user question
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        elif message.type == "human":
            break
    tool_messages = list(reversed(recent_tool_messages))

    # Add source citations
    citations = []
    for tm in tool_messages:
        docs = tm.artifact if isinstance(tm.artifact, list) else []
        for doc in docs:
            quote = doc.page_content.strip().replace("\n", " ")
            title = doc.metadata.get("title", "Unknown Product")
            similarity = doc.metadata.get("similarity", "N/A")
            citations.append(f"- (Product: {title}, Similarity: {similarity}) — “{quote[:150]}...”")

    # Add to response
    full = response.content.strip()
    full += "\n\n📚 **Sources:**\n" + "\n".join(citations)
    response.content = full

    return {"messages": [response]}

# Compile LangGraph
def get_langgraph():
    builder = StateGraph(MessagesState)
    builder.add_node("query_or_respond", query_or_respond)
    builder.add_node("tools", tools)
    builder.add_node("generate", generate)

    builder.set_entry_point("query_or_respond")
    builder.add_conditional_edges("query_or_respond", tools_condition, {
        END: END,
        "tools": "tools"
    })
    builder.add_edge("tools", "generate")
    builder.add_edge("generate", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)