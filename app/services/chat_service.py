from typing import List, Dict, Any, Optional, AsyncGenerator

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.database import supabase

# --- Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", stream=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- Helper: Fetch Message History & Sequence ---
async def get_chat_history(conversation_id: str) -> List[Any]:
    """Fetches messages for a conversation, ordered by sequence_number."""
    response = supabase.table("messages") \
        .select("role, content") \
        .eq("conversation_id", conversation_id) \
        .order("sequence_number", desc=False) \
        .execute()

    history = []
    for m in response.data:
        if m['role'] == 'user':
            history.append(HumanMessage(content=m['content']))
        else:
            history.append(AIMessage(content=m['content']))
    return history


async def get_next_sequence(conversation_id: str) -> int:
    """Calculates the next sequence number for a new message."""
    response = supabase.table("messages") \
        .select("sequence_number") \
        .eq("conversation_id", conversation_id) \
        .order("sequence_number", desc=True) \
        .limit(1) \
        .execute()

    if not response.data:
        return 1
    return response.data[0]['sequence_number'] + 1


# --- Main Chat Logic ---
async def handle_chat(
        history_ui: List[Dict],
        conv_id: Optional[str],
        hf_user_id: str
) -> AsyncGenerator[str, None]:
    user_query = history_ui[-1]["content"]

    # 1. Ensure Conversation Exists
    if not conv_id:
        # Create new conversation and auto-generate title
        new_conv = supabase.table("conversations").insert({
            "hf_user_id": hf_user_id,
            "title": "New Session"  # Default
        }).execute()
        conv_id = new_conv.data[0]['id']

        # Trigger background task to rename it (Gemini logic)
        title_prompt = f"Summarize this query into a 3-word title: {user_query}"
        title = (await llm.ainvoke(title_prompt)).content
        supabase.table("conversations").update({"title": title}).eq("id", conv_id).execute()

    # 2. Setup Vector Retriever (With Metadata Filter)
    # This enforces that user only sees Global data OR their own data
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )

    # Filter Logic: (is_global == true) OR (user_id == hf_user_id)
    # Note: Supabase/Postgres RPC 'match_documents' needs to handle this logic internally
    retriever = vector_store.as_retriever(search_kwargs={
        "filter": {"user_id": hf_user_id}  # Simplified filter
    })

    # 3. Construct RAG Chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an empathetic Life Coach. Use the context to guide your advice."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}\n\nContext:\n{context}")
    ])

    chain = (
            {"context": retriever, "chat_history": lambda x: x["chat_history"], "input": lambda x: x["input"]}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 4. Stream Response & Save to DB
    full_response = ""
    history_objs = await get_chat_history(conv_id)

    # Save User Message first
    user_seq = await get_next_sequence(conv_id)
    supabase.table("messages").insert({
        "conversation_id": conv_id, "role": "user",
        "content": user_query, "sequence_number": user_seq
    }).execute()

    async for chunk in chain.astream({"input": user_query, "chat_history": history_objs}):
        full_response += chunk
        yield chunk

    # Save AI Message
    ai_seq = user_seq + 1
    supabase.table("messages").insert({
        "conversation_id": conv_id, "role": "assistant",
        "content": full_response, "sequence_number": ai_seq
    }).execute()
