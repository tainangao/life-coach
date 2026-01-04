import os
from typing import List, Optional

from app.utils.chunking import get_chunker
from app.utils.parsers import parse_file_to_text
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.database import supabase

# Initialize Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GOOGLE_API_KEY
)


async def process_file(
        file_path: str,
        strategy_name: str,
        hf_user_id: Optional[str]
) -> str:
    """
    Processes an uploaded file: Parse -> Chunk -> Embed -> Store.
    """
    if not hf_user_id:
        return "❌ Error: You must be logged in to upload files."

    try:
        # 1. Parse File to Text
        # We extract the file name for metadata
        file_name = os.path.basename(file_path)
        text = parse_file_to_text(file_path)

        if not text.strip():
            return "❌ Error: The file appears to be empty."

        # 2. Get Chunking Strategy
        chunker = get_chunker(strategy_name)

        # 3. Create Documents with Metadata
        # We tag documents with user_id for RAG isolation
        raw_chunks = chunker.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "user_id": hf_user_id,
                    "is_global": False
                }
            )
            for chunk in raw_chunks
        ]

        # 4. Upsert to Supabase Vector Store
        # LangChain's SupabaseVectorStore handles the embedding and insertion
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"  # This matches the RPC function in Postgres
        )

        vector_store.add_documents(documents)

        return f"✅ Successfully processed {len(documents)} chunks using '{strategy_name}' strategy."

    except Exception as e:
        print(f"Ingestion Error: {e}")
        return f"❌ Error processing file: {str(e)}"


def ingest_global_docs(file_paths: List[str]):
    """
    Helper for the 'Batch' requirement. 
    Processes documents that all users can access.
    """
    # Implementation logic similar to process_file but sets is_global=True
    # and user_id=None
    pass
