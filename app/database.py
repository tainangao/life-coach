# from supabase import create_client, Client

# from config import settings

# # Initialize the Supabase Client
# # Note: We use the SERVICE_ROLE_KEY if we want to bypass RLS for administrative 
# # tasks (like batch ingest), but we'll use the user context for chat.
# supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# def get_supabase_client() -> Client:
#     """Returns the initialized Supabase client."""
#     return supabase


# def get_user_vector_filter(hf_user_id: str):
#     """
#     Returns a filter object for LangChain's SupabaseVectorStore 
#     to ensure users only retrieve their own docs or global docs.
#     """
#     return {
#         "or": f"(user_id.eq.{hf_user_id}, is_global.eq.true)"
#     }

# client = get_supabase_client()
# print("Supabase client initialized:", client)

# # Example usage of get_user_vector_filter
# example_user_id = "user_123"
# filter_obj = get_user_vector_filter(example_user_id)
# print("Filter object for user:", filter_obj)
# from sqlalchemy import create_engine
# # from sqlalchemy.pool import NullPool
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env
# load_dotenv()

# # Fetch variables
# USER = os.getenv("user")
# PASSWORD = os.getenv("password")
# HOST = os.getenv("host")
# PORT = os.getenv("port")
# DBNAME = os.getenv("dbname")

# # Construct the SQLAlchemy connection string
# DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

# # Create the SQLAlchemy engine
# engine = create_engine(DATABASE_URL)
# # If using Transaction Pooler or Session Pooler, we want to ensure we disable SQLAlchemy client side pooling -
# # https://docs.sqlalchemy.org/en/20/core/pooling.html#switching-pool-implementations
# # engine = create_engine(DATABASE_URL, poolclass=NullPool)

# # Test the connection
# try:
#     with engine.connect() as connection:
#         print("Connection successful!")
# except Exception as e:
#     print(f"Failed to connect: {e}")
    

import dotenv
import os
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, declarative_base
from models.db_models import ParentDocs, ChildDocs, Dummy
import random

dotenv.load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


Base.metadata.create_all(engine)


class DBService:
    def __init__(self):
        self.session = SessionLocal()

    def health_check(self):
        """Check database connectivity."""
        try:
            self.session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def clear_all(self):
        """Clear all documents (useful for reset)."""
        self.session.query(ParentDocs).delete()
        self.session.query(ChildDocs).delete()
        self.session.commit()

    def store_docs(self, chunks: list, embedding: list):
        for chunk, emb in zip(chunks, embedding):
            doc = Dummy(
                content=chunk,
                embedding=emb
            )
            self.session.add(doc)
        self.session.commit()
        print(f"Stored {len(chunks)} documents.")
    
    def retrieve_context(self, query_embedding: list, top_k: int = 3) -> str:
        """
        Retrieve most relevant context using vector similarity.
        <->: Euclidean distance (L2 distance).
        <=>: Cosine distance.
        <#>: Negative inner product (useful when embeddings are normalized, as it's equivalent to cosine distance).
        """
        results = self.session.execute(
            text(f"SELECT content FROM ingester.dummy ORDER BY (embedding <-> '{query_embedding}') LIMIT {top_k}"),
        ).fetchall()
        return "\n".join([r[0] for r in results])

if __name__ == "__main__":
    print("Running local DBService test...")

    db = DBService()
    if db.health_check():
        print("Database connection is healthy.")
    else:
        print("Database connection failed.")

    dim = 1536
    dummy_chunks = ["This is document A", "This is document B"]
    dummy_embeddings = [[random.random() for _ in range(dim)] for _ in range(len(dummy_chunks))]
    
    # print("Storing dummy documents...")
    # db.store_docs(dummy_chunks, dummy_embeddings)
    
    print("Retrieving context for a random query...")
    query_embedding = [random.random() for _ in range(dim)]
    context = db.retrieve_context(query_embedding, top_k=2)
    print(f"Retrieved context:\n{context}")
    
    