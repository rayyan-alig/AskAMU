# rag_pipeline.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.core.settings import Settings

# === Load environment variables ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chat-amu-index")
PERSIST_DIR = "data/vector_index"

# === Validate environment ===
if not all([PINECONE_API_KEY, GOOGLE_API_KEY, PINECONE_INDEX_NAME]):
    raise ValueError("❌ Missing environment variables. Please check your .env file.")

# === LlamaIndex settings ===
Settings.llm = Gemini(api_key=GOOGLE_API_KEY, model_name="models/gemini-2.0-flash")
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/text-embedding-004")
Settings.chunk_size = 1024
Settings.chunk_overlap = 100

# === Initialize Pinecone & VectorStore ===
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)



def build_and_save_index(documents):
    print(f"📄 Building index from {len(documents)} documents...")

    if not documents:
        print("⚠️ No documents to index.")
        return

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"🧱 Parsed into {len(nodes)} nodes.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    print("✅ Index built, upserted, and saved.")


# === Load index from local + vector store ===
def load_index():
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR,
        vector_store=vector_store,
    )
    index = load_index_from_storage(storage_context)
    return index

# === Query the index (RAG) ===
def query_rag(user_query, top_k=5):
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k, streaming=True)
    result = query_engine.query(user_query)
    return str(result.response), [node.metadata for node in result.source_nodes]

# === Get total vectors in Pinecone ===
def get_vector_count():
    stats = pinecone_index.describe_index_stats()
    return stats.get("total_vector_count", 0)

