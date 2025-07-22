# build_index.py
from preprocess_data import load_docs
from rag_pipeline import build_and_save_index

docs = load_docs()

if not docs:
    print("❌ No documents found.")
else:
    build_and_save_index(docs)
    print("✅ Indexing completed successfully.")