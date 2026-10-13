import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

if not documents:
    raise ValueError("No PDF files found in the documents folder.")


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

print("Loading embedding model (first run downloads ~80MB)...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


# ============================================================
# ADD DOCUMENTS
# ============================================================

vectorstore.add_documents(chunks)

print("Successfully embedded and stored documents.")
print(f"Vector database saved to: {CHROMA_DIR}")