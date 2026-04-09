from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import os
import json
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import hashlib

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DB_NAME = str(Path(__file__).parent / "chroma_db")

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 1000


folders = [f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_dir()]

documents = []
for  folder in folders:
    print(f"Processing folder: {folder}")
    doc_type = os.path.basename(folder)

    loader = DirectoryLoader(str(folder), glob="**/*.md" , loader_cls= TextLoader,loader_kwargs={'encoding': 'utf-8'}, show_progress=True)
    folder_docs = loader.load()
    print(f"Loaded {len(folder_docs)} documents from folder: {folder}")
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)


print(documents[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=AVERAGE_CHUNK_SIZE, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")


#embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_stable_id(chunk, index):
    """Creates a unique ID based on the file source and chunk index."""
    # 'source' is automatically added to metadata by DirectoryLoader
    source = chunk.metadata.get("source", "unknown")
    unique_str = f"{source}_{index}"
    return hashlib.md5(unique_str.encode()).hexdigest()

# 1. Generate stable IDs for all chunks
ids = [get_stable_id(chunk, i) for i, chunk in enumerate(chunks)]

client = PersistentClient(path=DB_NAME)

vector_store = Chroma(collection_name="my_collection", embedding_function=embeddings, persist_directory=DB_NAME)
vector_store.add_documents(documents=chunks, ids=ids)
print(client.list_collections())
