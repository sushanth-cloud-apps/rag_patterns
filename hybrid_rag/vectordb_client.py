
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

VECTOR_DB_DIR = "./chroma_db"
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=ollama_embeddings,
    collection_name="my_collection",
)

