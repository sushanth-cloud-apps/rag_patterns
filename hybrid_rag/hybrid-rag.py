import hashlib

from utils import chunk_utils

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,ToMarkdownLoader
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from vectordb_client import vectorstore


knowledge_base_dir = Path(__file__).parent.parent / "knowledge-base"

files = []
LOADER_MAP = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.pptx': UnstructuredPowerPointLoader
}

for folder in knowledge_base_dir.iterdir():
    if folder.is_dir():
        for file in folder.iterdir():
            files.append({'category': folder.name, 'path': file , "extension": file.suffix , "file_name": file.name})


def get_loader(file):
    loader_class = LOADER_MAP.get(file['extension'])
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file['extension']}")
    return loader_class(file['path'])



def stream_documents(files):
    for file in files:
        print(f"Processing file: {file['path']} with extension: {file['extension']}")
        try:
           loader = get_loader(file)
           print(f"Using loader: {type(loader).__name__} for file: {file['path']}")
           documents = loader.load()
           for doc in documents:
               doc.metadata["source"] = str(file['path'])
               doc.metadata["category"] = file['category']
               doc.metadata["file_name"] = file['file_name']
               doc.metadata["document_id"] = hashlib.sha256((str(file['path']) + file['file_name']).encode()).hexdigest()
               doc.metadata["document_hash"] = hashlib.sha256(doc.page_content.encode()).hexdigest()
               yield doc
        except Exception as e:
            print(f"Error processing file {file['path']}: {e}")
            exception_message = str(e)
            exit(1)




def stream_batches(files,batch_size=10):
    batch = []
    for doc in stream_documents(files):
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")


def get_chunks(new_or_changed_document_ids, batch):
    # Initialize the semantic splitter
    text_splitter = SemanticChunker(
        ollama_embeddings,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=0.5
    )
    
    # Filter batch for only the specific docs we need
    documents_to_chunk = [doc for doc in batch if doc.metadata["document_id"] in new_or_changed_document_ids]
    
    print(f"Creating semantic chunks for {len(documents_to_chunk)} documents...")
    
    # Generate chunks
    doc_chunks = text_splitter.split_documents(documents_to_chunk)
    for chunk in doc_chunks:
        chunk.metadata["chunk_id"] = chunk_utils.generate_chunk_id(chunk.metadata["document_id"], chunk)
    return doc_chunks
        
def update_index_store(index_store, chunks):
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        chunk_id = chunk.metadata["chunk_id"] 
        document_hash = chunk.metadata["document_hash"]

        if doc_id not in index_store.keys():
            index_store[doc_id] = {
                "source": chunk.metadata["source"],
                "file_name": chunk.metadata["file_name"],
                "document_hash": document_hash,
                "chunks": []
            }
        else:
            index_store[doc_id]["document_hash"] = document_hash

        index_store[doc_id]["chunks"].append({
            "chunk_id": chunk_id,
            "content_length": len(chunk.page_content)
        })
    
    return index_store





def process_batch(batch , index_store):
    print(f"Processing batch of {len(batch)} documents")

    ##check if document is new or has changed by comparing document_hash in metadata with existing index_store
    #if document is new or has changed, generate chunks and update index_store
    document_ids_in_batch = set([doc.metadata["document_id"] for doc in batch])
    existing_document_ids = set(index_store.keys())
    print(f"Existing document IDs in index store: {len(existing_document_ids)}")
    #totally new document ids
    new_or_changed_document_ids = document_ids_in_batch - existing_document_ids
    docs_to_remove = set()

    for doc in batch:
        if doc.metadata["document_id"] in existing_document_ids:
            existing_hash = index_store[doc.metadata["document_id"]]["document_hash"]
            if existing_hash != doc.metadata["document_hash"]:
                new_or_changed_document_ids.add(doc.metadata["document_id"])
                docs_to_remove.add(doc.metadata["document_id"])
    

     # Clean up Vector DB and local index for changed docs
    for doc_id in docs_to_remove:
        print(f"🧹 Removing old version of document: {doc_id}")
        # Delete all chunks associated with this document_id

        vectorstore.delete(where={"document_id": doc_id})
        # Reset the local tracking for chunks
        index_store[doc_id]["chunks"] = []

    chunks = get_chunks(new_or_changed_document_ids , batch)
    
    #update index_store with new document hashes for all documents in batch
    update_index_store(index_store, chunks)

    return chunks
    #embeddings = get_embeddings(chunks)
    #chroma.add_documents(chunks, embeddings)




if __name__ == "__main__":
    load_dotenv()
    index_store = chunk_utils.load_index_store("index_store.json")
    for batch in stream_batches(files, batch_size=10):
        print(f"Processing batch of {len(batch)} documents")
        chunks = process_batch(batch , index_store)
        enriched_chunks = chunk_utils.enrich_chunks(chunks)
        vectorstore.add_documents(enriched_chunks,ids = [chunk.metadata["chunk_id"] for chunk in enriched_chunks],collection_name="my_collection",)
    
    chunk_utils.save_index_store("index_store.json", index_store)

    print("Ingestion complete!")
    

    

    

        

