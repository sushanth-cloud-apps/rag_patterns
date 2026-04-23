
from collections import defaultdict
import hashlib
import json
import os


class chunk_utils:

    @staticmethod
    def generate_chunk_id(doc_id, chunk):
        normalized_chunk_content = " ".join(chunk.page_content.strip().lower().split())
        unique_string = f"{doc_id}|{normalized_chunk_content}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    @staticmethod
    def enrich_chunks(chunks):
        enriched_chunks = []
        for chunk in chunks:
            source = chunk.metadata["source"]
            doc_id = chunk.metadata["document_id"]
            chunk_id = chunk_utils.generate_chunk_id(doc_id,chunk)
            chunk.metadata.update({"chunk_id": chunk_id, "content_length": len(chunk.page_content)})
            enriched_chunks.append(chunk)

        return enriched_chunks
    

    

    @staticmethod
    def load_index_store(filename):
        if not os.path.exists(filename):
            return {}
        with open(filename, "r") as f:
            return json.load(f)
    
    @staticmethod
    def save_index_store(filename, index_store):
        
        with open(filename, "w") as f:
            json.dump(index_store, f, indent=4)



    
    @staticmethod
    def update_index_store(index_store, enriched_chunks):
        """
        Updates the index store dictionary with new chunks, grouped by document_id.
        """
        for chunk in enriched_chunks:
            doc_id = chunk.metadata["document_id"]
            chunk_id = chunk.metadata["chunk_id"] 
            
            # If we haven't seen this document before, initialize its entry
            if doc_id not in index_store:
                index_store[doc_id] = {
                    "source": chunk.metadata["source"],
                    "file_name": chunk.metadata["file_name"],
                    "document_hash": chunk.metadata["document_hash"],
                    "chunks": []
                }
            
            # Append the chunk ID if it's not already tracked
            if chunk_id not in index_store[doc_id]["chunks"]:
                index_store[doc_id]["chunks"].append(chunk_id)
                
        return index_store
