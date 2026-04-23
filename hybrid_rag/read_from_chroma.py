
from vectordb_client import vectorstore

doc_id = "3f96f3f8421994405ca8b3981037c4750bed14e8b02c3bbcfe8484fd17f87bbd"

results = vectorstore._collection.get(
   where={"document_id": {"$eq": doc_id}},
    

    include=["metadatas", "documents"],
)

sample = vectorstore._collection.peek(limit=1)
if sample['metadatas']:
    fields = sample['metadatas'][0].keys()
    print(f"Filterable fields: {list(fields)}")

# 2. Extract and print the details
ids = results['ids']
docs = results['documents']
metas = results['metadatas']

for i in range(len(ids)):
    print(f"--- Entry {i+1} ---")
    print(f"Chunk ID: {ids[i]}")
    # 'source' is the standard field for the original document path
    print(f"Document: {metas[i].get('source', 'No source found')}")
    print(f"Content: {docs[i][:100]}...") # Showing first 100 chars
    print("\n")