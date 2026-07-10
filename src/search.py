import os
import json
import argparse
import faiss
from sentence_transformers import SentenceTransformer

# Paths to load artifacts
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'ag_news.index')
DOCS_PATH = os.path.join(DATA_DIR, 'documents.json')

def search(query, top_k):
    # 1. Verify artifacts exist
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("Error: Index or documents not found. Run 'python src/build_index.py' first.")
        return

    # 2. Load FAISS index and documents
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # 3. Load the model (must be the same model used for indexing)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Embed the search query
    query_embedding = model.encode([query]).astype('float32')

    # 5. Execute search
    distances, indices = index.search(query_embedding, top_k)

    # 6. Print Results
    print(f"\nSearch Query: '{query}'")
    print("-" * 50)
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        print(f"[{i+1}] Distance: {dist:.4f}")
        print(f"Text: {documents[idx]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search using FAISS")
    parser.add_argument("query", type=str, help="The search query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    
    args = parser.parse_args()
    search(args.query, args.top_k)
