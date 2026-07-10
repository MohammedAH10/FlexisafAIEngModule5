import os
import json
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Paths for saving artifacts
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'ag_news.index')
DOCS_PATH = os.path.join(DATA_DIR, 'documents.json')

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading AG News dataset (1,000 records)...")
    dataset = load_dataset("ag_news", split="train[:1000]")
    documents = dataset['text']

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Saving index and documents to disk...")
    # Save the FAISS index
    faiss.write_index(index, INDEX_PATH)
    
    # Save the raw documents so we can retrieve the text later
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(documents, f)

    print(f"Success! Indexed {index.ntotal} documents.")

if __name__ == "__main__":
    main()
