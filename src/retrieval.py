import numpy as np
import faiss
import logging
from typing import List, Dict
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaissRetriever:
    """
    A retriever class using FAISS for efficient similarity search.

    Note: This implementation does not support document deletion. Adding this feature would require refactoring of the ID management system and the FAISS index handling.
    """
    
    def __init__(self, dimension: int):
        # This implementation assumes documents are never deleted.
        # Changing this would require refactoring ID management and index handling.
        self.index = faiss.IndexFlatL2(dimension)
        self.document_map: Dict[int, Dict] = {}
        logging.info(f"Initialized FAISS index with dimension {dimension}")

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the index.

        Note: This method only supports adding documents. Deletion is not supported in the current implementation.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        start_id = len(self.document_map)
        self.index.add(embeddings)

        for i, doc in enumerate(documents):
            self.document_map[start_id + i] = doc

        logging.info(f"Added {len(documents)} documents to the index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx in self.document_map:
                doc = self.document_map[idx].copy()
                doc['score'] = float(distances[0][list(indices[0]).index(idx)])
                results.append(doc)
        
        logging.info(f"Retrieved {len(results)} documents for the query")
        return results
    
    def save_state(self, index_path: str, map_path: str):
        self.save_index(index_path)
        with open(map_path, 'wb') as f:
            pickle.dump(self.document_map, f)
        logging.info(f"Saved FAISS index to {index_path} and document map to {map_path}")

    def load_state(self, index_path: str, map_path: str):
        self.load_index(index_path)
        with open(map_path, 'rb') as f:
            self.document_map = pickle.load(f)
        logging.info(f"Loaded FAISS index from {index_path} and document map from {map_path}")
    
    def save_index(self, file_path: str):
        faiss.write_index(self.index, file_path)
        logging.info(f"Saved FAISS index to {file_path}")

    def load_index(self, file_path: str):
        self.index = faiss.read_index(file_path)
        logging.info(f"Loaded FAISS index from {file_path}")

# Example usage
if __name__ == "__main__":
    # Example embeddings and documents
    embeddings = np.random.rand(10, 128).astype('float32')
    documents = [{"id": i, "content": f"Document {i}"} for i in range(10)]

    retriever = FaissRetriever(dimension=128)
    retriever.add_documents(documents, embeddings)

    # Example query
    query_embedding = np.random.rand(1, 128).astype('float32')
    results = retriever.search(query_embedding, k=3)

    print("Search Results:")
    for result in results:
        print(f"Document ID: {result['id']}, Score: {result['score']:.4f}")

    # Save and load index
    retriever.save_index("faiss_index.bin")
    retriever.load_index("faiss_index.bin")
