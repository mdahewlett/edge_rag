import os
import numpy as np
import faiss
from typing import List, Dict
import pickle
import logging
import time

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
        logging.info(f"Adding {len(documents)} documents to the index.")
        start_time = time.time()
        
        try:
            if len(documents) != embeddings.shape[0]:
                raise ValueError("Number of documents must match number of embeddings")
            
            start_id = len(self.document_map)
            self.index.add(embeddings)

            for i, doc in enumerate(documents):
                if i % 100 == 0:
                    logging.info(f"Adding {i+1}/{len(documents)} documents...")
                self.document_map[start_id + i] = doc

            elapsed_time = time.time() - start_time
            logging.info(f"Added {len(documents)} documents to the index in {elapsed_time:.2F} seconds.")
        except ValueError as e:
            logging.error(f"Value Error occured while adding documents: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occured while adding documents: {str(e)}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        logging.info(f"Searching for {k} closest results")
        start_time = time.time()

        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for idx in indices[0]:
                if idx in self.document_map:
                    doc = self.document_map[idx].copy()
                    doc['score'] = float(distances[0][list(indices[0]).index(idx)])
                    results.append(doc)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Retrieved {len(results)} documents for the query in {elapsed_time:.2F} seconds.")
            return results
        except Exception as e:
            logging.error(f"Error occured during search: {str(e)}")
            raise
    
    def save_state(self, file_path: str):
        logging.info(f"Saving state to '{file_path}'.")
        start_time = time.time()
        try:
            state = {
                'index': faiss.serialize_index(self.index),
                'document_map': self.document_map
            }
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            elapsed_time = time.time() - start_time
            logging.info(f"Saved state to {file_path} in {elapsed_time:.2F} seconds.")
        except Exception as e:
            logging.error(f"Error occured while saving state: {str(e)}")
            raise
        
    def load_state(self, file_path: str):
        logging.info(f"Loading state from '{file_path}'.")
        start_time = time.time()
        try:
            if not os.path.isfile(file_path):
                logging.error(f"State file '{file_path}' not found or is not a file.")
                raise FileNotFoundError(f"State file '{file_path}' not found or is not a file.")
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            self.index = faiss.deserialize_index(state['index'])
            self.document_map = state['document_map']
            elapsed_time = time.time() - start_time
            logging.info(f"Loaded state from {file_path} in {elapsed_time:.2F} seconds.")
        except Exception as e:
            logging.error(f"Error occured while loading state: {str(e)}")
            raise

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

    logging.info("Initial Search Results:")
    for result in results:
        logging.info(f"Document ID: {result['id']}, Score: {result['score']:.4f}")

    # Save state
    retriever.save_state("faiss_state.bin")

    # Create a new retriever and load the saved state
    new_retriever = FaissRetriever(dimension=128)
    new_retriever.load_state("faiss_state.bin")
    
    # Verify loaded state with a search
    new_results = new_retriever.search(query_embedding, k=3)
    logging.info("Search results after loading state:")
    for result in new_results:
        logging.info(f"Document ID: {result['id']}, Score: {result['score']:.4f}")