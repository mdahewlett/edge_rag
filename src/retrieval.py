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
    A retriever class using FAISS for efficient similarity search of document chunks.

    Note: This implementation does not support document deletion. Adding this feature would require refactoring of the ID management system and the FAISS index handling.
    """
    
    def __init__(self, dimension: int):
        # This implementation assumes documents are never deleted.
        # Changing this would require refactoring ID management and index handling.
        self.index = faiss.IndexFlatL2(dimension)
        self.embedding_map: List[Dict] = []
        logging.info(f"Initialized FAISS index with dimension {dimension}")

    def add_documents(self, embedding_map: List[Dict]):
        """
        Add document chunks and their embeddings to the index.

        Note: This method only supports adding documents. Deletion is not supported in the current implementation.
        """
        logging.info(f"Adding {len(embedding_map)} chunk embeddings to the index.")
        start_time = time.time()
        
        try:
            embeddings = []
            for i, item in enumerate(embedding_map):
                if i % 100 == 0:
                    logging.info(f"Processing {i+1}/{len(embedding_map)} chunk embeddings...")
                embeddings.append(item['embedding'])

            embeddings_array = np.array(embeddings)
            self.index.add(embeddings_array)
            self.embedding_map = embedding_map

            elapsed_time = time.time() - start_time
            logging.info(f"Added {len(embedding_map)} chunk embeddings to the index in {elapsed_time:.2F} seconds.")
        except Exception as e:
            logging.error(f"Unexpected error occured while adding chunk embeddings: {str(e)}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        logging.info(f"Searching for {k} closest results")
        start_time = time.time()

        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                chunk_info = self.embedding_map[idx].copy()
                chunk_info['score'] = float(distance)
                results.append(chunk_info)

            elapsed_time = time.time() - start_time
            logging.info(f"Retrieved {len(results)} chunks for the query in {elapsed_time:.2F} seconds.")
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
                'embedding_map': self.embedding_map
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
            self.embedding_map = state['embedding_map']
            elapsed_time = time.time() - start_time
            logging.info(f"Loaded state from {file_path} in {elapsed_time:.2F} seconds.")
        except Exception as e:
            logging.error(f"Error occured while loading state: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Example embeddings and documents with chunks
    num_docs = 100
    chunks_per_doc = 10
    embedding_dim = 128

    embedding_map = []
    for doc_index in range(num_docs):
        for chunk_index in range(chunks_per_doc):
            embedding_map.append({
                'doc_index': doc_index,
                'chunk_index': chunk_index,
                'embedding': np.random.rand(embedding_dim).astype('float32')
            })

    retriever = FaissRetriever(dimension=embedding_dim)
    retriever.add_documents(embedding_map)

    # Example query
    num_results = 10
    query_embedding = np.random.rand(1, embedding_dim).astype('float32')
    results = retriever.search(query_embedding, k=num_results)

    logging.info("Initial Search Results:")
    for result in results:
        logging.info(f"Document: {result['doc_index']}, Chunk: {result['chunk_index']}, Score: {result['score']:.4f}")

    # Save state
    retriever.save_state("faiss_state.bin")

    # Create a new retriever and load the saved state
    new_retriever = FaissRetriever(dimension=embedding_dim)
    new_retriever.load_state("faiss_state.bin")
    
    # Verify loaded state with a search
    new_results = new_retriever.search(query_embedding, k=num_results)
    logging.info("Search results after loading state:")
    for result in new_results:
        logging.info(f"Document: {result['doc_index']}, Chunk: {result['chunk_index']}, Score: {result['score']:.4f}")