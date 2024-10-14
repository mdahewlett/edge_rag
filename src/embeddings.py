import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import time
import logging
import pickle
import glob
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    """
    A text embedding class for document retrieval and query matching.

    This class processes documents and queries, generating vector embeddings using a pre-trained Sentence Transformer model. It supports operations for embedding generations, saving, and loading.
    
    Attributes:
        model (SentenceTransformer): The sentence transformer model used for generating embeddings.
    
    Methods:
        generate_document_embeddings(documents: List[Dict]) -> List[Dict]: 
            Generate embeddings for all chunks in the given documents.
            Returns a list of dictionaries containing document index, chunk index, and chunk embedding.
        
        generate_query_embedding(query: str) -> np.ndarray:
            Generate an embedding for a given query string.
            Returns a numpy array representing the query embedding.

        save_document_embeddings(embeddings: List[np.ndarray], output_dir: str): 
            Save document embeddings to a directory as individual .npy files.

        load_document_embeddings(input_str: str) -> List[np.ndarray]: 
            Load document embeddings from a directory.
            Returns a list of numpy arrays representing the embeddings.
    
    Raises:
        ValueError: If invalid input is provided to any method.
        IOError: If there are issues with file operations during saving or loading.
        FileNotFoundError: If the specified directory for loading embeddings doesn't exist.

        
    Example:
        documents = [
            {"chunks": ["This is the first document."]},
            {"chunks": ["Here's the second document.]}
        ]
        generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        embedding_map = generator.generate_document_embeddings(documents)
        query_embedding = generator.generate_query_embeddings("Sample query")
        generator.save_document_embeddings([emb['embedding'] for emb in embedding_map], '/path/to/save')
        loaded_embeddings = generator.load_document_embeddings('/path/to/save')

    Note:
        The class uses logging to provide information about its operations. Ensure proper logging configuration for detailed insights into the embedding process.
    """
    def __init__(self, model_name: Optional[str] = 'all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingGenerator with a SentenceTransformer model.

        Args:
            model_name (Optional[str]): The name of the SentenceTransformer model to use. 
                Defaults to 'all-MiniLM-L6-v2' if not specified.
        
        Returns:
            None
        
        Raises: 
            Exception: For any initialization errors.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Successfully initialized model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise 
    
    def generate_document_embeddings(self, documents: List[Dict]) -> List[Dict]:
        """
        Generates an embedding for each chunk in each document provided.

        Args:
            documents (List[Dict]): A list of document dictionaries. Each dictionary must contain a 'chunks' key with a list of text chunks as its value.

        Returns:
            List[Dict]: A list of dictionaries, one for each chunk, containing:
                - 'doc_index' (int): Index of the document containing the chunk.
                - 'chunk_index' (int): Index of the chunk within the flattened list of all chunks.
                - 'embedding' (numpy.ndarray): Vetcor embedding of the chunk.

        Raises:
            Exception: For any embedding generation issues.

        """
        logging.info(f"Generating embeddings for {len(documents)} documents.")
        try:
            all_chunks = [chunk for doc in documents for chunk in doc['chunks']]
            embeddings = self.model.encode(all_chunks)
            embedding_map= []
            embedding_index = 0
            for doc_index, doc in enumerate(documents):
                for chunk_index in range(len(doc['chunks'])):
                    embedding_map.append({
                        'doc_index': doc_index,
                        'chunk_index': chunk_index,
                        'embedding': embeddings[embedding_index]
                    })
                    embedding_index += 1
            logging.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embedding_map
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a provided query.
        
        Args:
            query (str): The user's query string.

        Returns:
            np.ndarray: A vector embedding of the query.

        Raises:
            Exception: For any embedding generation issues.
        """
        logging.info(f"Generating embedding for query: {query[:50]}...")
        try:
            embedding = self.model.encode([query])[0]
            logging.info(f"Successfully generate query embedding. Shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logging.error(f"Error generating query embedding: {str(e)}")
            raise

    def save_document_embeddings(self, embedding_map: List[Dict], output_dir: str, filename: str = "embedding_map.pkl") -> None:
        """
        Save embedding map to a directory.

        Args:
            embedding_map (List[Dict]): A list of dictionaries, each containing 'doc_index', 'chunk_index', and 'embedding'.
            output_dir (str): The path to the directory where embeddings will be saved.
            filename (str): The name of the file to save the embedding map. Defaults to "embedding_map.pkl".
        
        Returns:
            None

        Raises:
            IOError: If there's an issue creating the directory or writing files.
            Exception: For any other saving issues.

        """
        logging.info(f"Saving embedding map to '{output_dir}' directory.")
        start_time = time.time()

        try:
            if not os.path.exists(output_dir):
                logging.info(f"'{output_dir}' directory not found. Creating '{output_dir}'.")
                os.makedirs(output_dir)
            
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            with open(os.path.join(output_dir, filename), 'wb') as f:
                pickle.dump(embedding_map, f)

            elapsed_time = time.time() - start_time
            logging.info(f"Saved embedding map with {len(embedding_map)} items in {elapsed_time:.2f} seconds.")
        except IOError as e:
            logging.error(f"IO error occurred while saving embedding map: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred while saving embedding map: {str(e)}")
            raise

    def load_document_embeddings(self, input_dir: str, filename: Optional[str] = None) -> List[Dict]:
        """
        Load embedding map from a directory.

        Args:
            input_dir (str): The path to the directory containing embeddings saved as .npy files.
            filename (str, optional): The specific filename to load. If None, loads the first .pkl file found.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'doc_index', 'chunk_index', and 'embedding'.

        Raises:
            FileNotFoundError: If the directory, specified file, or any .pkl file does not exist.
            IOError: If there's an issue reading the embedding map file.
            Exception: For any other loading issues.
        """
        logging.info(f"Loading embedding map from '{input_dir}'.")
        start_time = time.time()

        try:
            if not os.path.exists(input_dir):
                logging.error(f"Directory '{input_dir}' not found.")
                raise FileNotFoundError(f"Directory '{input_dir}' not found.")
            
            if filename:
                embedding_map_path = os.path.join(input_dir, filename)
                if not os.path.exists(embedding_map_path):
                    logging.ERROR(f"Specified file '{filename}' not found in '{input_dir}'.")
                    raise FileNotFoundError(f"File '{filename}' not found in '{input_dir}'.")
            else:
                pkl_files = glob.glob(os.path.join(input_dir, '*.pkl'))
                if not pkl_files:
                    logging.error(f"No .pkl files found in '{input_dir}'.")
                    raise FileNotFoundError(f"No .pkl files found in '{input_dir}'.")
                embedding_map_path = pkl_files[0]
                
            with open(embedding_map_path, 'rb') as f:
                embedding_map = pickle.load(f)

            elapsed_time = time.time() - start_time
            logging.info(f"Loaded embedding map with {len(embedding_map)} items from {os.path.basename(embedding_map_path)} in {elapsed_time:.2f} seconds.")
            return embedding_map
        except IOError as e:
            logging.error(f"IO error occurred while loading embeddings: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred while loading embeddings: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    documents = [
        {'chunks': ["This is the first chunk of document 1.", "This is the second chunk of document 1."]},
        {'chunks': ["Here's the first chunk of document 2.", "And the second chunk of document 2.", "A third chunk for document 2."]},
        {'chunks': ["This is the only chunk for document 3."]},
        {'chunks': ["First chunk of the last document.", "Is this the last chunk?", "No, this is the last chunk."]}
    ]
    generator = EmbeddingGenerator()

    # Generate document embeddings
    doc_embedding_map = generator.generate_document_embeddings(documents)
    logging.info(f"Generated embedding map with {len(doc_embedding_map)} items")

    output_dir = "document_embeddings_output"

    # Generate query embedding
    query = "What is the capital of France?"
    query_embedding = generator.generate_query_embedding(query)

    # Save embedding map with default filename
    generator.save_document_embeddings(doc_embedding_map, output_dir)

    # Save embedding map with custom filename
    custom_filename = "my_embeddings.pkl"
    generator.save_document_embeddings(doc_embedding_map, output_dir, custom_filename)

    # Load embedding map without specifying filename (loads first .pkl file)
    loaded_embedding_map = generator.load_document_embeddings(output_dir)
    logging.info(f"Loaded embedding map (default) with {len(loaded_embedding_map)} items")

    # Load embedding map with specific filename
    specific_loaded_map = generator.load_document_embeddings(output_dir, custom_filename)
    logging.info(f"Loaded specific embedding map with {len(specific_loaded_map)} items")