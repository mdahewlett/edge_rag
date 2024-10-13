import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import time
import logging
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
            chunk_index = 0
            for doc_index, doc in enumerate(documents):
                for _ in doc['chunks']:
                    embedding_map.append({
                        'doc_index': doc_index,
                        'chunk_index': chunk_index,
                        'embedding': embeddings[chunk_index]
                    })
                    chunk_index += 1
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

    def save_document_embeddings(self, embeddings: List[np.ndarray], output_dir: str) -> None:
        """
        Save chunk embeddings, without document or chunk indices, to a directory.

        Args:
            embeddings (List[np.ndarray]): A list of vector embeddings, each as a numpy array.
            output_dir (str): The path to the directory where embeddings will be saved.
        
        Returns:
            None

        Raises:
            IOError: If there's an issue creating the directory or writing files.
            Exception: For any other saving issues.

        """
        logging.info(f"Saving embeddings to '{output_dir}' directory.")
        start_time = time.time()

        try:
            if not os.path.exists(output_dir):
                logging.info(f"'{output_dir}' directory not found. Creating '{output_dir}'.")
                os.makedirs(output_dir)

            for i, embedding in enumerate(embeddings):
                if i % 10 == 0:
                    logging.info(f"Saving embedding {i+1}/{len(embeddings)} ({(i+1)/len(embeddings)*100:.2f}%).")
                np.save(os.path.join(output_dir, f'embedding_{i:04d}.npy'), embedding)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Saved {len(embeddings)} embeddings in {elapsed_time:.2f} seconds.")
        except IOError as e:
            logging.error(f"IO error occurred while saving embeddings: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred while saving embeddings: {str(e)}")
            raise

    def load_document_embeddings(self, input_dir: str) -> List[np.ndarray]:
        """
        Load embeddings from a directory.

        Args:
            input_dir (str): The path to the directory containing embeddings saved as .npy files.
        
        Returns:
            List[np.ndarray]: A list of embeddings, each one a numpy array representing a chunk of text.

        Raises:
            FileNotFoundError: If the directory does not exist.
            IOError: If there's an issue reading the embedding files.
            Exception: For any other loading issues.
        """
        logging.info(f"Loading embeddings from '{input_dir}'.")
        start_time = time.time()

        try:
            if not os.path.exists(input_dir):
                logging.error(f"Directory '{input_dir}' not found.")
                raise FileNotFoundError(f"Directory '{input_dir}' not found.")
            
            embeddings = []
            for i, filename in enumerate(os.listdir(input_dir)):
                if filename.endswith('.npy'):
                    if i % 10 == 0:
                        logging.info(f"Loading embedding {i+1} ({filename})")
                    embedding = np.load(os.path.join(input_dir, filename), allow_pickle=True)
                    
                    logging.info(f"Loaded {filename}:")
                    logging.info(f"  Type: {type(embedding)}")
                    logging.info(f"  Shape: {embedding.shape if isinstance(embedding, np.ndarray) else 'Not a numpy array'}")
                    logging.info(f"  Size: {embedding.size if isinstance(embedding, np.ndarray) else 'Not a numpy array'}")
                
                    if isinstance(embedding, np.ndarray) and embedding.size > 0:
                        embeddings.append(embedding)
                        logging.info(f"  Status: Added to embeddings list")
                    else:
                        logging.warning(f"  Status: Skipped (not a valid numpy array or empty)")
        
            elapsed_time = time.time() - start_time
            logging.info(f"Loaded {len(embeddings)} embeddings in {elapsed_time:.2f} seconds.")
            return embeddings
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
    doc_embeddings = generator.generate_document_embeddings(documents)
    logging.info(f"Shape of first embedding: {doc_embeddings[0]['embedding'].shape}")
    for i, emb in enumerate(doc_embeddings):
        logging.info(f"Original embedding {i}:")
        logging.info(f"  Shape: {emb['embedding'].shape}")
        logging.info(f"  Size: {emb['embedding'].size}")
        logging.info(f"  Mean: {emb['embedding'].mean()}")
        logging.info(f"  Standard deviation: {emb['embedding'].std()}")

    # Generate query embedding
    query = "What is the capital of France?"
    query_embedding = generator.generate_query_embedding(query)
    logging.info(f"Query embedding shape: {query_embedding.shape}")
    logging.info(f"Query embedding size: {query_embedding.size}")
    logging.info(f"Query embedding mean: {query_embedding.mean()}")
    logging.info(f"Query embedding standard deviation: {query_embedding.std()}")

    # Save embeddings
    generator.save_document_embeddings([emb['embedding'] for emb in doc_embeddings], "document_embeddings_output")

    # Load embeddings
    loaded_embeddings = generator.load_document_embeddings("document_embeddings_output")
    print(f"Shape of first loaded document embedding: {loaded_embeddings[0].shape}")