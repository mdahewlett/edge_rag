import os
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Successfully initialized model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise 
    
    def generate_document_embeddings(self, documents):
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
        logging.info(f"Generating embedding for query: {query[:50]}...")
        try:
            embedding = self.model.encode([query])[0]
            logging.info(f"Successfully generate query embedding. Shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logging.error(f"Error generating query embedding: {str(e)}")
            raise

    def save_document_embeddings(self, embeddings, output_dir):
        logging.info(f"Saving embeddings to '{output_dir}' directory.")
        start_time = time.time()

        try:
            if not os.path.exists(output_dir):
                logging.info(f"'{output_dir}' directory not found. Creating '{output_dir}'.")
                os.makedirs(output_dir)

            for i, embedding in enumerate(embeddings):
                if i % 10 == 0:
                    logging.info(f"Saving embedding {i+1}/{len(embeddings)} ({(i+1)/len(embeddings)*100:.2f}%).")
                np.save(os.path.join(output_dir, f'embedding_{i}.npy'), embedding)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Saved {len(embeddings)} embeddings in {elapsed_time:.2f} seconds.")
        except IOError as e:
            logging.error(f"IO error occured while saving embeddings: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occured while saving embeddings: {str(e)}")
            raise

    def load_document_embeddings(self, input_dir):
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
                    embedding = np.load(os.path.join(input_dir, filename))
                    embeddings.append(embedding)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Loaded {len(embeddings)} embeddings in {elapsed_time:.2f} seconds.")
            return embeddings
        except IOError as e:
            logging.error(f"IO error occured while loading embeddings: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occured while loading embeddings: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    documents = [
        "This is the first document.",
        "Here's the second document.",
        "And this is the third one.",
        "Is this the last document?",
    ]

    generator = EmbeddingGenerator()

    # Generate document embeddings
    doc_embeddings = generator.generate_document_embeddings(documents)
    logging.info(f"Shape of first embedding: {doc_embeddings[0].shape}")

    # Generate query embedding
    query = "What is the capital of France?"
    query_embedding = generator.generate_query_embedding(query)

    # Save embeddings
    generator.save_document_embeddings(doc_embeddings, "document_embeddings_output")

    # Load embeddings
    loaded_embeddings = generator.load_document_embeddings("document_embeddings_output")
    print(f"Shape of first loaded document embedding: {loaded_embeddings[0].shape}")