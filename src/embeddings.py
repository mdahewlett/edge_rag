import os
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentEmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Successfully initialized model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise 
    
    def generate_embeddings(self, documents):
        logging.info(f"Generating embeddings for {len(documents)} documents.")
        try:
            embeddings = self.model.encode(documents)
            logging.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings, output_dir):
        logging.info(f"Searching for '{output_dir}' directory.")
        start_time = time.time()

        try:
            if not os.path.exists(output_dir):
                logging.info(f"'{output_dir}' directory not found. Creating '{output_dir}'.")
                os.makedirs(output_dir)

            logging.info(f"Saving embeddings to '{output_dir}' directory.")
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

    def load_embeddings(self, input_dir):
        logging.info(f"Loading embeddings from '{input_dir}'.")
        start_time = time.time()

        if not os.path.exists(input_dir):
            logging.error(f"Directory '{input_dir}' not found.")
            raise FileNotFoundError(f"Directory '{input_dir}' not found.")
        
        try:
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
        "Ant this is the third one.",
        "Is this the last document?",
    ]

    generator = DocumentEmbeddingGenerator()
    embeddings = generator.generate_embeddings(documents)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Shape of first embedding: {embeddings[0].shape}")

    # Save embeddings
    generator.save_embeddings(embeddings, "embeddings_output")

    # Load embeddings
    loaded_embeddings = generator.load_embeddings("embeddings_output")
    print(f"Loaded {len(loaded_embeddings)} embeddings.")
    print(f"Shape of first loaded embedding: {loaded_embeddings[0].shape}")