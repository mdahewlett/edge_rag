import os
import logging
from data_processing import load_data, process_documents
from embeddings import EmbeddingGenerator
from retrieval import FaissRetriever
from generation_openai import OpenAIGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self, data_dir, embedding_model='all-MiniLM-L6-v2', generator_model='facebook/bart-large-cnn'):
        self.data_dir = data_dir
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.retriever = None
        self.generator = OpenAIGenerator(model_path=generator_model)
        self.documents = None
        self.processed_documents = None

    def load_and_process_documents(self):
        logging.info("Loading and processing documents...")
        self.documents = load_data(self.data_dir)
        self.processed_documents = process_documents(self.documents)
        return self.processed_documents
    
    def create_embeddings(self, processed_documents):
        logging.info("Creating embeddings...")
        return self.embedding_generator.generate_document_embeddings(processed_documents)
    
    def build_index(self, embedding_map):
        logging.info("Building index...")
        self.retriever = FaissRetriever(dimension=len(embedding_map[0]['embedding']))
        self.retriever.add_documents(embedding_map)
    
    def process_query(self, query, k=3):
        logging.info(f"Processing query: {query}")
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        retrieved_chunks = self.retriever.search(query_embedding, k=k)
        
        context = []
        for chunk in retrieved_chunks:
            doc = self.processed_documents[chunk['doc_index']]
            context.append(doc['chunks'][chunk['chunk_index']])

        context = " ".join(context)
        response = self.generator.generate(query, context)
        return response, retrieved_chunks

    def run(self):
        documents = self.load_and_process_documents()
        embedding_map = self.create_embeddings(documents)
        self.build_index(embedding_map)

        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            response, retrieved_chunks = self.process_query(query)
            logging.info(f"\nResponse:\n{response}\n")
            logging.info("Retrieved documents:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                doc = documents[chunk['doc_index']]
                logging.info(f"{i}. {doc['filename']} (score: {chunk['score']:.4f})\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'example_raw')

    pipeline = RAGPipeline(data_dir)
    pipeline.run()