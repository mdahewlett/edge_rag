import os
import logging
from data_processing import load_data, process_documents
from embeddings import EmbeddingGenerator
from retrieval import FaissRetriever
from generation import Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self, data_dir, embedding_model='all-MiniLM-L6-v2', generator_model='google/flan-t5-small'):
        self.data_dir = data_dir
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.retriever = None
        self.generator = Generator(model_path=generator_model)

    def load_and_process_documents(self):
        logging.info("Loading and processing documents...")
        documents = load_data(self.data_dir)
        return process_documents(documents)
    
    def create_embeddings(self, documents):
        logging.info("Creating embeddings...")
        texts = [doc['text'] for doc in documents]
        return self.embedding_generator.generate_document_embeddings(texts)
    
    def build_index(self, documents, embeddings):
        logging.info("Buiklding index...")
        self.retriever = FaissRetriever(dimension=embeddings.shape[1])
        self.retriever.add_documents(documents, embeddings)
    
    def process_query(self, query, k=3):
        logging.info(f"Processing query: {query}")
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        retrieved_docs = self.retriever.search(query_embedding, k=k)
        context = [doc['text'] for doc in retrieved_docs]
        response = self.generator.generate(query, context)
        return response, retrieved_docs

    def run(self):
        documents = self.load_and_process_documents()
        embeddings = self.create_embeddings(documents)
        self.build_index(documents, embeddings)

        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            response, retrieved_docs = self.process_query(query)
            print(f"\nResponse: {response}\n")
            print("Retrieved documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"{i}. {doc['filename']} (score: {doc['score']:.4f})")
            print()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'example_raw')

    pipeline = RAGPipeline(data_dir)
    pipeline.run()