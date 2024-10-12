import os
from typing import List, Dict
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pdf(file_path: str) -> str:
    """
    Load text content from a PDF file.

    Args:
    file_path (str): Path to the PDF file.

    Returns:
    str: Text content of the PDF.
    """
    logging.info(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    total_pages = len(reader.pages)
    update_interval = max(1, total_pages // 10)

    for i, page in enumerate(reader.pages):
        if i % update_interval == 0 or i == total_pages - 1:
            progress = min(100, round((i + 1) / total_pages * 100))
            logging.info(f"  Processing page {i+1}/{total_pages} ({progress}% complete)")
        text += page.extract_text() + "\n"
    logging.info(f"Finished loading PDF: {file_path}")
    return text

def load_data(directory: str) -> List[Dict]:
    """
    Load PDF files from a directory.

    Args:
    directory (str): Path to the directory containing the PDF files.

    Returns:
    List[Dict]: A list of dictionaries, each representing a document.
    """
    logging.info(f"Loading data from directory: {directory}")
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    logging.info(f"Found {len(pdf_files)} PDF files")

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        logging.info(f"Processing file: {filename}")
        text = load_pdf(file_path)
        documents.append({
            "filename": filename,
            "text": text
        })
    logging.info(f"Finished loading {len(documents)} documents")
    return documents

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by converting to lowercase and removing extra whitespace.

    Args:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    logging.info("Preprocessing text...")
    text = text.lower()
    text = ' '.join(text.split())
    logging.info("Finished preprocessing text")
    return text

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:1+chunk_size] for i in range(0, len(text), chunk_size)]

def process_documents(documents: List[Dict]) -> List[Dict]:
    """
    Process a list of documents by preprocessing their text content.

    Args:
    documents (List[Dict]): A list of document dictionaries.

    Returns:
    List[Dict]: A list of processed document dictionaries.
    """
    logging.info(f"Processing {len(documents)} documents")
    processed_documents = []
    for i, doc in enumerate(documents):
        logging.info(f"Processing document {i+1}/{len(documents)}")
        processed_doc = doc.copy()
        if 'text' in processed_doc:
            full_text = preprocess_text(processed_doc['text'])
            chunks = chunk_text(full_text)
            processed_doc['chunks'] = chunks
            processed_doc['text'] = full_text
            logging.info(f"Document {i+1} chunked into {len(chunks)} chunks") 
        processed_documents.append(processed_doc)
    logging.info("Finished processing documents")
    return processed_documents

def main():
    logging.info("Starting main function")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '..', 'data', 'example_raw')
    logging.info(f"Raw data directory: {raw_data_dir}")

    if not os.path.exists(raw_data_dir):
        logging.error(f"Error: Directory not found: {raw_data_dir}")
        return

    documents = load_data(raw_data_dir)
    processed_documents = process_documents(documents)
    logging.info(f"Processed {len(processed_documents)} documents.")

    # Print a sample of the first document's text (first 500 characters)
    if processed_documents:
        logging.info("\nSample of first document:")
        logging.info(processed_documents[0]['text'][:500])
    else:
        logging.info("No documents were processed. Check if there are PDF files in the data/raw directory.")

    logging.info("Main function completed")

if __name__ == "__main__":
    main()