import os
from typing import List, Dict
from pypdf import PdfReader

def load_pdf(file_path: str) -> str:
    """
    Load text content from a PDF file.

    Args:
    file_path (str): Path to the PDF file.

    Returns:
    str: Text content of the PDF.
    """
    print(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    total_pages = len(reader.pages)
    update_interval = max(1, total_pages // 10)

    for i, page in enumerate(reader.pages):
        if i % update_interval == 0 or i == total_pages - 1:
            progress = min(100, round((i + 1) / total_pages * 100))
            print(f"  Processing page {i+1}/{total_pages} ({progress}% complete)")
        text += page.extract_text() + "\n"
    print(f"Finished loading PDF: {file_path}")
    return text

def load_data(directory: str) -> List[Dict]:
    """
    Load PDF files from a directory.

    Args:
    directory (str): Path to the directory containing the PDF files.

    Returns:
    List[Dict]: A list of dictionaries, each representing a document.
    """
    print(f"Loading data from directory: {directory}")
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {filename}")
        text = load_pdf(file_path)
        documents.append({
            "filename": filename,
            "text": text
        })
    print(f"Finished loading {len(documents)} documents")
    return documents

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by converting to lowercase and removing extra whitespace.

    Args:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    print("Preprocessing text...")
    text = text.lower()
    text = ' '.join(text.split())
    print("Finished preprocessing text")
    return text

def process_documents(documents: List[Dict]) -> List[Dict]:
    """
    Process a list of documents by preprocessing their text content.

    Args:
    documents (List[Dict]): A list of document dictionaries.

    Returns:
    List[Dict]: A list of processed document dictionaries.
    """
    print(f"Processing {len(documents)} documents")
    processed_documents = []
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}")
        processed_doc = doc.copy()
        if 'text' in processed_doc:
            processed_doc['text'] = preprocess_text(processed_doc['text'])
        processed_documents.append(processed_doc)
    print("Finished processing documents")
    return processed_documents

def main():
    print("Starting main function")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '..', 'data', 'raw')
    print(f"Raw data directory: {raw_data_dir}")

    if not os.path.exists(raw_data_dir):
        print(f"Error: Directory not found: {raw_data_dir}")
        return

    documents = load_data(raw_data_dir)
    processed_documents = process_documents(documents)
    print(f"Processed {len(processed_documents)} documents.")

    # Print a sample of the first document's text (first 500 characters)
    if processed_documents:
        print("\nSample of first document:")
        print(processed_documents[0]['text'][:500])
    else:
        print("No documents were processed. Check if there are PDF files in the data/raw directory.")

    print("Main function completed")

if __name__ == "__main__":
    main()