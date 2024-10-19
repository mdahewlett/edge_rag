import os
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Defining the structure of the ToC
class Subsection(BaseModel):
    title: str
    content: Optional[str]

class Section(BaseModel):
    title: str
    subsections: List[Subsection]

class TableOfContents(BaseModel):
    sections: List[Section]


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

def load_pdf(file_path: str) -> str:
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

def extract_toc(text: str) -> TableOfContents:
    prompt = f"""
    Given the following text from a manual, generate a detailed table of contents.
    Format the output as a JSON object with the following structure:
    {{
        "sections": [
            {{
                "title": "Section Title",
                "subsections": [
                    {{
                        "title": "Subsection Title",
                        "content": "Brief description or summary of the subsection"
                    }}
                ]
            }}
        ]
    }}

    Text:
    {text[:4000]}
    """
    try:
        response = client.beta.chat.completions.parse(
            model='gpt-4o-mini-2024-07-18',
            messages=[
                {"role":"system", "content": "You are an assistant that formats outputs as structured JSON."},
                {"role":"user", "content": prompt}
            ],
            response_format=TableOfContents,
        )

        print(response.choices[0].message.content)
        toc = response.choices[0].message.parsed
        return toc
    except ValidationError as ve:
        logging.info(f"Validation Error: {str(ve)}")
        return None
    except Exception as e:
        logging.info(f"An error occured: {str(e)}")
        return None

def split_content(text: str, toc: Dict[str, Any]) -> Dict[str, Any]:
    def find_content(section: Dict[str, Any], remaining_text:str) -> tuple[Dict[str, Any], str]:
        start = remaining_text.lower().find(section['title'].lower())
        if start == -1:
            # If exact title not found, find a close match
            words = section['title'].lower().split()
            for i in range(len(words), 0, -1):
                partial_title = ' '.join(words[:i])
                start = remaining_text.lower().find(partial_title)
                if start != -1:
                    break
        
        if start == -1:
            logging.warning(f"Could not find section: {section['title']}")
            return section, remaining_text
        
        remaining_text = remaining_text[start:]
        end = len(remaining_text)

        if 'subsections' in section:
            for subsection in section['subsections']:
                subsection, remaining_text = find_content(subsection, remaining_text)
                end = min(end, remaining_text.find(subsection['title']))

        section['content'] = remaining_text[:end].strip()
        return section, remaining_text[end:]
    
    remaining_text = text
    for section in toc['sections']:
        section, remaining_text = find_content(section, remaining_text)

    return toc

def process_documents(documents: List[Dict]) -> List[Dict]:
    logging.info(f"Processing {len(documents)} documents")
    processed_documents = []
    for i, doc in enumerate(documents):
        logging.info(f"Processing document {i+1}/{len(documents)}")
        processed_doc = doc.copy()
        toc = extract_toc(processed_doc['text'])
        structured_doc = split_content(processed_doc['text'], toc)
        processed_doc['structured_content'] = structured_doc
        del processed_doc['text'] # Remove original text to save memory
        processed_documents.append(processed_doc)
    logging.info("Finished processing documents")
    return processed_documents

def save_processed_documents(processed_documents: List[Dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for doc in processed_documents:
        filename = os.path.splitext(doc['filename'])[0] + '.json'
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(doc['structured_content'], f, indent=2)
        logging.info(f"Saved processed document: {output_path}")

def main():
    logging.info("Starting main function")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '..', 'data', 'example_raw')
    processed_data_dir = os.path.join(current_dir, '..', 'data', 'processed')
    logging.info(f"Raw data directory: {raw_data_dir}")
    logging.info(f"Proessed data directory: {processed_data_dir}")

    if not os.path.exists(raw_data_dir):
        logging.error(f"Error: Directory not found: {raw_data_dir}")
        return

    documents = load_data(raw_data_dir)
    processed_documents = process_documents(documents)
    save_processed_documents(processed_documents, processed_data_dir)
    
    logging.info(f"Main function completed. Processed {len(processed_documents)} documents.")

if __name__ == "__main__":
    main()