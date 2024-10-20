import os
from openai import OpenAI
from PyPDF2 import PdfReader
import base64
from dotenv import load_dotenv
from pdf2image import convert_from_path
import tempfile
import logging
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional

# Defining the structure of the ToC
class Subsection(BaseModel):
    title: str
    page: str

class Section(BaseModel):
    title: str
    page: str
    subsections: List[Subsection]

class TableOfContents(BaseModel):
    sections: List[Section]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Set OpenAI key, initialize client
openai_api_key = os.getenv('OPENAI_API_KEY')
logging.info("Initializing OpenAI client")
client = OpenAI(api_key=openai_api_key)

def encode_image(image_path):
    logging.info("Encoding image...")
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_table_of_contents(pdf_path):
    reader = PdfReader(pdf_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        for page_num in range(len(reader.pages)):

            # Prevent scanning whole pdf
            if page_num > 0 and page_num % 5 == 0:
                checkin = input(f"I have checked up to page {page_num} and not found a table of contents. Enter 'quit' to exit, or anything else to proceed: ")
                if checkin.lower() == 'quit':
                    return None

            logging.info(f"Checking page {page_num+1}...")

            # Convert PDF page to image
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150, output_folder=temp_dir)
            temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
            images[0].save(temp_image_path, 'PNG')

            # Encode image
            base64_image = encode_image(temp_image_path)

            # Prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Does this page contain a table of contents? If yes, please respond with 'Yes'. If no, respond with 'No'."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]

            # API call
            logging.info("Calling API...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1
            )

            # Check response
            if response.choices[0].message.content.strip().lower() == 'yes':
                logging.info("Found it!")
                return page_num + 1
            
            logging.info("Not that one...")
            
    return None # if no ToC found

def make_toc(pdf_path, page_num):
    reader = PdfReader(pdf_path)

    with tempfile.TemporaryDirectory() as temp_dir:

        # Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150, output_folder=temp_dir)
        temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
        images[0].save(temp_image_path, 'PNG')

        # Encode image
        base64_image = encode_image(temp_image_path)

        # Prompt
        messages = [
            {
                "role":"system", 
                "content": [
                    {"type": "text", "text": "You are an assistant that formats outputs as structured JSON."}
                ]
            },    
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a table of contents from this page."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # API call
        try:
            logging.info("Calling API...")
            response = client.beta.chat.completions.parse(
                model='gpt-4o-2024-08-06',
                messages= messages,
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
            
def process_documents(directory):
    results = {}
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, file)
            logging.info(f"Processing: {file}")
            toc_page = find_table_of_contents(pdf_path)
            if toc_page:
                results[file] = f"{toc_page}"
            else:
                results[file] = ""
    return results

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    vanagon_filepath = os.path.join(current_dir, '..', 'data', 'test_raw', 'test1.pdf')

    make_toc(vanagon_filepath, 3)

if __name__ == '__main__':
    main()