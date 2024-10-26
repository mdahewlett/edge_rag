import boto3
from PIL import Image
from pdf2image import convert_from_path
import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import io
import re

# Defining structure of the page summary text
class PageSummary(BaseModel):
    page_num: int
    section_num: str

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

logging.info("Initializing OpenAI client")
client = OpenAI(api_key=openai_api_key)
textract = boto3.client(
    'textract',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def load_coordinates(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return (data['left'], data['top'], data['right'], data['bottom'])

def extract_text_from_zone(image, coordinates):
    zone = image.crop(coordinates)

    img_byte_arr = io.BytesIO()
    zone.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = textract.detect_document_text(
        Document={'Bytes': img_byte_arr}
    )

    text = ""
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + "\n"
    
    return text.strip()

def extract_text_from_pages(pdf_path, coordinates):
    pages = convert_from_path(pdf_path)

    results = []
    for i, page in enumerate(pages):
        if i != 0 and i % 10 == 0:
            logging.info(f"extracting text from page number {i}")
        page_dict = {'topic_list': []}
        text = extract_text_from_zone(page, coordinates)
        split_text = text.split('\n')
        for item in split_text:
            if re.match(r"^\d+(\.\d+)?[A-Za-z]?$", item):
                page_dict['section_num'] = item
            elif re.match(r"^[A-Za-z]-\d+$", item):
                continue
            else:
                page_dict['topic_list'].append(item)
        page_dict['page_num'] = i
        results.append(page_dict)
    return results

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(current_dir, '..', 'data', 'raw', 'ocr_1page_right.pdf')
    json_path = os.path.join(current_dir, '..', 'data', 'processed', 'ocr_coordinates.json')

    coordinates = load_coordinates(json_path)
    logging.info(f"Loaded coordinates: {coordinates}")

    results = extract_text_from_pages(pdf_path, coordinates)
    logging.info(f"Results:\n")
    for result in results:
        logging.info(f"{result}\n")

if __name__ == "__main__":
    main()