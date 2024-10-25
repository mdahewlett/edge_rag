import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_coordinates(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return (data['left'], data['top'], data['right'], data['bottom'])

def extract_text_from_zone(image, coordinates):
    zone = image.crop(coordinates)

    return pytesseract.image_to_string(zone)

def extract_text_from_pages(pdf_path, coordinates):
 
    pages = convert_from_path(pdf_path)

    results = []
    for page in pages:
        text = extract_text_from_zone(page, coordinates)
        results.append(text)
    
    return results

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(current_dir, '..', 'data', 'raw', 'ocr_1page_left.pdf')
    json_path = os.path.join(current_dir, '..', 'data', 'processed', 'ocr_coordinates.json')

    coordinates = load_coordinates(json_path)
    logging.info(f"Loaded coordinates: {coordinates}")

    results = extract_text_from_pages(pdf_path, coordinates)
    logging.info(f"Result:\n{results}")

if __name__ == "__main__":
    main()