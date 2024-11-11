from PyPDF2 import PdfReader
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
document_path = os.path.join(current_dir, "..", "data", "raw", "manual_130.pdf")
page_list = [29, 31, 42]

reader = PdfReader(document_path)

pages = {}
for page_number in page_list:
    page_info = {}
    page_info['relevant'] = 'yes'
    text = reader.pages[page_number].extract_text()
    page_info['page_text'] = text
    pages[str(page_number)] = page_info

print(json.dumps(pages))
