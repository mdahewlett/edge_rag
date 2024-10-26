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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

logging.info("Initializing OpenAI client")
client = OpenAI(api_key=openai_api_key)
textract = boto3.client(
    "textract",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)


def load_coordinate_zones(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def extract_text_from_zones(image, zones):
    zone_texts = {"page_header": [], "page_footer": []}

    for zone_name, zone_coords in zones.items():

        coordinates = (
            zone_coords["left"],
            zone_coords["top"],
            zone_coords["right"],
            zone_coords["bottom"],
        )

        # crop the image to the zone of interest
        zone = image.crop(coordinates)

        # convert image format
        img_byte_arr = io.BytesIO()
        zone.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # call API to extract text
        response = textract.detect_document_text(Document={"Bytes": img_byte_arr})

        # sort text into header/footer to separate section name from topics and tags
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                zone_texts[zone_name].append(item["Text"])

    return zone_texts["page_header"], zone_texts["page_footer"]


def extract_text_from_pages(pdf_path, zones):

    # convert pdf to images
    pages = convert_from_path(pdf_path)

    results = []
    for i, page in enumerate(pages):

        # give developper progress updates
        if i != 0 and i % 10 == 0:
            logging.info(f"extracting text from page number {i}")

        page_dict = {
            "section_num": "",
            "section_name": "",
            "section_num_detailed": "",
            "page_num": i,
            "topic_list": [],
        }

        # ignore first 20 pages as edge cases for now
        if i < 19:
            results.append(page_dict)
            continue

        # extract text
        page_header, page_footer = extract_text_from_zones(page, zones)

        # add section name and number from header text
        for item in page_header:
            item = item.strip()
            if re.match(r"^\d+$", item):
                page_dict["section_num"] = item
            else:
                page_dict["section_name"] = item

        # add remaining page info from footer text
        for item in page_footer:
            item = item.strip()
            # section number
            if re.match(r"^\d+(\.\d+)?[A-Za-z]?$", item):
                page_dict["section_num_detailed"] = item

            # throw out page format e.g. A-10, G-4
            elif re.match(r"^[A-Za-z]-\d+$", item):
                continue

            # topics and tags go together
            else:
                page_dict["topic_list"].append(item)

        results.append(page_dict)
    return results


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # select manual pdfs and coordinate zones to test
    pdf_path = os.path.join(current_dir, "..", "data", "raw", "ocr_130page.pdf")
    json_path = os.path.join(
        current_dir, "..", "data", "processed", "ocr_coordinates.json"
    )
    save_filepath = os.path.join(
        current_dir, "..", "data", "processed", "test1_page_summaries.json"
    )

    zones = load_coordinate_zones(json_path)
    logging.info(f"Loaded coordinate zones: {zones}")

    results = extract_text_from_pages(pdf_path, zones)
    logging.info(f"Results:\n")
    for result in results:
        logging.info(f"{result}\n")

    logging.info(f"Format of results object:\n{results}")

    logging.info(f"Saving json page summaries")
    with open(save_filepath, "w") as f:
        json.dump(results, f)
    logging.info(f"Saved json page summaries")


if __name__ == "__main__":
    main()
