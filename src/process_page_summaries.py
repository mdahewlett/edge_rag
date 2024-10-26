import re
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def process_sections(data_list):
    section_num_misplaced = 0
    section_num_from_detailed = 0
    tagged_as_index = 0
    trailing_punct_removed = 0

    for item in data_list:

        # If section_num in section_name, move to section_num
        if not item['section_num'] and item['section_name']:
            matches = re.search(r'(?:^(\d{1,2})\s+|\s+(\d{1,2})\b)', item['section_name'])
            
            if matches:
                number = matches.group(1) if matches.group(1) else matches.group(2)
                item['section_num'] = number
                item['section_name'] = re.sub(r'^\d{1,2}\s+|\s+\d{1,2}\b', '', item['section_name']).strip()
                section_num_misplaced += 1
    
        # If section_num still missing, pull from section_num_detailed
        if not item['section_num'] and item['section_num_detailed']:
            matches = re.match(r'(\d+)[a-zA-Z.]?.*', item['section_num_detailed'])
            if matches:
                item['section_num'] = matches.group(1)
                section_num_from_detailed += 1

        # Remove trailing punctuation from section_name
        if item['section_name']:
            original_section_name = item['section_name']
            item['section_name'] = re.sub(r'[-\s.,;:]+$', '', item['section_name']).strip()
            if original_section_name != item['section_name']:
                trailing_punct_removed += 1
        
        # If section_name missing and empty topic_list after page 19, mark as index
        if not item['section_name'] and not item['topic_list'] and item['page_num'] >= 19:
            item['topic_list'].append('Index')
            tagged_as_index += 1

    logging.info(f"Number of section numbers moved from section_name to section_num: {section_num_misplaced}")
    logging.info(f"Number of section numbers added from section_num_detailed: {section_num_misplaced}")
    logging.info(f"Number of pages with 'Index' added to topic_list: {section_num_misplaced}")
    logging.info(f"Number of section names with trailing punctuation removed: {trailing_punct_removed}")

    return data_list

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_filepath = os.path.join(
        current_dir, "..", "data", "processed", "test1_page_summaries.json"
    )

    data = load_json(data_filepath)

    logging.info("Processing data...")
    processed_data = process_sections(data)

    with open(data_filepath, "w") as f:
        json.dump(processed_data, f)
    logging.info(f"Saved json page summaries")

if __name__ == "__main__":
    main()
