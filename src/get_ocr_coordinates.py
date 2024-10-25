import cv2
import json
from pdf2image import convert_from_path
import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_coordinates(pdf_path: str, output_path: str) -> Optional[Dict[str, int]]:
    points: List[Tuple[int, int]] = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
            print(f"Clicked at: ({x}, {y})")
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.rectangle(img, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow('image', img)
    
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    first_page = pages[0]    

    img = np.array(first_page)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    print("Click top-left and bottom-right corners. Press 'q' when done.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

    if len(points) >= 2:
        coords = {
            "left": points[0][0],
            "top": points[0][1],
            "right": points[1][0],
            "bottom": points[1][1]
        }

        with open(output_path, 'w') as f:
            json.dump(coords, f, indent=4)
        
        print(f"Coodinates saved to {output_path}")
        return coords
    return None

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    raw_filepath = os.path.join(current_dir, '..', 'data', 'raw', 'ocr_1page_right.pdf')
    processed_filepath = os.path.join(current_dir, '..', 'data', 'processed', 'ocr_coordinates.json')

    coords = get_coordinates(raw_filepath, processed_filepath)

    logging.info(f"The coordinates are:\n{coords}")