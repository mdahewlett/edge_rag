import cv2
import json
import os
import logging
import numpy as np
from pdf2image import convert_from_path
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_coordinates(pdf_path: str, output_path: str) -> Optional[Dict[str, Dict[str, int]]]:
    zones: Dict[str, Dict[str, int]] = {}
    current_points: List[Tuple[int, int]] = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x,y))
            print(f"Clicked at: ({x}, {y})")
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            if len(current_points) == 2:
                cv2.rectangle(img, current_points[0], current_points[1], (0, 255, 0), 2)
            cv2.imshow('image', img)
    
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    first_page = pages[0]    

    img = np.array(first_page)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    original_img = img.copy()
    
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    print("Click top-left and bottom-right corners. Press 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and len(current_points) == 2:
            zone_name = input("Enter zone name (e.g., 'page_header', 'page_footer'): ").strip()
            if zone_name:
                zones[zone_name] = {
                    "left": current_points[0][0],
                    "top": current_points[0][1],
                    "right": current_points[1][0],
                    "bottom": current_points[1][1]
                }
                logging.info(f"Saved zone '{zone_name}'")

                current_points = []
                img = original_img.copy()

                for name, coords in zones.items():
                    pt1 = (coords["left"], coords["top"])
                    pt2 = (coords["right"], coords["bottom"])
                    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

                    cv2.putText(img, name, (pt1[0], pt1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('image', img)

        elif key == ord('r'):
            current_points = []
            img = original_img.copy()

            for name, coords in zones.items():
                pt1 = (coords["left"], coords["top"])
                pt2 = (coords["right"], coords["bottom"])
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(img, name, (pt1[0], pt1[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('image', img)

    cv2.destroyAllWindows()

    if zones:
        with open(output_path, 'w') as f:
            json.dump(zones, f, indent=4)
        
        print(f"Coodinates saved to {output_path}")
        return zones
    return None

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    raw_filepath = os.path.join(current_dir, '..', 'data', 'raw', 'ocr_1page_right.pdf')
    processed_filepath = os.path.join(current_dir, '..', 'data', 'processed', 'ocr_coordinates.json')

    coords = get_coordinates(raw_filepath, processed_filepath)

    logging.info(f"The coordinates are:\n{coords}")