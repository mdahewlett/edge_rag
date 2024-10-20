import unittest
from find_toc import find_table_of_contents
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestFindTOC(unittest.TestCase):
    def test_find_table_of_contents(self):

        test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_raw')

        test_cases = [
            ('test1.pdf', 4), # Vanagon 
            ('test2.pdf', 1), # Dyson, has returned None before
            ('test3.pdf', 7), # Raspberry Py
            ('test4.pdf', None) # Vevor Heater
        ]

        for filename, expected_page in test_cases:
            with self.subTest(filename=filename):
                pdf_path = os.path.join(test_data_dir, filename)
                logging.info(f"Testing {filename}")
                result = find_table_of_contents(pdf_path)

                if expected_page is None:
                    self.assertIsNone(result, f"Expected no TOC for {filename}, but found one on page {result}")
                else:
                    if result is None:
                        self.fail(f"For {filename}, expected TOC on page {expected_page}, but no TOC was found")
                    else:
                        self.assertEqual(result, expected_page, f"For {filename}, expected TOC on page {expected_page}, but found it on page {result}")

if __name__ == '__main__':
    unittest.main()