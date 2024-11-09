import unittest
import json
import os
from src.retrieval_openai import OpenAIRetriever
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class InstrumentedRetriever(OpenAIRetriever):
    def __init__(self, model=None):
        super().__init__(model)
        self.request_count = 0

    def search(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.request_count += 1

        # Log before making the request
        logging.info(f"\n=== API Request #{self.request_count} ===")
        logging.info(f"Query: {query}")

        # Make the API call and get result
        response, usage = super().search(query, context)

        # Log the completion of the request
        logging.info(f"Request completed")
        logging.info(f"Token usage:")
        logging.info(f"  Prompt tokens: {usage.prompt_tokens}")
        logging.info(f"  Completion tokens: {usage.completion_tokens}")
        logging.info(f"  Total tokens: {usage.total_tokens}")
        logging.info("========================\n")

        return response


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = InstrumentedRetriever()
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load page summaries
        cls.page_summaries_path = os.path.join(
            current_dir, "test_data", "test_page_summaries.json"
        )
        cls.page_summaries = cls.retriever.load_page_summaries(cls.page_summaries_path)

        # Load test cases
        test_cases_path = os.path.join(
            current_dir, "test_data", "retrieval_test_cases.json"
        )
        with open(test_cases_path, "r") as f:
            cls.test_cases = json.load(f)

    def test_single_page_search_results(self):
        for test_case in self.test_cases["single_page_tests"]:

            logging.info(f"Testing the following case: {test_case}")
            with self.subTest(msg=test_case["description"]):
                result = self.retriever.search(
                    query=test_case["query"], context=self.page_summaries
                )
                self.assertEqual(
                    test_case["page_num"],
                    result["page_num"],
                    f"Page numbers don't match for: {test_case['description']}",
                )
                self.assertEqual(
                    test_case["section_num_detailed"],
                    result["section_num_detailed"],
                    f"Section numbers don't match for: {test_case['description']}",
                )

    def test_single_section_multiple_pages_search_results(self):
        for test_case in self.test_cases["single_section_multiple_page_tests"]:

            logging.info(f"Testing the following case: {test_case}")
            with self.subTest(msg=test_case["description"]):
                result = self.retriever.search(
                    query=test_case["query"], context=self.page_summaries
                )
                self.assertCountEqual(
                    test_case["page_num"],
                    result["page_num"],
                    f"Page numbers don't match for: {test_case['description']}",
                )
                self.assertCountEqual(
                    test_case["section_num_detailed"],
                    result["section_num_detailed"],
                    f"Section numbers don't match for: {test_case['description']}",
                )


if __name__ == "__main__":
    unittest.main()
