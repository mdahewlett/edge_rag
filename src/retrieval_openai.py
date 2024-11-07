import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
import openai
from openai.types.completion_usage import CompletionUsage
import json
from pydantic import BaseModel

# Output structure
class PageNumbers(BaseModel):
     page_num: List[str]
     section_num_detailed: List[str]

logging.basicConfig(level=logging.INFO, format='%(asctime)s = %(levelname)s = %(message)s')

load_dotenv()

openai.api_key = os.getenv('OPEN_AI_API_KEY')

class OpenAIRetriever:

    DEFAULT_MODEL = "gpt-4o-2024-08-06"

    def __init__(self, model: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL

    def load_page_summaries(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def search(self,
               query: str,
               context: List[Dict], # specifc for page summaries
               ) -> Tuple[Dict, CompletionUsage]:

        messages = [
            {
                "role": "system",
                "content": f"""
                     You find relevant page numbers based on topics. Here are the page summaries in a structured list:
                     {json.dumps(context, indent=2)}

                    Respond with the page numbers and detailed section numbers.
                     """,
            },
            {"role": "user", "content": query},
        ]

        response = openai.beta.chat.completions.parse(
                model=self.model,
                messages= messages,
                response_format=PageNumbers,
            )

        generated_reponse = json.loads(response.choices[0].message.content)
        usage = response.usage

        return (generated_reponse, usage)
    
if __name__ == "__main__":
        current_dir = os.path.dirname(os.path.abspath(__file__))

        page_summaries_filepath = os.path.join(
            current_dir, "..", "tests", "test_data", "test_page_summaries.json"
        )

        retriever = OpenAIRetriever()

        page_summaries = retriever.load_page_summaries(page_summaries_filepath)

        query = "Where can I find information on engine codes?"
        context = page_summaries

        response, _ = retriever.search(query, context)
        logging.info(f"The question was: {query}")
        logging.info(f"The response is: {response}")
        logging.info(f"The page and detailed section number should have been: 21 and 2a")