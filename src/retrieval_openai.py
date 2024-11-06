import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict
import openai
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s = %(levelname)s = %(message)s')

load_dotenv()

openai.api_key = os.getenv('OPEN_AI_API_KEY')

class OpenAIretriever:

    DEFAULT_MODEL = "gpt-4o-2024-08-06"

    def __init__(self, model: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL

    def load_page_summaries(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def search(self,
               query: str,
               context: List[Dict], # specifc for page summaries
               max_tokens: Optional[int] = 150,
               temperature: Optional[float] = 0.7
               ) -> str:

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

        response = openai.chat.completions.create(
            model = self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        generated_reponse = response.choices[0].message.content

        return generated_reponse
    
if __name__ == "__main__":
        current_dir = os.path.dirname(os.path.abspath(__file__))

        page_summaries_filepath = os.path.join(
            current_dir, "..", "data", "processed", "test1_page_summaries.json"
        )

        retriever = OpenAIretriever()

        page_summaries = retriever.load_page_summaries(page_summaries_filepath)

        query = "Where can I find information on engine codes?"
        context = page_summaries

        reponse = retriever.search(query, context)
        logging.info(f"The question was: {query}")
        logging.info(f"The response is: {reponse}")
        logging.info(f"The page and detailed section number should have been: 22 and 2a")