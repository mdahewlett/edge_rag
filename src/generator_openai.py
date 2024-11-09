import logging.config
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
import logging
from openai.types.completion_usage import CompletionUsage
from PyPDF2 import PdfReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

logging.info("Initializing OpenAI client")
client = OpenAI(api_key=openai_api_key)


class OpenAIGenerator:
    DEFAULT_MODEL = "gpt-4o-2024-08-06"

    def __init__(self, model: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        logging.info(f"Initialized generator with model {self.model}")

    def get_context(self, document_path, page_numbers: List[str]):
        reader = PdfReader(document_path)
        context = {}  # create empty dict object
        for page_number in page_numbers:
            page = reader.pages[int(page_number)]
            text = page.extract_text()
            context[page_number] = text
        return context  # a dict key page number keys, and page text values

    def report(self, query: str, context: Dict) -> Tuple[str, CompletionUsage]:

        messages = [
            {
                "role": "system",
                "content": f"""
            You are given pages that are likely to be useful to a user's question. 
            
            Your job is to check which pages are actually relevant based on their text, then answer the user's question. 
            
            Only use the page numbers and page text below to answer the user's question.

            Here is are the pages that are likely to be relevant in a dictionary with page numbers as keys and page text as values:
            {context}

            Your answer should include a list of pages. 
            
            For each relevant page, give:
            - the page number
            - a short explanation of why you think it is relevant to the user's question
            - a short excerpt from the page text that supports your explanation

            For each irrelevant page, give:
            - the page number
            - a short explanation of why you think it is not relevant to the user's question

            """,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = client.chat.completions.create(messages=messages, model=self.model)

        generated_response = response.choices[0].message.content
        usage = response.usage
        return (generated_response, usage)


if __name__ == "__main__":
    generator = OpenAIGenerator()

    example_document = "manual_130.pdf"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    document_path = os.path.join(current_dir, "..", "data", "raw", example_document)
    pages = ["36", "38", "41", "82"]

    context = generator.get_context(document_path, pages)
    query = "Where can I find information on torque values for installing the engine?"

    response, usage = generator.report(query, context)

    logging.info(f"Query: {query}")
    logging.info(f"Answer: {response}")
    logging.info(
        f"""\n
        Prompt tokens: {usage.prompt_tokens} \n
        Completion tokens: {usage.completion_tokens} \n
        Total tokens: {usage.prompt_tokens + usage.completion_tokens}
        """
    )
    logging.info(f"Context:\n{context}")
