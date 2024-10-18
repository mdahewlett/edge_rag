import os
from dotenv import load_dotenv
import logging
from typing import List, Optional
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s = %(levelname)s = %(message)s')

# Load environment variables from the .env file
load_dotenv()

# Access env vars
openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_cost_per_token = float(os.getenv('OPENAI_PROMPT_TKN'))
completion_cost_per_token = float(os.getenv('OPENAI_COMPLETION_TKN'))

class OpenAIGenerator:
    DEFAULT_MODEL = "gpt-4o-2024-08-06"

    def __init__(self, model: Optional[str] = None):

        self.model = model or self.DEFAULT_MODEL
        logging.info(f"Initializing OpenAIGenerator with model: {self.model}")
        
    def generate(self, 
                 query: str, 
                 context: str, 
                 max_tokens: Optional[int] = 150,
                 temperature: Optional[float] = 0.7) -> str:
        
        logging.info(f"Generating response for query: {query[:50]}...")
        try:
            # Generate response
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract the response
            generated_text = response.choices[0].message.content
            logging.info(f"Generated response: {generated_text[:50]}...")

            # Estimating cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            prompt_cost = prompt_tokens*prompt_cost_per_token
            completion_cost = completion_tokens*completion_cost_per_token
            total_cost = prompt_cost+completion_cost
            logging.info(f"Prompt tokens used: {prompt_tokens}")
            logging.info(f"Prompt tokens cost: ${prompt_cost}")
            logging.info(f"Completion tokens used: {completion_tokens}")
            logging.info(f"Completion tokens cost: ${completion_cost}")
            logging.info(f"Total tokens cost: ${total_cost}")
            logging.info(f"Number of runs to cost $1: {1/total_cost:.0F} runs")

            return generated_text
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":

    # Default model
    generator = OpenAIGenerator()

    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Paris is the largest city in France."

    response = generator.generate(query, context)
    logging.info(f"Query: {query}")
    logging.info(f"Response: {response}")

    # Generating summaries
    query_summary = "Summarize the following text:"
    context_summary = 'The Volkswagen Westfalia, often called the "Westy," is a classic camper van that originated as a collaboration between Volkswagen (VW) and the German coachbuilder Westfalia-Werke. This iconic vehicle is based on Volkswagen\'s Transporter series and became widely popular from the 1950s to the early 2000s. Here''s a breakdown of its history: 1. Origins in the 1950s The Volkswagen Type 2 (the Transporter) was introduced in 1950, following the success of the VW Beetle. Westfalia, a German company that specialized in building vehicle conversions, began converting these VW Transporters into camper vans. The earliest conversions were simple, with fold-out beds, a small table, and storage areas, offering basic functionality for travel and camping. 2. 1960s: The Split-Windshield Era The VW T1 (or Type 2) with its split windshield became the foundation for early Westfalia conversions. During this time, the camper van gained immense popularity with outdoor enthusiasts and the growing counterculture movement, especially in the U.S. The Westfalia conversions typically included pop-up roofs, which allowed for more standing space inside the van when parked, as well as fold-out beds, a kitchen area, and cupboards. The iconic "pop-top" design was introduced, making it practical for long camping trips by providing additional space for sleeping and living. 3. 1970s: The Bay Window Era The T2 (Bay Window) model, introduced in 1967, further refined the VW Westfalia. This version featured a single-piece windshield, more powerful engines, and improvements in design and handling. The 1970s models became symbols of the hippie movement, embodying freedom and adventure. Westfalia continued to innovate with practical additions, including water tanks, refrigerators, stoves, and propane heating. These vans were marketed as affordable alternatives to larger motorhomes, making them popular with young families and adventure-seekers. 4. 1980s: The Vanagon Era (T3) In 1979, Volkswagen introduced the T3 (known as the Vanagon in North America). This model had a more boxy, modern look and a rear-engine layout. The T3 Westfalia campers became more sophisticated, offering features like swiveling captain\'s chairs, larger beds, and better insulation. Some models even had an optional pop-up roof tent, allowing campers to sleep in an upstairs area. The T3 also introduced four-wheel-drive (the Syncro variant), making it more versatile for off-road adventures. It was during this era that the Westfalia solidified its status as a practical yet adventurous vehicle for long-distance travel. 5. 1990s: The Eurovan Era (T4) The T4, known as the Eurovan in North America, marked a major shift in design, as the engine was now moved to the front of the vehicle, giving the van more interior space. The Westfalia versions of the T4 continued to offer pop-up roofs, fold-out beds, and kitchenettes, but with a more modern touch. However, the T4 never achieved the same cult status as the earlier models, and by the late 1990s, production of the VW Westfalia camper was phased out for most markets, with the rise of other, more modern camper van options. 6. 2000s: End of an Era By the early 2000s, the official partnership between VW and Westfalia ended. However, independent companies and enthusiasts continued converting Volkswagen vans into campers, maintaining the spirit of the Westfalia camper van. Legacy and Popularity The VW Westfalia remains an iconic vehicle, beloved by van-life enthusiasts, collectors, and adventurers worldwide. Its practical design, coupled with the nostalgic charm of the VW Transporter, has cemented it as a symbol of freedom and exploration. Even today, original and restored VW Westfalias fetch high prices and are cherished as classic vehicles. Its influence continues, with modern camper vans and the resurgence of "van life" culture taking inspiration from the simplicity and functionality of the original Westfalia camper.' # Imagine this is a long document
    response = generator.generate(query_summary, context_summary, max_tokens=150) # Limit summary to max 150 tokens
    logging.info(f"Summary:\n\n {response}")



