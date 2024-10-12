import os
import logging
from typing import List, Optional
import torch
import platform
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s = %(levelname)s = %(message)s')

class Generator:
    """
    A text generation class using pre-trained language models.

    This class provides functionality to generate text responses based on input queries and context using various pre-trained language models. It supports loading models from local paths or Hugging Face's model hub, and can utilize GPU acceleration when available.

    Attributes:
        model_source (str): The source of the loaded model (local path or model name).
        tokenizer (AutoTokenizer): The tokenizer associated with the loaded model.
        model (AutoModelForSeq2SeqLM): The loaded language model.
        input_max_length (int): Maximum input length for the model.
        device (torch.device): The device used for computation.

    Methods:
        generate: Generate a response based on input query and context.
        save_model: Save the current model to a specified path.
        load_model: Load a model from a specified path.

    Raises:
        ValueError: If an invalid model path is provided during initialization.
        Exception: For any other initialization, generation, or I/O errors.
    """
    DEFAULT_MODEL = "google/flan-t5-small"

    def __init__(self, 
                 model_path: Optional[str] = None, 
                 input_max_length: Optional[int] = 512, 
                 device: Optional[str] = None):
        """
        Initialize the Generator with a model path or default model, and specify device.

        Args:
            model_path (Optional[str]): The path to a local model or name of a Hugging Face model.
            input_max_length (int): Maximum input length for model.
            device (Optional[str]): The device to use for computation ('cpu', 'cuda', 'mps', or None for auto-detect).
        
        Raises:
            ValueError: If the provided model_path does not exist.
            Exception: For any other initialization errors.
        """
        self.model_source = self._get_model_source(model_path)
        logging.info(f"Initializing Generator with model: {self.model_source}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_source)
            self.input_max_length = input_max_length

            self.device = self._get_device(device)
            self.model.to(self.device)

            logging.info(f"Model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise
    
    def _get_model_source(self, model_path: Optional[str]) -> str:
        """
        Determine the model source based on the provided path or use default.

        Args:
            model_path (Optional[str]): The path to a local model or name of a Hugging Face model.

        Returns:
            str: The model source to use (either a local path or a model name).

        Raises:
            ValueError: If the provided model_path does not exist locally and is not a valid Hugging Face model.
        """
        if model_path is not None:
            if os.path.exists(model_path):
                return model_path
            else:

                try:
                    AutoTokenizer.from_pretrained(model_path)
                    return model_path
                except Exception:
                    logging.error(f"Provided model path '{model_path}' does not exist locally and is not a valid Hugging Face model.")
                    raise ValueError(f"Provided model path '{model_path}' does not exist locally and is not a valid Hugging Face model. Please provide a valid path, a valid Hugging Face model name, or None to use the default model.")

        logging.info(f"No model path provided. Using default model: {self.DEFAULT_MODEL}")
        return self.DEFAULT_MODEL

    def _get_device(self, device: Optional[str]) -> torch.device:
        """
        Determine the appropriate device to use.

        Args:
            device (Optional[str]): The requested device, or None for auto-detect.

        Returns:
            torch.device: The device to use for computation.
        """
        if device is not None:
            return torch.device(device)
        
        # Check if GPU is available, move model to GPU if possible
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            mac_version = platform.mac_ver()[0]
            if mac_version >= "14.0":
                return torch.device("mps")
            else:
                logging.warning("MPS is available but may not be fully compatible. Falling back to CPU.")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def generate(self, 
                 query: str, 
                 context: List[str], 
                 output_max_length: Optional[int] = 512, 
                 num_beams: Optional[int] = 4, 
                 temperature: Optional[float] = 1.0) -> str:
        """
        Generate a response based on the query and context.

        Args:
            query (str): The input query.
            context (List[str]): A list of relevant context strings.
            output_max_length (int): Maximum length of the generated output.
            num_beams (int): 
            temperature (float): Temperature for sampling.

        Returns:
            str: The generated response.

        Raises:
            Exception: If an error occurs during generation.
        """
        logging.info(f"Generating response for query: {query[:50]}...")
        try:
            # Prepare input by combining query and context
            logging.info("Combining query and context...")
            input_text = f"Query: {query}\n\nContext: {' '.join(context)}\n\nAnswer:"

            # Tokenize input
            logging.info("Tokenizing input...")
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.input_max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            # Generate output
            logging.info("Generating output...")
            outputs = self.model.generate(
                **inputs, 
                max_length=output_max_length, 
                num_beams=num_beams,
                temperature=temperature,
                num_return_sequences=1 
                )

            # Decode output
            logging.info("Decoding output...")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logging.info(f"Generated response: {response[:50]}...")
            return response
        except ValueError as e:
            logging.error(f"Invalid input parameters: {str(e)}")
            raise
        except RuntimeError as e:
            logging.error(f"Error during generation process: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            raise

    def save_model(self, path: str):
        """
        Saves the model to a path provided.

        Args:
            path (str): The path to save the model at.

        Returns:
            None
        
        Raises:
            Exception: If an error occurs during saving.
        """
        logging.info(f"Saving model to {path}")
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """
        Loads a model from a provided path, updating the instance's model and tokenizer attributes.

        Args:
            path (str): The path of the existing model.
        
        Returns:
            None

        Raises:
            Exception: If an error occurs loading the model.
        """
        logging.info(f"Loading model from {path}")
        try:
            # Check if the saved model matches the current model type
            config_path = os.path.join(path, "model_config.txt")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_info = f.read()
                    if type(self.model).__name__ not in config_info:
                        raise ValueError("Saved model type does not match current model type")

            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model.to(self.device)
            logging.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":

    # Default model
    generator_default = Generator()

    # This will raise an error if the path doesn't exist
    try:
        generator_local = Generator(model_path="/path/to/local/model")
    except ValueError as e:
        logging.error(f"Error demonstrating invalid local path: {str(e)}")
        logging.info("Continuing with execution...")

    # This will work with a Hugging Face model
    generator_hf = Generator(model_path="facebook/bart-large")

    query = "What is the capital of France?"
    context = ["France is a country in Western Europe.", "Paris is the largest city in France."]

    response = generator_default.generate(query, context)
    logging.info(f"Query: {query}")
    logging.info(f"Response: {response}")

    # Example saving and loading the model
    generator_default.save_model("./saved_model")
    generator_default.load_model("./saved_model")

    # Generate again to verify loaded model works
    new_response = generator_default.generate(query, context)
    logging.info(f"New Response: {new_response}")

    # Generating summaries
    generator_summary = Generator(input_max_length=1024)
    query_summary = "Summarize the following text:"
    context_summary = ["A very long piece of text..."] # Imagine this is a long document
    response = generator_default.generate(query_summary, context_summary, output_max_length=150) # Limit summary to max 150 tokens
    logging.info(f"Resonse: {response}")