# edge_rag

## Flow
Open a one page pdf, draw bounding boxes, output the coordinates
input(s): 1 page pdf
processor: get_ocr_coordinates.py
output(s): coordinates json

Go through each page of the manual pdf, crop it using the coordinates, extract the text with OCR, store the information in a page summary dictionary, output the list of page summaries
input(s): manual pdf, coordinates json
processor: make_page_summaries.py
output(s): page summaries json

Take a query and the page summaries, send to an LLM, return the likely relevant page numbers
Input(s): query, page summaries json
Processor: retrieval_openai.py
Output(s): page numbers dictionary

## Tests
Take retrieval test cases and page summaries, check that the right page numbers are returned
Input(s): retrieval test cases json, test page summaries json, retrieval processor
Processor: test_retriever.py
Output(s): test results and errors


## Inactive
Transfer information page summaries into section summaries
Input(s): page summaries json
Processor: make_section_summaries.py
Output(s): section summaries json

Take a pdf, convert each page to an image, send to an LLM to identify the page with the table of contents, send that page to a second LLM to generate a table of contents, output the table of contents
Input(s): a pdf
Processor: make_toc.py
Output(s): table of contents json



preprocessing file
take in docs, make them lower case, put all the text into one string
dictionary
    filename
    full text
    chunks

embedding file
take in document dictionaries, create embedding dictionary
dictionary
    document index
    chunk index
    chunk embedding
take in query, create an embedding

retrieval file
take in a embedding dictionary, load it into an index
take in a query embedding, retrieve embedding dictionaries similar
dictionary
    document index
    chunk index
    chunk embedding
    score as distance

pipeline
takes the embedding dictionaries, gets the chunk text, sticks it together, send context and query to the generatoe

generator



embeddings.py - takes preprocessed documents item, make a List[Dict] where each dictionary is for one chunk from one document. The dictionary has the document index, chunk index, and the embedding of that chunk.


## Limitations

- The FaissRetriever does not support document deletion. Adding this feature would require refacctoring of the ID management system and the FAISS index handling.