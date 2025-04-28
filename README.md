# Biomedical Article Insight Extractor using Gemini & LlamaIndex

This Python script leverages the Google Gemini LLM via the LlamaIndex framework to extract structured biomedical insights from a collection of research articles based on their titles and abstracts. It processes articles specified in a filtered list and outputs the extracted entities into a JSON file.

## Features

*   **Structured Extraction:** Uses Pydantic models and LlamaIndex's structured output capabilities to ensure the LLM returns data in a predefined JSON format.
*   **Targeted Entities:** Extracts specific biomedical information:
    *   Diseases
    *   Genes/Proteins
    *   Biological Pathways
    *   Experimental Methods
    *   Key Findings (1-2 sentence summary)
*   **Filtered Processing:** Operates only on a predefined list of PubMed IDs provided in a separate JSON file.
*   **LLM Configuration:** Configured to use Google's `gemini-2.0-flash` model with specific settings (temperature, max tokens).
*   **Error Handling:** Includes basic error handling for LLM API calls, logging errors and providing default empty results.
*   **Environment Variable Management:** Uses `python-dotenv` to load the necessary API key.

## Workflow

1.  **Load Filtered IDs:** Reads a list of PubMed IDs from `filtered_articles.json`.
2.  **Load Article Data:** Reads article details (title, abstract, keywords, PubMed ID) from `output.csv`.
3.  **Filter Data:** Filters the loaded articles to keep only those whose PubMed IDs are in the filtered list.
4.  **Setup LLM:** Initializes the Gemini LLM via LlamaIndex, configured for structured output based on the `ArticleInsights` Pydantic model.
5.  **Process Articles:** Iterates through each filtered article:
    *   Constructs a detailed prompt containing the article's title, abstract, keywords (if available), and PubMed ID, instructing the LLM to extract entities according to the specified structure.
    *   Sends the prompt to the structured Gemini LLM.
    *   Parses the JSON response from the LLM.
    *   Handles potential errors during the LLM call.
6.  **Store Results:** Collects the extracted insights (as `ArticleInsights` objects or dictionaries) into a dictionary keyed by PubMed ID.
7.  **Save Output:** Saves the collected results to `extracted_entities.json`.

