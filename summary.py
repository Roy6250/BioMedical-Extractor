import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import json

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load environment variables
load_dotenv()

# Define structured output models
class ArticleInsights(BaseModel):
    """A structured representation of extracted biomedical article insights."""
    diseases: List[str] = Field(
        default_factory=list,
        description="List of disease entities mentioned in the article"
    )
    genes_proteins: List[str] = Field(
        default_factory=list,
        description="List of gene and protein identifiers discussed in the article"
    )
    pathways: List[str] = Field(
        default_factory=list,
        description="List of biological pathways referenced in the research"
    )
    experimental_methods: List[str] = Field(
        default_factory=list,
        description="List of techniques used (e.g., CRISPR, bulk RNASeq, etc.)"
    )
    key_findings: str = Field(
        default="",
        description="A 1-2 sentence summary of the article's main scientific insight"
    )
    pubmed_id: str = Field(
        default="",
        description="The PubMed ID of the article"
    )

def load_filtered_articles(filepath: str) -> List[str]:
    """Load filtered article IDs from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('filtered_articles', [])

def load_article_details(filepath: str, filtered_ids: List[str]) -> pd.DataFrame:
    """Load article details from CSV file and filter by PubMed IDs."""
    df = pd.read_csv(filepath)
    # Ensure PubMed IDs are treated as strings
    df['pubmed_id'] = df['pubmed_id'].astype(str)
    # Filter to include only articles in the filtered list
    filtered_df = df[df['pubmed_id'].isin(filtered_ids)]
    return filtered_df

def setup_gemini_llm() -> Gemini:
    """Set up Gemini LLM with API key."""
    api_key = os.environ.get('GOOGLE_API_KEY') or "AIzaSyA991yvb37XuLvC24A5-WSOKlCP-P-pfDA"
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Create Gemini LLM instance
    gemini = Gemini(
        model="models/gemini-2.0-flash",
        temperature=0.1,  # Low temperature for more deterministic outputs
        max_tokens=1024,
        top_p=0.95,
    )
    
    # Set as default LLM
    Settings.llm = gemini
    
    # Set up as structured LLM
    structured_gemini = gemini.as_structured_llm(ArticleInsights)
    
    return structured_gemini

def create_extraction_prompt(title: str, abstract: str, pubmed_id: str = "", keywords: Optional[str] = None) -> str:
    """Create a structured prompt for entity extraction."""
    keywords_text = f"Keywords: {keywords}\n" if keywords else ""
    
    prompt = f"""
        You are an expert biomedical information extractor.  
        Extract entities from the following research article and output a structured JSON object **strictly** with the following keys:  
        - diseases (list of strings)
        - genes_proteins (list of strings)
        - pathways (list of strings)
        - experimental_methods (list of strings)
        - key_findings (string, 1–2 sentences summarizing the main scientific insight) Use the Abstract to extract the key_findings.

        If a category has no relevant information, return an **empty list** (for lists) or an **empty string** (for key_findings).  
        Do not hallucinate or infer beyond the given text.

        Article:
        PubMed ID: {pubmed_id}
        Title: {title}
        {keywords_text}Abstract: {abstract}

        """
    return prompt

def extract_entities(structured_llm, article: pd.Series) -> ArticleInsights:
    """Extract entities from an article using structured LLM output."""
    title = article['title'] if pd.notna(article['title']) else ""
    abstract = article['abstract'] if pd.notna(article['abstract']) else ""
    keywords = article['keywords'] if pd.notna(article['keywords']) else None
    pubmed_id = article['pubmed_id']
    
    # Create prompt
    prompt = create_extraction_prompt(title, abstract, pubmed_id, keywords)

    print(prompt)
    
    # Query LLM with structured output
    try:
        result = structured_llm.complete(prompt)
        structured_response= json.loads(result.text)
        return structured_response
    except Exception as e:
        print(f"Error processing article {pubmed_id}: {str(e)}")
        # Return empty result with the PubMed ID
        empty_result = ArticleInsights(pubmed_id=pubmed_id)
        return empty_result

def process_articles(structured_llm, articles_df: pd.DataFrame) -> Dict[str, ArticleInsights]:
    """Process all articles and extract entities."""
    results = {}
    
    for _, article in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
        pubmed_id = article['pubmed_id']
        extracted_data = extract_entities(structured_llm, article)
        results[pubmed_id] = extracted_data
        
    return results


def main():
    # File paths
    filtered_ids_path = "/home/ubuntu/sayantan/ui/search/filtered_articles.json"
    articles_csv_path = "/home/ubuntu/sayantan/ui/search/output.csv"
    output_path = "extracted_insights.json"
    
    # Load filtered article IDs
    print("Loading filtered article IDs...")
    filtered_ids = load_filtered_articles(filtered_ids_path)
    print(f"Loaded {len(filtered_ids)} filtered article IDs")
    
    # Load article details
    print("Loading article details...")
    articles_df = load_article_details(articles_csv_path, filtered_ids)
    print(f"Loaded {len(articles_df)} articles")
    
    # Set up Gemini LLM with structured output
    print("Setting up Gemini LLM with structured output...")
    structured_llm = setup_gemini_llm()
    
    # Process articles
    print("Processing articles...")
    results = process_articles(structured_llm, articles_df)

    print(results)

    # with open('extracted_entities.json', 'w') as f:
    #     json.dump(results, f, indent=4)
    
    # Save results

    serializable_results = {}

    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = v
        else:
            serializable_results[k] = v.__dict__

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    
    print("Entity extraction completed!")

if __name__ == "__main__":
    main()