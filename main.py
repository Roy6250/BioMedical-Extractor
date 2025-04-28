import os
import re
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def parse_markdown_file(file_path):
    """Parse a markdown file into a dictionary with sections."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract the PubMed ID from the filename
    pubmed_id = os.path.basename(file_path).replace('.md', '')
    
    # Extract title
    title_match = re.search(r'^# (.*)', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    # Extract journal
    journal_match = re.search(r'\*\*Journal:\*\* (.*)', content)
    journal = journal_match.group(1).strip() if journal_match else None

    # Extract keywords
    keywords_match = re.search(r'\*\*Keywords:\*\* (.*)', content)
    keywords = keywords_match.group(1).strip() if keywords_match else None

    # Extract abstract
    abstract_match = re.search(r'## Abstract\s*\n(.*?)(?:\n#|\Z)', content, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else None
    
    return {
        'pubmed_id': pubmed_id,
        'title': title,
        'journal': journal,
        'abstract': abstract,
        'keywords': keywords
    }


def load_articles(directory):
    """Load all markdown files from a directory into a pandas DataFrame."""
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            article_data = parse_markdown_file(file_path)
            print(article_data)
            articles.append(article_data)
    
    return pd.DataFrame(articles)

def create_combined_text(df):
    """Create a combined text field from title, abstract, and keywords."""
    return df.apply(
        lambda row: ' '.join(filter(None, [
            # row['title'] if pd.notna(row['title']) else '',
            # row['abstract'] if pd.notna(row['abstract']) else '',
            row['keywords'] if pd.notna(row['keywords']) else (row['title'] if pd.notna(row['title']) else '')
        ])), 
        axis=1
    )

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, tokenizer, model, batch_size=8):
    """Get embeddings for a list of texts using the PubMedBERT model."""
    all_embeddings = []
    
    # Process in batches to prevent memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the texts
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Convert to numpy array
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    return np.vstack(all_embeddings)

def filter_with_pubmedbert(df, threshold=0.3):
    """
    Filter articles using PubMedBERT embeddings from Hugging Face and cosine similarity.
    
    Args:
        df: DataFrame containing article data
        threshold: Similarity threshold for filtering (default: 0.45)
    
    Returns:
        DataFrame containing only filtered articles
    """
    print(f"Starting with {len(df)} articles")
    
    # Create combined text field for embedding
    df['cleaned_keywords'] = create_combined_text(df)

    
    # Define queries for cancer and immunology
    # queries = [
    #     "Research on cancer, oncology, tumors, carcinomas, leukemia, lymphoma, metastasis, and malignant neoplasms",
    #     "Research on immunology, immune system, antibodies, antigens, cytokines, lymphocytes, T-cells, B-cells, inflammation, and autoimmune disorders"
    # ]
    queries = [
        "This article discusses cancer research, including studies on oncology, tumors, carcinomas, leukemia, lymphoma, metastasis, malignant neoplasms, oncogenic mutations, cancer treatment, cancer therapy, cancer progression, or cancer biomarkers",
        "This article discusses immunology research, including studies on the immune system, antibodies, antigens, cytokines, lymphocytes, T-cells, B-cells, natural killer cells, macrophages, inflammation, autoimmune disorders, immune response, immunotherapy, or vaccines"
    ]
    
    try:
        # Load PubMedBERT model and tokenizer
        print("Loading PubMedBERT model and tokenizer...")
        model_name = "neuml/pubmedbert-base-embeddings"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Generate embeddings for queries
        print("Generating embeddings for queries...")
        query_embeddings = get_embeddings(queries, tokenizer, model)
        
        # Generate embeddings for documents
        print(f"Generating embeddings for {len(df)} articles...")
        document_texts = df['cleaned_keywords'].tolist()
        document_embeddings = get_embeddings(document_texts, tokenizer, model)
        
        print(len(document_embeddings))
        # Calculate similarity between queries and documents
        print("Calculating similarities...")
        similarities = cosine_similarity(query_embeddings, document_embeddings)
        
        # Get maximum similarity for each document (across all queries)
        max_similarities = similarities.max(axis=0)
        print(max_similarities)
        # Add similarity scores to dataframe
        df['similarity_score'] = max_similarities
        
        # Filter articles with similarity above threshold
        filtered_df = df[df['similarity_score'] > threshold]
        
        print(f"Filtered to {len(filtered_df)} articles with threshold {threshold}")
        return filtered_df
        
    except Exception as e:
        print(f"Error during embedding filtering: {e}")
        print(f"Exception details: {str(e)}")
        return pd.DataFrame(columns=df.columns)



def main():
    # Set the directory path where the markdown files are located
    data_dir = '/home/ubuntu/sayantan/ui/search/data'  # Adjust this path as needed
    
    # Load all articles
    print("Loading articles...")
    articles_df = load_articles(data_dir)
    print(f"Loaded {len(articles_df)} articles")
    articles_df.to_csv('output.csv', index=False)

    
    # Option 1: Filter with a specific threshold
    threshold = 0.2  # You can adjust this based on your needs
    filtered_df = filter_with_pubmedbert(articles_df, threshold)
    
    # # Inspect some of the filtered results
    # inspect_filtered_results(articles_df, filtered_df)
    
    # Get filtered PubMed IDs
    filtered_ids = filtered_df['pubmed_id'].tolist()
    
    # Save to JSON file
    output = {"filtered_articles": filtered_ids}
    with open('filtered_articles.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Successfully filtered {len(filtered_ids)} articles out of {len(articles_df)}")
    print(f"Results saved to filtered_articles.json")

if __name__ == "__main__":
    main()




# API Key : AIzaSyA991yvb37XuLvC24A5-WSOKlCP-P-pfDA