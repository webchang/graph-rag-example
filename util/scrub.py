"""
This module provides functions for cleaning and 
preprocessing documents using the unstructured library.
"""

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean

def clean_and_preprocess_documents(documents):
    """
    Cleans and preprocesses a list of documents using unstructured.

    This function partitions the HTML content of each document, cleans the text content,
    and updates the document content with the cleaned text.

    Parameters:
    documents (list): List of documents to clean and preprocess.

    Returns:
    list: List of cleaned and preprocessed documents.
    """
    cleaned_documents = []
    html_content = "<html><body><h1>Hello</h1><p>World!</p></body></html>"
    elements = partition_html(text=html_content)
    print(f"Found {len(elements)} elements in the document.")
    for doc in documents:
        print(f"Cleaning document: {doc.metadata.get('title', 'No Title')}")
        # Partition the HTML content
        # print(doc.page_content[:200])  # See what you're passing to partition_html
        # print(doc.page_content)  # See what you're passing to partition_html
        elements = partition_html(text=doc.page_content)
        print(f"Found {len(elements)} elements in the document.")
        # Clean the text content
        cleaned_text = clean(" ".join([element.text for element in elements]))
        # Update the document content with cleaned text
        doc.page_content = scrub(cleaned_text)
        cleaned_documents.append(doc)
        print(f"Cleaned document: {doc.metadata.get('title', 'No Title')} - Length: {len(doc.page_content)} characters")
    return cleaned_documents

def scrub(content):
    """
    Scrubs specific unwanted phrases from the content.

    This function replaces specific unwanted phrases in the content with an empty string.

    Parameters:
    content (str): The content to scrub.

    Returns:
    str: The scrubbed content.
    """
    content = content.replace("What's your", "")
    content = content.replace("Login to use TMDB's new rating system.", "")
    content = content.replace("Welcome to Vibes, TMDB's new rating system! For more information, visit the  contribution bible.", "")
    content = content.replace("Looks like we're missing the following data in en-US or en-US...", "")
    content = content.replace("Login to edit", "")
    content = content.replace("Login to report an issue", "")
    return content
