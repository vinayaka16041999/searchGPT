import os
from serpapi import GoogleSearch
from elasticsearch import Elasticsearch
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor

# API Keys from environment variables
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')

# Initialize Elasticsearch
es = Elasticsearch(ELASTICSEARCH_URL)

def search_web(query):
    """
    Perform a web search using SerpAPI.
    
    :param query: The search query string
    :return: List of search results
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 10  # Number of results to retrieve
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get('organic_results', [])

def fetch_and_index_page(url):
    """
    Fetch a web page, extract text content, and index it in Elasticsearch.
    
    :param url: URL of the page to fetch
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            
            # Normalize and clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Index the content
            es.index(index='web_pages', document={
                'url': url,
                'content': text[:10000]  # Limit to 10k characters for simplicity
            })
    except Exception as e:
        print(f"Failed to fetch or index {url}: {e}")

def index_search_results(results):
    """
    Index the search results into Elasticsearch for later retrieval or further processing.
    
    :param results: List of search result dictionaries
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        for result in results:
            executor.submit(fetch_and_index_page, result['link'])

def query_elasticsearch(query):
    """
    Query Elasticsearch for content matching the query.
    
    :param query: The query to search for
    :return: List of matching documents
    """
    search_body = {
        "query": {
            "match": {
                "content": query
            }
        }
    }
    results = es.search(index="web_pages", body=search_body)
    return [hit['_source'] for hit in results['hits']['hits']]

def search_engine(query):
    """
    Perform a comprehensive search combining web results with indexed content.
    
    :param query: User's query string
    :return: Formatted search results
    """
    # Fetch fresh results from web
    fresh_results = search_web(query)
    index_search_results(fresh_results[:3])  # Only index top 3 for this example

    # Search indexed content
    indexed_results = query_elasticsearch(query)
    
    # Combine results
    combined_results = []
    for result in fresh_results:
        combined_results.append({
            'title': result['title'],
            'link': result['link'],
            'snippet': result['snippet']
        })
    
    for doc in indexed_results:
        combined_results.append({
            'title': doc['url'],
            'link': doc['url'],
            'snippet': doc['content'][:200] + '...'
        })

    return combined_results

if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search_engine(query)
    for result in results[:10]:  # Display top 10 results
        print(f"Title: {result['title']}")
        print(f"URL: {result['link']}")
        print(f"Snippet: {result['snippet']}")
        print()