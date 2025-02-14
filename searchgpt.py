import os
import openai
from serpapi import GoogleSearch
from elasticsearch import Elasticsearch
import spacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Environment Variables for API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Elasticsearch Client
es_client = Elasticsearch(ELASTICSEARCH_URL)

# NLP for query understanding
nlp = spacy.load("en_core_web_sm")

# Fine-tuned model for question-answering (example, replace with your model)
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get('organic_results', [])

def index_to_elasticsearch(results, query):
    for result in results:
        es_client.index(index="search_results", document={
            "title": result['title'],
            "link": result['link'],
            "snippet": result['snippet'],
            "query": query
        })

def retrieve_from_elasticsearch(query):
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "snippet"]
            }
        }
    }
    res = es_client.search(index="search_results", body=search_body)
    return [hit['_source'] for hit in res['hits']['hits']]

def enhanced_query_understanding(query):
    doc = nlp(query)
    # Extract entities, verbs, etc., to understand query intent better
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {'query': query, 'entities': entities}

def query_gpt_with_context(query_info, search_results):
    context = "\n".join([f"{result['title']}: {result['snippet']}" for result in search_results[:3]])
    prompt = f"Given the context:\n{context}\n\nAnswer the question considering: {query_info['entities']}\nQuery: {query_info['query']}"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def search_gpt(query):
    query_info = enhanced_query_understanding(query)
    web_results = search_web(query)
    
    # Index new results if desired
    index_to_elasticsearch(web_results, query)
    
    # Retrieve possibly cached or enriched results from Elasticsearch
    es_results = retrieve_from_elasticsearch(query)
    if es_results:
        results = es_results
    else:
        results = web_results
    
    # Use QA model for more precise answers
    qa_answer = qa_pipeline({
        'question': query,
        'context': "\n".join([result['snippet'] for result in results[:3]])
    })

    # Get LLM response with enhanced context
    gpt_answer = query_gpt_with_context(query_info, results)

    return f"QA Model Answer: {qa_answer['answer']}\n\nGPT Answer: {gpt_answer}\n\nSearch Results:\n" + \
           "\n".join([f"- {result['title']}\n  URL: {result['link']}" for result in results[:3]])

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    print(search_gpt(user_query))