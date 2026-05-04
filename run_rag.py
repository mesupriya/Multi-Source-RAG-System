import os
import json
import re
import time
import numpy as np
import pandas as pd
import gradio as gr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

# 1. SETUP & DATA LOADING
print("Initializing Mahabharata RAG System...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

load_dotenv() # Load keys from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file")
    exit(1)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Load Data
JSON_FILE_PATH = 'Data.json'
try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} documents")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned_text'] = df['text'].apply(clean_text)

def tokenize_and_filter(text):
    tokens = word_tokenize(text.lower())
    important = {'krishna', 'arjuna', 'bhima', 'draupadi', 'yudhishthira', 'duryodhana', 'karna', 'drona', 'bhishma', 'dharma'}
    return [t for t in tokens if t not in STOP_WORDS or t in important]

tokenized_corpus = df['cleaned_text'].apply(tokenize_and_filter).tolist()

# 2. INDEXING
print("Building retrieval indices...")
# BM25
bm25_index = BM25Okapi(tokenized_corpus)

# FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDINGS_PATH = 'embeddings.npy'
FAISS_INDEX_PATH = 'faiss_index.bin'

if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Loading cached embeddings and FAISS index...")
    embeddings = np.load(EMBEDDINGS_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("Generating embeddings (this may take a minute)...")
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.astype('float32'))
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

# Retrieval Corpus (Metadata)
retrieval_corpus = []
for idx, row in df.iterrows():
    retrieval_corpus.append({
        'id': row['id'],
        'text': row['text'],
        'parva': row['parva'],
        'section': row['section']
    })

# 3. RETRIEVAL LOGIC
MAJOR_CHARACTERS = ['krishna', 'arjuna', 'yudhishthira', 'bhima', 'draupadi', 'duryodhana', 'karna', 'drona', 'bhishma', 'dhritarashtra']

def classify_query(query):
    query_lower = query.lower()
    entity_focused = any(char in query_lower for char in MAJOR_CHARACTERS)
    question_words = ['who', 'what', 'where', 'when', 'why', 'how']
    question_count = sum(1 for word in question_words if word in query_lower.split())
    has_conjunction = ' and ' in query_lower or ' or ' in query_lower
    multi_hop = question_count > 1 or (has_conjunction and '?' in query)
    
    if multi_hop: return 'multi-hop'
    if entity_focused: return 'entity-focused'
    return 'simple'

def retrieve_bm25(query, top_k=5):
    tokens = tokenize_and_filter(query)
    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [retrieval_corpus[i] for i in top_indices]

def retrieve_faiss(query, top_k=5):
    query_embedding = model.encode([query])[0]
    distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)
    return [retrieval_corpus[int(i)] for i in indices[0]]

def retrieve_hybrid(query, top_k=10):
    bm25_results = retrieve_bm25(query, top_k)
    faiss_results = retrieve_faiss(query, top_k)
    seen_ids = set()
    combined = []
    for doc in bm25_results + faiss_results:
        if doc['id'] not in seen_ids:
            seen_ids.add(doc['id'])
            combined.append(doc)
            if len(combined) >= top_k: break
    return combined

def retrieve_documents(query, query_type, top_k=5):
    if query_type == 'entity-focused': return retrieve_bm25(query, top_k)
    if query_type == 'simple': return retrieve_faiss(query, top_k)
    return retrieve_hybrid(query, top_k)

def decompose_query(query):
    if ' and ' in query.lower():
        parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
        return [p.strip() + ('?' if not p.strip().endswith('?') else '') for p in parts]
    return [query]

def multi_hop_rag_retrieval(query, max_hops=2, top_k=5):
    query_type = classify_query(query)
    sub_queries = decompose_query(query) if query_type == 'multi-hop' else [query]
    all_docs = []
    for hop, sub_query in enumerate(sub_queries[:max_hops]):
        docs = retrieve_documents(sub_query, 'simple' if hop > 0 else query_type, top_k)
        all_docs.extend(docs)
    return all_docs

def search_web(query, max_results=5):
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        return response['results']
    except Exception as e:
        print(f"Web search error: {e}")
        return []

# 4. CHATBOT FUNCTION
def chatbot_rag(message, history, use_web=False):
    print(f"Processing: {message}")
    query_type = classify_query(message)
    
    start_retrieval = time.time()
    if query_type == 'multi-hop':
        dataset_docs = multi_hop_rag_retrieval(message, top_k=5)[:3]
    else:
        dataset_docs = retrieve_documents(message, query_type, top_k=5)[:3]
    retrieval_time = time.time() - start_retrieval
    
    # Step 3: Web search
    web_docs = []
    web_time = 0
    if use_web:
        start_web = time.time()
        web_docs = search_web(message, max_results=3)
        web_time = time.time() - start_web
    
    dataset_ctx = ""
    for i, doc in enumerate(dataset_docs, 1):
        dataset_ctx += f"[Dataset-{i}: {doc['id']}, {doc['parva']}]\n{doc['text'][:700]}\n\n"
    
    web_ctx = ""
    if web_docs:
        for i, doc in enumerate(web_docs, 1):
            web_ctx += f"[Web-{i}: {doc['title']}]\nURL: {doc['url']}\n{doc['content']}\n\n"

    prompt = f"""You are a Mahabharata expert.
Answer using the provided dataset and web sources. Cite every fact with [Dataset-X] or [Web-X].
If the information is not in the dataset or web sources, say you don't know based on the current records.

QUESTION: {message}

DATASET:
{dataset_ctx}

WEB SOURCES:
{web_ctx}

ANSWER (with citations):"""

    start_gen = time.time()
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing Mahabharata information with citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        gen_time = time.time() - start_gen
        answer = response.choices[0].message.content
        
        # Metadata section
        answer += "\n\n" + "="*40 + "\n"
        answer += "SOURCES:\n"
        for i, doc in enumerate(dataset_docs, 1):
            answer += f"- [Dataset-{i}] {doc['id']} ({doc['parva']})\n"
        
        if web_docs:
            for i, doc in enumerate(web_docs, 1):
                answer += f"- [Web-{i}] {doc['title']} ({doc['url']})\n"
        
        answer += f"\nPerformance: {retrieval_time + web_time + gen_time:.2f}s"
        return answer
    except Exception as e:
        return f"Error: {e}"

# 5. GRADIO INTERFACE
with gr.Blocks(title="Mahabharata RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Mahabharata RAG System")
    gr.Markdown("Ask questions about the 18 Parvas of the Mahabharata. (Web search is enabled)")
    
    with gr.Row():
        use_web_checkbox = gr.Checkbox(label="Use Live Web Search (Tavily)", value=True)
    
    chatbot = gr.ChatInterface(
        fn=chatbot_rag,
        additional_inputs=[use_web_checkbox],
        examples=[["Who was Arjuna?"], ["What is dharma?"], ["How did Bhima defeat Hidimba?"]],
    )

if __name__ == "__main__":
    print("Launching Gradio...")
    share_url = demo.launch(share=True, prevent_thread_lock=True)
    print(f"\nSUCCESS! Your RAG system is live at:")
    print(f"Public URL: {demo.share_url}")
    print(f"Local URL: http://127.0.0.1:7860")
    
    # Keep it running
    import time
    while True:
        time.sleep(10)
