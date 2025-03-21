###############################################################################
# IMPORT PACKAGES
###############################################################################
import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF for text extraction
from pdf2image import convert_from_path
import camelot
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

###############################################################################
# Load GPT-Neo for response generation
###############################################################################
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

###############################################################################
# Load Sentence Transformer for dense embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or "all-mpnet-base-v2" for better performance
###############################################################################

###############################################################################
# Financial Reports loading
###############################################################################
def get_financial_reports(folder_path):
    """Gets financial report file paths of a given folder"""
    file_list = os.listdir(folder_path)
    financial_report_paths = []
    for file in file_list:
        file_path = os.path.join(financial_reports_path, file)
        financial_report_paths.append(file_path)
    return financial_report_paths

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    full_text = [page.get_text("text") for page in doc]
    return "\n".join(full_text)

def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using Camelot"""
    tables = camelot.read_pdf(pdf_path, pages="all")
    extracted_tables = [t.df for t in tables]
    return extracted_tables

def convert_table_to_text(table_df):
    """Convert a pandas DataFrame to structured text"""
    headers = table_df.iloc[0].tolist()
    rows = [", ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers))) for _, row in table_df.iloc[1:].iterrows()]
    return " ".join(rows)

def process_pdf(pdf_path):
    """Extract text and tables, then convert tables into text"""
    raw_text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)

    table_texts = [convert_table_to_text(table) for table in tables]
    full_text = raw_text + "\n".join(table_texts)

    return full_text

def chunk_text(document_text, chunk_size=500, chunk_overlap=50):
    """Generates chunks of given size with ovelap using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Wrap the text in a Document object
    doc = Document(page_content=document_text)
    chunk_documents = text_splitter.split_documents([doc])
    chunk_texts = []
    for chunk in chunk_documents:
      chunk_texts.append(chunk.page_content)
    return chunk_texts
    
def preprocess_files_data(file_paths):
    """Extracts chunks of each document"""
    all_documents = []
    for file_path in file_paths:
      file_text = process_pdf(file_path)
      chunks = chunk_text(file_text, chunk_size=200)
      for c in chunks:
          if len(c.strip()) > 0:
              all_documents.append(c.strip())
    
    return all_documents

def index_documents(docs):
    """Create embeddings and stores in Faiss"""
    global faiss_index, bm25, doc_texts, tokenized_docs

    doc_texts = docs
    
    # Generate embeddings
    embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

    # Create FAISS index (L2 distance)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to FAISS index
    faiss_index.add(embeddings)

    # Step 2: BM25 Index
    # Tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

current_path = os.getcwd()
financial_reports_path = os.path.join(current_path, "data")
financial_report_paths = get_financial_reports(financial_reports_path)
docs = preprocess_files_data(financial_report_paths)

def sparse_retrieval(query, k=5):
    # Generate sparse BM25 vector for the query
    tokenized_query = query.lower().split()
    query_vector = np.array(bm25.get_scores(tokenized_query), dtype='float32').reshape(1, -1)

    # Ensure dimension matches FAISS index
    if query_vector.shape[1] < faiss_index.d:
        padding = np.zeros((query_vector.shape[0], faiss_index.d - query_vector.shape[1]), dtype='float32')
        query_vector = np.concatenate((query_vector, padding), axis=1)
        
    # Normalize for cosine similarity (if using IndexFlatIP)
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

    # Search FAISS index for top-k matches
    distances, indices = faiss_index.search(query_vector, k)
    
    # Convert L2 distances to similarity scores
    scores = 1 / (1 + distances[0])
    
    # Return top documents
    results = [docs[i] for i in indices[0]]
    return results, scores

def hybrid_search(query, k=5, alpha=0.5):
    results, scores = sparse_retrieval(query, k)
    
    # Create a combined score dictionary
    combined_scores = {}

    for i, (result, score) in enumerate(zip(results, scores)):
        combined_scores[result] = alpha * score
    
    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k]

def generate_answer(query, conversation_history):
    history_str = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in conversation_history[-3:]]
    )
    # Retrieve top hybrid search results
    top_results = hybrid_search(query, k=3)
    context_str = "\n".join([result[0] for result in top_results])

    prompt = (
        f"{history_str}\n"
        f"User: {query}\n\n"
        f"Relevant Apple Filings:\n{context_str}\n\n"
        f"Assistant:"
    )

    #input_text = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    output = model.generate(**input_ids, 
                            max_new_tokens=300, 
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
    
    response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

def calculate_confidence(query, context):
    query_embedding = embedder.encode(query).reshape(1, -1)
    context_embedding = embedder.encode(context).reshape(1, -1)

    # Cosine similarity between query and context
    similarity_score = cosine_similarity(query_embedding, context_embedding)[0][0]

    return (similarity_score + 1) / 2  # Normalize to 0–1

def calculate_bm25_score(query):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    return np.mean(scores) / max(scores)  # Normalize to 0–1

def calculate_log_prob(input_text, response):
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = tokenizer(response, return_tensors="pt").input_ids

    # Fix size mismatch
    if labels.size(1) != inputs.input_ids.size(1):
        labels = F.pad(labels, (0, inputs.input_ids.size(1) - labels.size(1)), value=tokenizer.pad_token_id)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, labels=labels)
        log_prob = -outputs.loss.item()
    
    return np.exp(log_prob)  # Convert log-prob to probability

def calculate_confidence_score(query, context, response):
    similarity_score = calculate_confidence(query, context)
    bm25_score = calculate_bm25_score(query)
    log_prob_score = calculate_log_prob(context, response)

    # Weighted combination
    confidence_score = (0.4 * similarity_score) + (0.3 * bm25_score) + (0.3 * log_prob_score)
    return round(confidence_score, 4)

BANNED_WORDS = ["politics", "personal life", "health", "religion", "hack", "attack", "malware", "exploit", "harm"]
def is_relevant_query(query):
    """Checks if the query is finance-related using NER and regex filtering."""
    
    # Check for query length
    query_tokens = query.lower().split()
    if len(query_tokens) > 300:
        return False, "Blocked: Too many tokens (>300) with the query. Please check."
    
    # Check for banned words
    if any(word in query.lower() for word in BANNED_WORDS):
        return False, "Blocked: Question contains restricted topics."

    return True, "Query is valid."

def main():
    st.title("Apple Inc. Financial RAG Chatbot with Hybrid Search")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    index_documents(docs=docs)

    st.success("Document Indexes built successfully.")

    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question about Apple's financials")

    if st.button("Submit Query"):
        # Input guide-rail check
        is_relevant, msg = is_relevant_query(user_query)
        print('is_relevant_query', is_relevant)
        print('msg', msg)
        if not is_relevant:
            st.warning(f"Sorry I cannot answer your question since: {msg}")
        else:
            #Generate response and write to conversation history
            response_text = generate_answer(user_query, st.session_state.conversation_history)
            st.session_state.conversation_history.append({"user": user_query, "assistant": response_text})
            st.markdown(f"**Answer:** {response_text}")

            # Calculate confidence score
            top_results = hybrid_search(user_query, k=3)
            context = "\n".join([result[0] for result in top_results])
            confidence = calculate_confidence_score(user_query, context, response_text)

            st.markdown(f"**Confidence Score:** {confidence:.4f}")

    st.subheader("Testing & Validation")
    st.write("Try queries like:")
    st.write("- 'What is the total revenue mentioned in the report for 2023?'")
    st.write("- 'What is the Cash paid for interest in 2024?'")
    st.write("- 'Who are Apple’s executive officers?' (non-financial but still in the 10-K)")
if __name__ == "__main__":
    main()
