import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

# Load GPT-Neo for response generation
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Load Sentence Transformer for dense embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or "all-mpnet-base-v2" for better performance

def index_documents(docs):
    global faiss_index, bm25, doc_texts, tokenized_docs

    doc_texts = docs
    # Generate embeddings
    embeddings = embedder.encode(docs)

    # Create FAISS index (L2 distance)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to FAISS index
    faiss_index.add(embeddings)

    # Step 2: BM25 Index
    # Tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

docs = [
    "OpenAI released ChatGPT in November 2022.",
    "Elon Musk founded SpaceX in 2002.",
    "Python is a popular programming language known for its simplicity.",
    "The Earth revolves around the Sun.",
    "Artificial Intelligence is transforming the world."
]

# Initialize BM25 with tokenized documents
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

def sparse_retrieval(query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)
    faiss_scores = 1 / (1 + distances[0])  # Convert L2 distance to similarity
    
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    # Normalize scores
    faiss_scores /= faiss_scores.max()  # Normalize to 0–1 range
    bm25_scores = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    
    # Combine scores (alpha controls contribution of each)
    combined_scores = alpha * faiss_scores + (1 - alpha) * bm25_scores
    
    # Rank based on combined scores
    ranked_indices = np.argsort(combined_scores)[::-1]
    
    results = [documents[i] for i in ranked_indices[:k]]
    scores = [combined_scores[i] for i in ranked_indices[:k]]
    
    return results, scores

def hybrid_search(query, k=5, alpha=0.5):
    sparse_results = sparse_retrieval(query, k)
    
    # Create a combined score dictionary
    combined_scores = {}

    for text, score in sparse_results:
        combined_scores[text] = alpha * score
    
    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k]

def generate_answer(query):
    # Retrieve top hybrid search results
    top_results = hybrid_search(query, k=3)
    context = "\n".join([result[0] for result in top_results])

    input_text = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=200, num_return_sequences=1)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
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

def main():
    st.title("Apple Inc. Financial RAG Chatbot with Hybrid Search")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    index_documents(docs=docs)

    st.success("Document Indexes built successfully.")

    st.subheader("1. Ask a Question")
    user_query = st.text_input("Enter your question about Apple's financials")

    if st.button("Submit Query"):
        response_text = generate_answer(user_query)
        st.session_state.conversation_history.append({"user": user_query, "assistant": response_text})
        st.markdown(f"**Answer:** {response_text}")

        # Calculate confidence score
        top_results = hybrid_search(user_query, k=3)
        context = "\n".join([result[0] for result in top_results])
        confidence = calculate_confidence_score(user_query, context, response_text)

        st.markdown(f"**Confidence Score:** {confidence:.4f}")

if __name__ == "__main__":
    main()
