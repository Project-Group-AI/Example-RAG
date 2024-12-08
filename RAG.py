import requests
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Initialization of the embedding model and the FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, quick model
documents = [
    "The Golden Ball is an award given each year to the best football player.",
    "The 2024 Golden Ball was awarded to Rodri.",
    "Rodri is a Spanish football player playing as a midfielder.",
    "Rodri also plays for Manchester City in the Premier League.",
]

# Create embeddings for each document
doc_embeddings = embedding_model.encode(documents)

# Build a FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

#Step 2: Function to search for relevant documents
def find_relevant_docs(query, k=2):
    """
    Searches for relevant documents for a query.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0]]

# Step 3: Use Mistral to generate a response
def mistral_via_api(prompt):
    """
    Uses the Hugging Face API to generate text with Mistral.
    """
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    HF_TOKEN = HfFolder.get_token()
    if HF_TOKEN is None:
        return "Error: No tokens found. Log in with `huggingface-cli login`."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 5000,
            "temperature": 0.7,
            "top_k": 50,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error : {response.status_code} - {response.json()}"

# Étape 4 : RAG - Combiner retrieval et génération
def rag_pipeline(query, k=2):
    """
    Implements the RAG method by combining retrieval and generation.
    """
    # Retrieve relevant documents
    relevant_docs = find_relevant_docs(query, k)
    context = "\n".join(relevant_docs)
    
    # Build the prompt
    prompt = f"Context : {context}\n\nQuestion : {query}\n\nAnswer :"
    
    # Generate a response with Mistral
    return mistral_via_api(prompt)

# Example of use
if __name__ == "__main__":
    query = "Who is the 2024 Golden Ball and and who is he?"
    answer = rag_pipeline(query)
    print("Response generated:")
    print(answer)
