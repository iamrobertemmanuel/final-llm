# final-llm/vectordb_handler.py

# --- IMPORTANT: pysqlite3 patch MUST be at the very top ---
import os
import sys
__import__('pysqlite3') # Explicitly import pysqlite3
sys.modules['sqlite3'] = sys.modules['pysqlite3'] # Replace the default sqlite3 with pysqlite3
# -----------------------------------------------------------

# Now, it's safe to import chromadb and other libraries
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from utils import load_config
import numpy as np
import json

# Load configuration (assuming load_config() is defined in utils.py)
config = load_config()

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class VectorDB:
    def __init__(self):
        # Initialize ChromaDB PersistentClient.
        # This will use the patched sqlite3 due to the code at the top of the file.
        self.client = chromadb.PersistentClient(path=config["chromadb"]["chromadb_path"])
        self.collection = self.client.get_or_create_collection(
            name=config["chromadb"]["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )
        
        # Configure Gemini for embeddings
        # Ensure GEMINI_API_KEY is set in your config.json or environment variables
        genai.configure(api_key=config["gemini"]["api_key"])
        self.model = genai.GenerativeModel('embedding-001')

    def add_texts(self, texts):
        embeddings = []
        for text in texts:
            try:
                # Generate embeddings using Gemini's embedding model
                # Note: result.prompt_feedback.safety_ratings[0].probability is not the embedding.
                # The embedding is typically found in result.embedding or similar attribute.
                # Please verify the correct path to the embedding from Gemini API response.
                # For 'embedding-001', it's usually `response.embedding.values`.
                response = self.model.embed_content(model="embedding-001", content=text)
                embedding = response['embedding'] # Access the embedding values
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: '{text[:50]}...' - {str(e)}")
                # Use a zero vector as fallback if embedding generation fails
                # The embedding dimension for 'embedding-001' is 768
                embeddings.append([0.0] * 768) 
        
        # Add to ChromaDB
        # Ensure that the number of embeddings, documents, and ids match
        if embeddings and len(embeddings) == len(texts):
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=[f"doc_{i}" for i in range(len(texts))]
            )
        else:
            print("Warning: No embeddings generated or mismatch in lengths. No documents added to ChromaDB.")

    def similarity_search(self, query, k=4):
        # Generate query embedding
        try:
            # Correct way to get embedding for query
            response = self.model.embed_content(model="embedding-001", content=query)
            query_embedding = response['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            query_embedding = [0.0] * 768  # Fallback to zero vector
        
        # Search in ChromaDB
        # Note: query_embeddings expects a list of embeddings, even for a single query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Return results as Document objects
        # Ensure results["documents"] is not empty before accessing [0]
        if results and results["documents"] and results["documents"][0]:
            return [Document(page_content=doc) for doc in results["documents"][0]]
        else:
            return [] # Return empty list if no results found

def load_vectordb():
    # This function will now correctly initialize VectorDB using the patched sqlite3
    return VectorDB()

# --- SimpleVectorDB (if you intend to use this, it's a separate implementation) ---
# This class seems to be a fallback or alternative if ChromaDB is not used.
# It uses a local JSON file for storage and manual similarity calculation.
# It also needs the Gemini API key for embeddings.
class SimpleVectorDB:
    def __init__(self, db_path="chroma_db"): # Note: This db_path is for a JSON file, not Chroma's path
        self.db_path = db_path
        # Configure Gemini for embeddings (redundant if VectorDB is used, but necessary for SimpleVectorDB)
        genai.configure(api_key=config["gemini"]["api_key"])
        self.model = genai.GenerativeModel('embedding-001')
        os.makedirs(db_path, exist_ok=True)
        self.vectors_file = os.path.join(db_path, "vectors.json")
        self.load_db()

    def load_db(self):
        if os.path.exists(self.vectors_file):
            with open(self.vectors_file, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {"texts": [], "embeddings": []}
            self.save_db()

    def save_db(self):
        with open(self.vectors_file, 'w') as f:
            json.dump(self.db, f)

    def add_texts(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = self.model.embed_content(model="embedding-001", content=text)
                embedding = response['embedding']
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: '{text[:50]}...' - {str(e)}")
                embeddings.append([0.0] * 768)
        
        self.db["texts"].extend(texts)
        self.db["embeddings"].extend(embeddings)
        self.save_db()

    def similarity_search(self, query, k=4):
        if not self.db["texts"]:
            return []

        try:
            response = self.model.embed_content(model="embedding-001", content=query)
            query_embedding = response['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            query_embedding = [0.0] * 768

        similarities = []
        for doc_embedding in self.db["embeddings"]:
            # Ensure both embeddings are numpy arrays for dot product if they aren't already
            # Or convert to list and use sum(a*b for a,b in zip(query_embedding, doc_embedding))
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append(similarity)

        # Get indices of top k similar documents
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return Document objects
        return [Document(page_content=self.db["texts"][idx]) for idx in top_k_indices]

# This class seems to be a simple data structure, not a part of the DB logic.
class SimpleDocument:
    def __init__(self, page_content):
        self.page_content = page_content
