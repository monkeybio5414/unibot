import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from .query_rewriter import QueryRewriter
import logging

class SemanticSearch:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        try:
            # Use local model files
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder='./models/sentence-transformers'
            )
            print("Successfully loaded model files")
            
            # Initialize query rewriter
            self.query_rewriter = QueryRewriter()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.index = None
        self.documents = None
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def build_index(self, documents: List[str], embeddings: np.ndarray = None):
        """Build FAISS index from documents"""
        self.documents = documents
        
        if embeddings is None:
            # Generate embeddings using Sentence Transformer
            embeddings = self.sentence_transformer.encode(
                documents,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True
            )
        
        # Initialize FAISS index with the correct dimension
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Add vectors to the index
        self.index.add(embeddings.astype('float32'))
        
        # Build TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
    def _get_keyword_score(self, query: str, doc: str) -> float:
        """Calculate keyword-based similarity score"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return 0.0
            
        # Transform query to TF-IDF vector
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity
        doc_idx = self.documents.index(doc)
        doc_vec = self.tfidf_matrix[doc_idx]
        
        # Calculate similarity
        similarity = cosine_similarity(query_vec, doc_vec)[0][0]
        return similarity
    
    def _get_semantic_score(self, query: str, doc: str) -> float:
        """Calculate semantic similarity score"""
        # Generate embeddings
        query_embedding = self.sentence_transformer.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        doc_embedding = self.sentence_transformer.encode(
            [doc],
            normalize_embeddings=True
        )[0]
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding)
        return similarity
    
    def _get_combined_score(self, query: str, doc: str, alpha: float = 0.7) -> float:
        """Calculate combined score from semantic and keyword search"""
        semantic_score = self._get_semantic_score(query, doc)
        keyword_score = self._get_keyword_score(query, doc)
        
        # Combine scores with weighted average
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        return combined_score
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents available in the index")
        
        # Understand query intent and rewrite
        intent_analysis = self.query_rewriter.understand_intent(query)
        rewritten_queries = self.query_rewriter.rewrite_query(query)
        
        logging.info(f"Query intent: {intent_analysis}")
        logging.info(f"Rewritten queries: {rewritten_queries}")
        
        # Search with all query variations
        all_results = []
        seen_docs = set()
        
        for q in rewritten_queries:
            # Generate query embedding
            query_embedding = self.sentence_transformer.encode(
                [q],
                normalize_embeddings=True
            )[0]
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k * 5  # Get more results for each query variation
            )
            
            # Process results
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    if doc not in seen_docs:
                        similarity = 1 / (1 + distance)  # Convert distance to similarity score
                        
                        # Apply intent-specific boosts
                        if intent_analysis['intent'] == 'location':
                            # Boost for address information
                            if ('jalan broga' in doc.lower() and 
                                '43500' in doc.lower() and 
                                'semenyih' in doc.lower()):
                                similarity *= 2.0
                            # Boost for location-related content
                            elif any(word in doc.lower() for word in ['address', 'location', 'located']):
                                similarity *= 1.5
                            # Penalty for facility-only content
                            elif all(word in doc.lower() for word in ['facility', 'facilities']):
                                similarity *= 0.5
                        
                        all_results.append((doc, similarity))
                        seen_docs.add(doc)
        
        # Sort all results by similarity
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return all_results[:k]
    
    def save_index(self, path: str):
        """Save the FAISS index and documents to disk"""
        if self.index is None:
            raise ValueError("No index to save.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save documents
        with open(f"{path}_documents.txt", 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(doc + '\n')
        
        # Save TF-IDF vectorizer and matrix
        import pickle
        with open(f"{path}_tfidf.pkl", 'wb') as f:
            pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix), f)
    
    def load_index(self, path: str):
        """Load the FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load documents
        with open(f"{path}_documents.txt", 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f.readlines()]
        
        # Load TF-IDF vectorizer and matrix
        import pickle
        with open(f"{path}_tfidf.pkl", 'rb') as f:
            self.tfidf_vectorizer, self.tfidf_matrix = pickle.load(f) 