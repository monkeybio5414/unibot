import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import torch

class SemanticSearch:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        try:
            # Use local model files
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder='./models/sentence-transformers'  # Use local cache directory
            )
            print("Successfully loaded model files")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.index = None
        self.documents = None
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
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
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents available in the index")
            
        # Generate query embedding
        query_embedding = self.sentence_transformer.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # Search in FAISS index with more results for better filtering
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k * 10  # Get more results for better filtering
        )
        
        # Return results with documents and similarity scores
        results = []
        seen_docs = set()  # Track unique documents
        
        # Location-specific keywords and variations
        location_keywords = [
            'jalan', 'semenyih', 'selangor', 'malaysia', 'address',
            'location', 'located', 'place', 'where', 'campus',
            'unm', 'university of nottingham malaysia'
        ]
        address_keywords = [
            'jalan broga', '43500', 'semenyih', 'selangor',
            'address', 'location', 'where'
        ]
        distance_keywords = [
            'distance', 'far', 'kilometer', 'km', 'away',
            'from kuala lumpur', 'to kl', 'between'
        ]
        facility_keywords = [
            'schools', 'business', 'education', 'science', 'engineering',
            'sports', 'library', 'accommodation', 'facilities'
        ]
        
        # Log initial results
        print(f"Initial search returned {len(indices[0])} results")
        
        # Check if this is a location-related query
        is_location_query = (
            query.lower().startswith('where') or
            'location' in query.lower() or
            'address' in query.lower() or
            'unm' in query.lower() or
            any(word in query.lower() for word in location_keywords) or
            any(word in query.lower() for word in address_keywords) or
            any(word in query.lower() for word in distance_keywords)
        )
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Ensure valid index
                doc = self.documents[idx]
                if doc not in seen_docs:  # Only add unique documents
                    similarity = 1 / (1 + distance)  # Convert distance to similarity score
                    
                    # Enhanced location query handling
                    if is_location_query:
                        # Check for location-specific information
                        has_location = any(loc in doc.lower() for loc in location_keywords)
                        has_address = any(word in doc.lower() for word in address_keywords)
                        has_distance = any(word in doc.lower() for word in distance_keywords)
                        has_facilities = any(fac in doc.lower() for fac in facility_keywords)
                        
                        # Very strong boost for documents with complete address
                        if has_location and has_address:
                            similarity *= 5.0  # Increased from 3.0
                            print(f"Found complete address: {doc[:100]}...")
                        # Strong boost for location information
                        elif has_location:
                            similarity *= 3.0  # Increased from 2.0
                            print(f"Found location info: {doc[:100]}...")
                        # Medium boost for address information
                        elif has_address:
                            similarity *= 2.0  # Increased from 1.5
                            print(f"Found address info: {doc[:100]}...")
                        # Medium boost for distance information
                        elif has_distance:
                            similarity *= 2.0  # Increased from 1.5
                            print(f"Found distance info: {doc[:100]}...")
                        # Strong penalty for facility-only descriptions
                        elif has_facilities:
                            similarity *= 0.2  # Decreased from 0.5
                            print(f"Found facility-only info: {doc[:100]}...")
                    
                    # Add result if it meets minimum similarity threshold
                    if similarity > 0.3:  # Lowered threshold to get more results
                        results.append((doc, similarity))
                        seen_docs.add(doc)
                        print(f"Added result with similarity {similarity:.3f}: {doc[:100]}...")
                    
                    # Stop if we have enough unique results
                    if len(results) >= k:
                        break
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # For location queries, prioritize location information
        if is_location_query:
            # First try to find documents with complete address
            address_results = [(doc, score) for doc, score in results 
                             if any(loc in doc.lower() for loc in location_keywords) and 
                             any(addr in doc.lower() for addr in address_keywords)]
            if address_results:
                # Always include distance information if available
                distance_info = [(doc, score) for doc, score in results 
                               if any(word in doc.lower() for word in distance_keywords)]
                if distance_info:
                    return [address_results[0], distance_info[0]]  # Return both address and distance
                return address_results[:2]  # Return top 2 address results
            
            # Then try to find documents with location information
            location_results = [(doc, score) for doc, score in results 
                              if any(loc in doc.lower() for loc in location_keywords)]
            if location_results:
                return location_results[:2]
            
            # Then try to find documents with distance information
            distance_results = [(doc, score) for doc, score in results 
                              if any(word in doc.lower() for word in distance_keywords)]
            if distance_results:
                return distance_results[:2]
            
            # If no location results but we have other results, return them
            if results:
                return results[:2]
            return []
            
        return results
    
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
    
    def load_index(self, path: str):
        """Load the FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load documents
        with open(f"{path}_documents.txt", 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f.readlines()] 