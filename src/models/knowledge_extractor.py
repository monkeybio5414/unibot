import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeExtractor:
    def __init__(self):
        logger.info("Initializing KnowledgeExtractor...")
        
        # Set environment variables to prevent downloads
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # Initialize SentenceTransformer directly with the model path
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder='./models/sentence-transformers'
            )
            
            logger.info("KnowledgeExtractor initialized successfully with local model")
            
        except Exception as e:
            logger.error(f"Error initializing KnowledgeExtractor: {str(e)}")
            raise
        
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.word2vec = None
        
    def process_documents(self, documents: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """Process documents using multiple embedding methods"""
        logger.info(f"Processing {len(documents)} documents...")
        
        # TF-IDF embeddings
        logger.info("Computing TF-IDF embeddings...")
        tfidf_embeddings = self.tfidf.fit_transform(documents)
        
        # Use sentence transformer for embeddings (same as semantic search)
        logger.info("Computing sentence transformer embeddings...")
        embeddings = self.sentence_transformer.encode(
            documents,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True
        )
        
        # Word2Vec embeddings (only if needed)
        logger.info("Computing Word2Vec embeddings...")
        word2vec_embeddings = self._get_word2vec_embeddings(documents)
        
        logger.info("Document processing completed")
        return {
            'tfidf': tfidf_embeddings,
            'bert': embeddings,  # Using sentence transformer embeddings
            'word2vec': word2vec_embeddings,
            'documents': documents
        }
    
    def _get_word2vec_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate Word2Vec embeddings for documents"""
        # Train Word2Vec model if not already trained
        if self.word2vec is None:
            logger.info("Training Word2Vec model...")
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.word2vec = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1)
        
        # Generate document embeddings
        logger.info("Generating Word2Vec embeddings...")
        embeddings = []
        for i, doc in enumerate(documents):
            if i % 1000 == 0:
                logger.info(f"Processing Word2Vec document {i}/{len(documents)}")
            words = doc.lower().split()
            word_vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
                embeddings.append(doc_vector)
            else:
                embeddings.append(np.zeros(100))
        return np.array(embeddings)
    
    def save_knowledge_base(self, knowledge_base: Dict[str, Any], path: str):
        """Save the processed knowledge base to disk"""
        logger.info(f"Saving knowledge base to {path}")
        # Convert numpy arrays to lists for JSON serialization
        serializable_kb = {
            'documents': knowledge_base['documents'],
            'tfidf': knowledge_base['tfidf'].toarray().tolist(),
            'bert': knowledge_base['bert'].tolist(),
            'word2vec': knowledge_base['word2vec'].tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(serializable_kb, f)
        logger.info("Knowledge base saved successfully")
    
    def load_knowledge_base(self, path: str) -> Dict[str, Any]:
        """Load the knowledge base from disk"""
        logger.info(f"Loading knowledge base from {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        return {
            'documents': data['documents'],
            'tfidf': np.array(data['tfidf']),
            'bert': np.array(data['bert']),
            'word2vec': np.array(data['word2vec'])
        } 