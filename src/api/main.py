from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import logging
import numpy as np
import time
import uuid
import socket
import signal
import sys
import psutil
import uvicorn
from datetime import datetime

from ..models.knowledge_extractor import KnowledgeExtractor
from ..models.semantic_search import SemanticSearch
from ..models.nlg import NLG
from ..models.sentiment_analyzer import SentimentAnalyzer

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set logging levels for specific modules
logging.getLogger('src.models.knowledge_extractor').setLevel(logging.DEBUG)
logging.getLogger('src.models.nlg').setLevel(logging.DEBUG)
logging.getLogger('src.api.main').setLevel(logging.DEBUG)

app = FastAPI(title="Intelligent Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
knowledge_extractor = KnowledgeExtractor()
semantic_search = SemanticSearch()
nlg = NLG()
sentiment_analyzer = SentimentAnalyzer()

# Load knowledge base from JSONL files
def load_knowledge_base():
    try:
        logger.info("Starting knowledge base loading...")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if cached embeddings exist
        embeddings_cache = os.path.join(cache_dir, "embeddings.npz")
        documents_cache = os.path.join(cache_dir, "documents.json")
        
        if os.path.exists(embeddings_cache) and os.path.exists(documents_cache):
            logger.info("Loading cached embeddings and documents...")
            # Load cached embeddings
            embeddings_data = np.load(embeddings_cache)
            embeddings = embeddings_data['embeddings']
            
            # Load cached documents
            with open(documents_cache, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents from cache")
            
            # Build search index with cached data
            semantic_search.build_index(
                documents=documents,
                embeddings=embeddings
            )
            logger.info("Built search index from cached data")
            return
        
        # If no cache exists, load and process documents
        documents = []
        
        # Load pretrain data
        pretrain_path = os.path.join(data_dir, "pretrain_hq.jsonl")
        if os.path.exists(pretrain_path):
            logger.info(f"Loading pretrain data from {pretrain_path}")
            with open(pretrain_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    documents.append(data.get('text', ''))
            logger.info(f"Loaded {len(documents)} documents from pretrain data")
        else:
            logger.warning(f"Pretrain data file not found at {pretrain_path}")
        
        # Load SFT data
        sft_path = os.path.join(data_dir, "sft_data.jsonl")
        if os.path.exists(sft_path):
            logger.info(f"Loading SFT data from {sft_path}")
            with open(sft_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    for conv in data.get('conversations', []):
                        if conv.get('role') == 'assistant':
                            documents.append(conv.get('content', ''))
            logger.info(f"Loaded {len(documents)} total documents after SFT data")
        else:
            logger.warning(f"SFT data file not found at {sft_path}")
        
        # Process documents and build search index
        if documents:
            logger.info("Processing documents with knowledge extractor...")
            knowledge_base = knowledge_extractor.process_documents(documents)
            
            # Cache the embeddings and documents
            logger.info("Caching embeddings and documents...")
            np.savez(embeddings_cache, embeddings=knowledge_base['bert'])
            with open(documents_cache, 'w', encoding='utf-8') as f:
                json.dump(documents, f)
            
            logger.info("Building semantic search index...")
            semantic_search.build_index(
                documents=documents,
                embeddings=knowledge_base['bert']
            )
            logger.info(f"Successfully loaded {len(documents)} documents into knowledge base")
        else:
            logger.warning("No documents were loaded into the knowledge base")
            
    except Exception as e:
        logger.error(f"Error loading knowledge base: {str(e)}", exc_info=True)
        raise

# Load knowledge base on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_knowledge_base()
    except Exception as e:
        logger.error(f"Failed to load knowledge base during startup: {str(e)}")
        # Continue startup even if knowledge base loading fails
        pass

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class Query(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    answer: str
    confidence: float
    source: str
    is_generated: bool
    sentiment_analysis: Dict[str, Any]
    reasoning_steps: List[str]
    metadata: Dict[str, Any]

@app.post("/chat", response_model=Response)
async def chat(query: Query):
    try:
        # Start timing the request
        request_start_time = time.time()
        
        # Log the incoming query with a unique identifier
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id} - Received query: {query.text}")
        
        # Check if query should be redirected to human support
        sentiment_start_time = time.time()
        should_redirect, sentiment_analysis = sentiment_analyzer.should_redirect_to_human(query.text)
        sentiment_time = time.time() - sentiment_start_time
        logger.info(f"Request {request_id} - Sentiment analysis completed in {sentiment_time:.2f}s")
        logger.info(f"Request {request_id} - Should redirect to human: {should_redirect}")
        logger.debug(f"Request {request_id} - Sentiment analysis: {sentiment_analysis}")
        
        if should_redirect:
            response_data = Response(
                answer="I apologize, but I think it would be better to connect you with a human support representative.",
                confidence=1.0,
                source="human_redirect",
                is_generated=False,
                sentiment_analysis=sentiment_analysis,
                reasoning_steps=["Query requires human intervention", "Redirecting to human support"],
                metadata={
                    'bert_score': 0.0,
                    'context_score': 0.0,
                    'relevance_score': 0.0,
                    'processing_time': time.time() - request_start_time
                }
            )
            logger.info(f"Request {request_id} - Redirecting to human support. Total time: {time.time() - request_start_time:.2f}s")
            return response_data
        
        # Get search results
        search_start_time = time.time()
        search_results = semantic_search.search(query.text, k=10)
        search_time = time.time() - search_start_time
        logger.info(f"Request {request_id} - Search completed in {search_time:.2f}s")
        logger.info(f"Request {request_id} - Found {len(search_results)} search results")
        
        # Log detailed search results
        for i, result in enumerate(search_results):
            logger.debug(f"Request {request_id} - Search result {i+1}:")
            logger.debug(f"Text: {result[0][:200]}...")
            logger.debug(f"Similarity score: {result[1]:.4f}")
        
        if not search_results:
            logger.warning(f"Request {request_id} - No search results found")
            response_data = Response(
                answer="I apologize, but I couldn't find any relevant information to answer your question.",
                confidence=0.0,
                source="no_results",
                is_generated=False,
                sentiment_analysis=sentiment_analyzer.analyze_sentiment(query.text)[1],
                reasoning_steps=["No relevant information found"],
                metadata={
                    'bert_score': 0.0,
                    'context_score': 0.0,
                    'relevance_score': 0.0,
                    'processing_time': time.time() - request_start_time
                }
            )
            logger.info(f"Request {request_id} - Returning no results response. Total time: {time.time() - request_start_time:.2f}s")
            return response_data
        
        # Generate response using NLG
        try:
            generation_start_time = time.time()
            nlg_response = nlg.generate_response(query.text, search_results)
            generation_time = time.time() - generation_start_time
            logger.info(f"Request {request_id} - Response generation completed in {generation_time:.2f}s")
            logger.debug(f"Request {request_id} - Generated response: {nlg_response}")
            
            if not nlg_response or nlg_response.isspace():
                logger.warning(f"Request {request_id} - Empty response generated")
                response_data = Response(
                    answer="I apologize, but I encountered an error while generating the response. Please try again.",
                    confidence=0.0,
                    source="empty_response",
                    is_generated=False,
                    sentiment_analysis=sentiment_analyzer.analyze_sentiment(query.text)[1],
                    reasoning_steps=["Empty response generated"],
                    metadata={
                        'bert_score': search_results[0][1] if search_results else 0.0,
                        'context_score': 0.0,
                        'relevance_score': 0.0,
                        'processing_time': time.time() - request_start_time,
                        'search_time': search_time,
                        'generation_time': generation_time
                    }
                )
                logger.info(f"Request {request_id} - Returning empty response error. Total time: {time.time() - request_start_time:.2f}s")
                return response_data
                
            # Get sentiment analysis for response
            response_sentiment_start_time = time.time()
            response_sentiment_score, response_sentiment = sentiment_analyzer.analyze_sentiment(nlg_response)
            response_sentiment_time = time.time() - response_sentiment_start_time
            logger.info(f"Request {request_id} - Response sentiment analysis completed in {response_sentiment_time:.2f}s")
            
            # Calculate final confidence score
            confidence_score = min(
                search_results[0][1] if search_results else 0.0,
                response_sentiment_score
            )
            
            # Prepare final response
            response_data = Response(
                answer=nlg_response,
                confidence=confidence_score,
                source="nlg",
                is_generated=True,
                sentiment_analysis={
                    'query_sentiment': sentiment_analysis,
                    'response_sentiment': response_sentiment,
                    'response_sentiment_score': response_sentiment_score
                },
                reasoning_steps=nlg.generate_reasoning(query.text, search_results),
                metadata={
                    'bert_score': search_results[0][1] if search_results else 0.0,
                    'context_score': confidence_score,
                    'relevance_score': search_results[0][1] if search_results else 0.0,
                    'processing_time': time.time() - request_start_time,
                    'search_time': search_time,
                    'generation_time': generation_time,
                    'sentiment_time': sentiment_time + response_sentiment_time
                }
            )
            
            # Log final timing and response details
            total_time = time.time() - request_start_time
            logger.info(f"Request {request_id} - Request completed successfully in {total_time:.2f}s")
            logger.info(f"Request {request_id} - Final confidence score: {confidence_score:.4f}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Request {request_id} - Error generating response: {str(e)}", exc_info=True)
            response_data = Response(
                answer="I apologize, but I encountered an error while generating the response. Please try again.",
                confidence=0.0,
                source="error",
                is_generated=False,
                sentiment_analysis=sentiment_analyzer.analyze_sentiment(query.text)[1],
                reasoning_steps=["Error in response generation"],
                metadata={
                    'bert_score': search_results[0][1] if search_results else 0.0,
                    'context_score': 0.0,
                    'relevance_score': 0.0,
                    'processing_time': time.time() - request_start_time,
                    'error': str(e)
                }
            )
            logger.info(f"Request {request_id} - Returning error response. Total time: {time.time() - request_start_time:.2f}s")
            return response_data
            
    except Exception as e:
        logger.error(f"Request {request_id if 'request_id' in locals() else 'Unknown'} - Unhandled error: {str(e)}", exc_info=True)
        return Response(
            answer="I apologize, but I encountered an unexpected error. Please try again later.",
            confidence=0.0,
            source="error",
            is_generated=False,
            sentiment_analysis={},
            reasoning_steps=["Unhandled error occurred"],
            metadata={
                'error': str(e),
                'processing_time': time.time() - request_start_time if 'request_start_time' in locals() else 0.0
            }
        )

@app.post("/update_knowledge_base")
async def update_knowledge_base(documents: List[str]):
    try:
        # Process documents
        knowledge_base = knowledge_extractor.process_documents(documents)
        
        # Build search index
        semantic_search.build_index(
            documents=documents,
            embeddings=knowledge_base['bert']
        )
        
        # Save knowledge base
        knowledge_extractor.save_knowledge_base(knowledge_base, "data/knowledge_base.json")
        semantic_search.save_index("data/search_index")
        
        return {"message": "Knowledge base updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int = 8001, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

def cleanup_server():
    """Clean up server processes."""
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    
    # Wait for processes to terminate
    _, alive = psutil.wait_procs(children, timeout=3)
    
    # Force kill if still alive
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_server()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down server...")
    cleanup_server()

def main():
    """Main entry point with proper port handling."""
    try:
        port = find_available_port()
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            "src.api.main:app",
            host="127.0.0.1",
            port=port,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 