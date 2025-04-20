from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Optional, Dict, Any, Tuple, Union
import re
import difflib
import os
import logging
import time
from functools import lru_cache
import hashlib
from difflib import SequenceMatcher
from datetime import datetime

logger = logging.getLogger(__name__)

class NLG:
    def __init__(self, model_name: str = 'models/TinyLlama-1.1B-Chat-v1.0'):
        # Configure logging with more details
        logger.info(f"Initializing NLG model with {model_name}")
        
        # Force CPU usage instead of MPS
        self.device = torch.device("cpu")
        logger.info("Using CPU device for model inference (MPS disabled)")
            
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,  # Disable automatic device mapping
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU
                local_files_only=True
            ).to(self.device)  # Explicitly move model to device
            logger.info("Model loaded successfully and moved to device")
            
            # Log model details
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
        
        self.similarity_threshold = 0.4  # Lower threshold to include more relevant docs
        self.max_context_docs = 3  # Reduced from 5 to 3 to speed up generation
        self.response_cache = {}
        self.cache_ttl = 3600  # Cache responses for 1 hour
        self.min_similarity_threshold = 0.4
        self.max_context_length = 3  # Reduced from 5 to 3 to speed up generation
        
        # Initialize cache
        self._init_cache()
        
    def _init_cache(self):
        """Initialize response cache"""
        self.cache = {}
        self.cache_timestamps = {}
        
    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate a cache key from query and context"""
        return hashlib.md5((query + context).encode()).hexdigest()
        
    def _check_cache(self, query: str, context: str) -> Optional[str]:
        """Check if response exists in cache and is not expired"""
        cache_key = self._get_cache_key(query, context)
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self.cache[cache_key]
        return None
        
    def _update_cache(self, query: str, context: str, response: str):
        """Update cache with new response"""
        cache_key = self._get_cache_key(query, context)
        self.cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
        
    def _prepare_context(self, query: str, search_results: List[Union[Tuple[str, float], Dict]]) -> str:
        """Prepare context for the model"""
        # Log the input query and search results
        logger.info(f"Preparing context for query: {query}")
        logger.info(f"Number of search results: {len(search_results)}")
        
        # Start with the query
        context = f"Question: {query}\n\nRelevant Information:\n"
        
        # Add search results (limited to max_context_docs)
        for i, result in enumerate(search_results[:self.max_context_docs], 1):
            # Handle both tuple and dictionary formats
            if isinstance(result, tuple):
                text, similarity = result
            else:
                text = result.get('text', '')
                similarity = result.get('similarity', 'N/A')
            
            # Truncate long texts to reduce context length
            if len(text) > 100:  # Reduced from 200 to 100
                text = text[:100] + "..."
                
            logger.debug(f"Processing search result {i}")
            logger.debug(f"Text: {text}")
            logger.debug(f"Similarity: {similarity}")
            context += f"{i}. {text}\n"
        
        logger.debug(f"Final context: {context}")
        return context

    def _generate_with_model(self, context: str) -> str:
        """Generate response using the TinyLlama model"""
        try:
            logger.info("Starting model generation...")
            
            # Simple prompt for direct response generation
            prompt = f"""<|system|>
You are a helpful assistant for the University of Nottingham Malaysia. Answer the question using only the information provided.
</s>
<|user|>
{context}
</s>
<|assistant|>"""
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    min_length=32,
                    num_beams=2,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|assistant|>")[-1].strip()
            
            # Basic cleaning
            response = response.replace("Answer:", "").strip()
            if not response.endswith('.'):
                response += '.'
            
            logger.info("Model generation completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}", exc_info=True)
            raise

    def _validate_response(self, response: str, context: str) -> bool:
        """Validate that the response only contains information from the context"""
        # Check for common hallucination patterns
        if "I don't have that specific information" in response:
            return True  # This is an acceptable response
            
        # Extract key facts from response
        response_facts = set(response.lower().split())
        
        # Extract key facts from context
        context_facts = set(context.lower().split())
        
        # Check if response contains facts not in context
        new_facts = response_facts - context_facts
        
        # Allow some common words and punctuation
        allowed_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'from',
                        '.', ',', ':', ';', '!', '?'}
        
        # Remove allowed words from new facts
        new_facts = new_facts - allowed_words
        
        # If there are significant new facts not in context, response is invalid
        if len(new_facts) > 3:  # Allow for minor variations
            logger.warning(f"Response contains facts not in context: {new_facts}")
            return False
            
        return True

    def _clean_response(self, response: str) -> str:
        """Clean and format the response to be more natural"""
        # Remove common prefixes
        prefixes_to_remove = [
            r'^Answer:\s*',
            r'^Response:\s*',
            r'^The answer is:\s*',
            r'^Here is the answer:\s*'
        ]
        
        for prefix in prefixes_to_remove:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE)
            
        # Remove common suffixes and signatures
        suffixes_to_remove = [
            r'\s*Best regards,.*$',
            r'\s*Regards,.*$',
            r'\s*Sincerely,.*$',
            r'\s*Thank you,.*$',
            r'\s*\[Your Name\].*$',
            r'\s*\[Name\].*$',
            r'\s*\[Assistant\].*$',
            r'\s*\[AI Assistant\].*$',
            r'\s*Let me know if you have any other questions or concerns.*$',
            r'\s*I hope this helps!.*$',
            r'\s*Please let me know if you need anything else.*$',
            r'\s*We invite you to visit us soon!.*$',
            r'\s*Thank you.*$',
            r'\s*You\'re welcome.*$'
        ]
        
        for suffix in suffixes_to_remove:
            response = re.sub(suffix, '', response, flags=re.IGNORECASE)
            
        # Remove any trailing whitespace and ensure proper ending
        response = response.strip()
        if not response.endswith('.'):
            response += '.'
            
        return response

    def generate_response(self, query: str, search_results: List[Tuple[str, float]]) -> str:
        """Generate a response using the language model"""
        try:
            # Prepare context from search results
            context = self._prepare_context(query, search_results)
            
            # Check cache first
            cached_response = self._check_cache(query, context)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
            
            # Generate response
            response = self._generate_with_model(context)
            
            # Cache the response
            self._update_cache(query, context, response)
            
            # Return raw response without post-processing
            if not response or response.isspace():
                logger.warning("Empty response from model")
                raise ValueError("Empty response generated")
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def generate_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate a fallback response when no good match is found"""
        is_location_question = query.lower().startswith('where')
        is_facility_question = any(word in query.lower() for word in ['facility', 'facilities', 'amenities', 'services'])
        
        if is_location_question:
            response = "The University of Nottingham Malaysia campus is located at Jalan Broga, 43500 Semenyih, Selangor Darul Ehsan, Malaysia. The campus is situated in the town of Semenyih, which is approximately 30 kilometers south of Kuala Lumpur, the capital city of Malaysia."
            reasoning = ["No specific location information found in context", 
                       "Using default campus location information"]
        elif is_facility_question:
            response = "The University of Nottingham Malaysia campus offers comprehensive facilities including Schools of Business, Engineering, and Science, well-equipped research laboratories, a modern library, sports facilities, student accommodation, and various student support services."
            reasoning = ["No specific facility information found in context",
                       "Using general facility information"]
        else:
            response = "I apologize, but I don't have enough specific information to answer your question accurately. Please try rephrasing your question or ask about our location, facilities, or general campus information."
            reasoning = ["No relevant information found in context",
                       "Suggesting alternative topics"]
            
        return {
            'response': response,
            'reasoning_steps': reasoning,
            'confidence': 0.5,
            'source': 'fallback',
            'metadata': {
                'bert_score': 0.0,
                'context_score': 0.0,
                'relevance_score': 0.0
            }
        }
    
    def generate_reasoning(self, query: str, search_results: List[Tuple[str, float]]) -> List[str]:
        """Generate reasoning steps for the response"""
        reasoning = []
        
        # Check query type
        if query.lower().startswith('where'):
            reasoning.append("Query type: Location question")
            # Check for location information
            location_docs = [(doc, score) for doc, score in search_results 
                           if any(loc in doc.lower() for loc in ['jalan', 'semenyih', 'selangor', 'malaysia'])]
            if location_docs:
                reasoning.append(f"Found {len(location_docs)} relevant location documents")
                reasoning.append(f"Best location match has confidence {location_docs[0][1]:.2f}")
            else:
                reasoning.append("No specific location information found")
                
        elif any(word in query.lower() for word in ['facility', 'facilities', 'amenities', 'services']):
            reasoning.append("Query type: Facility question")
            # Check for facility information
            facility_docs = [(doc, score) for doc, score in search_results 
                           if any(fac in doc.lower() for fac in ['facility', 'center', 'library', 'accommodation'])]
            if facility_docs:
                reasoning.append(f"Found {len(facility_docs)} relevant facility documents")
                reasoning.append(f"Best facility match has confidence {facility_docs[0][1]:.2f}")
            else:
                reasoning.append("No specific facility information found")
                
        else:
            reasoning.append("Query type: General question")
            if search_results:
                reasoning.append(f"Found {len(search_results)} relevant documents")
                reasoning.append(f"Best match has confidence {search_results[0][1]:.2f}")
            else:
                reasoning.append("No relevant information found")
                
        return reasoning
    
    def generate_multiple_responses(self, 
                                  query: str,
                                  num_responses: int = 3,
                                  additional_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate multiple diverse responses"""
        responses = []
        for _ in range(num_responses):
            # Vary temperature for diversity
            temperature = 0.6 + torch.rand(1).item() * 0.3
            response = self.generate_response(
                query,
                additional_context=additional_context
            )
            responses.append(response)
        return responses
    
    def save_model(self, path: str):
        """Save the model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load the model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path) 

    def _remove_duplicates(self, docs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Remove duplicate or highly similar documents while preserving the highest scoring ones."""
        unique_docs = []
        for doc, score in sorted(docs, key=lambda x: x[1], reverse=True):
            is_duplicate = False
            for unique_doc, _ in unique_docs:
                if self._calculate_text_similarity(doc, unique_doc) > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append((doc, score))
            if len(unique_docs) >= self.max_context_length:
                break
        return unique_docs

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _normalize_confidence(self, score: float) -> float:
        """Normalize confidence scores to be between 0 and 1."""
        if score < 0:
            return 0.0
        return min(1.0, score)

    def _clean_response(self, response: str) -> str:
        """Clean and format the response."""
        # Remove duplicate sentences
        sentences = response.split('. ')
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            normalized = sentence.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        response = '. '.join(unique_sentences)
        if not response.endswith('.'):
            response += '.'
            
        return response

    async def generate(self, query: str, relevant_docs: List[str], query_type: str = "General") -> Tuple[str, float]:
        start_time = time.time()
        request_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Request {request_id} - Starting processing for query: {query}")
        
        try:
            # Log document processing
            logger.info(f"Request {request_id} - Processing {len(relevant_docs)} relevant documents")
            doc_scores = [(doc, self._calculate_text_similarity(query, doc)) for doc in relevant_docs]
            filtered_docs = [(doc, score) for doc, score in doc_scores if score > self.min_similarity_threshold]
            logger.info(f"Request {request_id} - Found {len(filtered_docs)} documents above similarity threshold")
            
            # Log top matches
            sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)[:3]
            for i, (doc, score) in enumerate(sorted_docs, 1):
                logger.info(f"Request {request_id} - Top match {i}: Score {score:.3f}")
                logger.debug(f"Request {request_id} - Document {i} preview: {doc[:100]}...")
            
            # Prepare prompt
            prompt = self._prepare_prompt(query, relevant_docs, query_type)
            logger.info(f"Request {request_id} - Prepared prompt with {len(prompt)} characters")
            
            # Generate response
            generation_start = time.time()
            logger.info(f"Request {request_id} - Starting TinyLlama generation")
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            logger.info(f"Request {request_id} - Input tokens: {inputs['input_ids'].shape[1]}")
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    min_length=50,
                    num_beams=2,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_time = time.time() - generation_start
            logger.info(f"Request {request_id} - Generation completed in {generation_time:.2f}s")
            logger.info(f"Request {request_id} - Generated response length: {len(response)} characters")
            
            # Clean response
            clean_start = time.time()
            response = self._clean_response(response)
            logger.info(f"Request {request_id} - Response cleaned in {time.time() - clean_start:.2f}s")
            
            # Calculate confidence
            confidence = max([score for _, score in filtered_docs]) if filtered_docs else 0.0
            confidence = self._normalize_confidence(confidence)
            
            total_time = time.time() - start_time
            logger.info(f"Request {request_id} - Total processing time: {total_time:.2f}s")
            logger.info(f"Request {request_id} - Final confidence score: {confidence:.3f}")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_cached = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(f"Request {request_id} - GPU Memory: {memory_allocated:.2f}MB allocated, {memory_cached:.2f}MB cached")
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Request {request_id} - Error during generation: {str(e)}", exc_info=True)
            return f"An error occurred while processing your request (ID: {request_id}). Please try again.", 0.0
    
    def _prepare_prompt(self, query: str, relevant_docs: List[str], query_type: str) -> str:
        """Prepare a prompt with logging."""
        context = "\n".join([doc for doc, _ in sorted(relevant_docs, key=lambda x: x[1], reverse=True)[:5]])
        
        prompt = f"""Based on the following information about the University of Nottingham Malaysia:
{context}

Question: {query}

Please provide a clear and concise answer. Focus on the most relevant details and avoid repetition."""
        
        logger.debug(f"Prepared prompt with {len(context)} characters of context")
        return prompt

    def _log_response_metrics(self, response: str, request_id: str):
        """Log detailed metrics about the response."""
        sentences = response.split('. ')
        logger.info(f"Request {request_id} - Response metrics:")
        logger.info(f"Request {request_id} - Number of sentences: {len(sentences)}")
        logger.info(f"Request {request_id} - Average sentence length: {sum(len(s.split()) for s in sentences)/len(sentences):.1f} words")
        logger.info(f"Request {request_id} - Total response length: {len(response)} characters") 