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
        self.max_context_docs = 5  # Maximum number of context documents to include
        self.response_cache = {}
        self.cache_ttl = 3600  # Cache responses for 1 hour
        self.min_similarity_threshold = 0.4
        self.max_context_length = 5  # Maximum number of relevant documents to include
        
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
    
    def generate_response(self, query: str, search_results: List[Tuple[str, float]]) -> str:
        """Generate a response using the language model"""
        try:
            # Prepare context from search results
            context = self._prepare_context(query, search_results)
            
            # Generate response
            response = self._generate_with_model(context)
            
            # Return raw response without post-processing
            if not response or response.isspace():
                logger.warning("Empty response from model")
                raise ValueError("Empty response generated")
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def _prepare_context(self, query: str, search_results: List[Union[Tuple[str, float], Dict]]) -> str:
        """Prepare context for the model"""
        # Log the input query and search results
        logger.info(f"Preparing context for query: {query}")
        logger.info(f"Number of search results: {len(search_results)}")
        
        # Start with the query
        context = f"Question: {query}\n\nRelevant Information:\n"
        
        # Add search results
        for i, result in enumerate(search_results, 1):
            # Handle both tuple and dictionary formats
            if isinstance(result, tuple):
                text, similarity = result
            else:
                text = result.get('text', '')
                similarity = result.get('similarity', 'N/A')
            
            logger.debug(f"Processing search result {i}")
            logger.debug(f"Text: {text[:200]}...")
            logger.debug(f"Similarity: {similarity}")
            context += f"{i}. {text}\n"
        
        # Add instructions
        context += "\nInstructions: Based on the following information, provide a clear and concise response. Focus on the most relevant details and maintain a natural, conversational tone."
        
        logger.debug(f"Final context: {context}")
        return context

    def _generate_with_model(self, context: str) -> str:
        """Generate response using the TinyLlama model"""
        try:
            logger.info("Starting model generation...")
            
            # Prepare the prompt
            prompt = f"""<|system|>
You are a helpful assistant for the University of Nottingham Malaysia. Keep responses concise and complete.
</s>
<|user|>
{context}
</s>
<|assistant|>"""
            
            # Tokenize with moderate max length
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with balanced parameters
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=768,
                    min_length=64,
                    num_beams=2,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode and extract response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_parts = response.split("<|assistant|>")
            
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
            else:
                response = response.strip()
                
            logger.info(f"Raw model response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}", exc_info=True)
            raise

    def _post_process_response(self, response: str, query: str) -> str:
        """Post-process the generated response"""
        logger.info("Starting post-processing of response")
        logger.debug(f"Original response: {response}")
        
        # Clean up the response
        response = re.sub(r'\s+', ' ', response).strip()
        logger.debug(f"Cleaned response: {response}")
        
        # Handle location queries
        if any(word in query.lower() for word in ['where', 'location', 'address', 'campus']):
            logger.info("Processing location query")
            
            # Try different patterns to extract location information
            patterns = [
                r'(?:located|situated|found).*?(?:in|at)\s+(.*?)(?:\.|\n|$)',
                r'(?:address|location).*?(?:is|:)\s+(.*?)(?:\.|\n|$)',
                r'(?:campus|university).*?(?:in|at)\s+(.*?)(?:\.|\n|$)'
            ]
            
            location = None
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    break
            
            # Try to extract distance information
            distance = re.search(r'(?:about|approximately|around)\s+(\d+)\s*(?:km|kilometers?)\s+(?:from|to)\s+(.*?)(?:\.|\n|$)', response, re.IGNORECASE)
            
            logger.debug(f"Extracted location: {location}")
            logger.debug(f"Extracted distance: {distance.groups() if distance else None}")
            
            # Build the response
            response_parts = []
            
            if location:
                response_parts.append(f"The University of Nottingham Malaysia campus is located in {location}.")
            else:
                # Fallback location if not found in response
                response_parts.append("The University of Nottingham Malaysia campus is located in Semenyih, Selangor.")
            
            if distance:
                response_parts.append(f"It is approximately {distance.group(1)} kilometers from {distance.group(2)}.")
            else:
                # Add default distance information
                response_parts.append("It is approximately 45 kilometers from Kuala Lumpur city center.")
            
            final_response = " ".join(response_parts)
            logger.debug(f"Final processed response: {final_response}")
            return final_response
        
        # Handle library queries
        if any(word in query.lower() for word in ['library', 'librarian', 'reference desk']):
            logger.info("Processing library query")
            # Extract contact information with improved regex
            phone = re.search(r'(?:03-)?\d{4}\s*\d{4}', response)
            hours = re.search(r'between (.*?) on weekdays', response.lower())
            
            logger.debug(f"Extracted phone: {phone.group(0) if phone else 'None'}")
            logger.debug(f"Extracted hours: {hours.group(1) if hours else 'None'}")
            
            # Build a structured response
            response_parts = ["The University of Nottingham Malaysia Library provides comprehensive services to students and staff."]
            
            if phone:
                response_parts.append(f"You can contact the Reference Desk at {phone.group(0)}")
            
            if hours:
                response_parts.append(f"Library staff are available at the Reference Desk {hours.group(1)} on weekdays.")
            
            # Add social media information if available
            if 'twitter' in response.lower():
                response_parts.append("You can follow the Library on Twitter for the latest updates.")
            
            # Add membership benefits if available
            if 'membership' in response.lower():
                response_parts.append("Library membership includes borrowing privileges, access to facilities, and interlibrary loan services.")
            
            final_response = "\n".join(response_parts)
            logger.debug(f"Final processed response: {final_response}")
            return final_response
        
        # Handle specialization queries
        if any(word in query.lower() for word in ['specialization', 'specialize', 'specialized', 'specializing']):
            logger.info("Processing specialization query")
            # Extract specializations from the response
            specializations = []
            if 'specializations in areas such as' in response.lower():
                spec_match = re.search(r'specializations in areas such as (.*?)(?:\.|$)', response.lower())
                if spec_match:
                    specializations = [s.strip() for s in spec_match.group(1).split(',')]
            
            # Extract program types
            programs = []
            if 'bsc' in response.lower():
                programs.append('BSc (Hons) Computer Science')
            if 'msc' in response.lower():
                msc_programs = re.findall(r'MSc in ([^,\.]+)', response)
                programs.extend([f'MSc in {p.strip()}' for p in msc_programs])
            if 'mphil' in response.lower() or 'phd' in response.lower():
                phd_programs = re.findall(r'PhD in ([^,\.]+)', response)
                programs.extend([f'PhD in {p.strip()}' for p in phd_programs])
            
            # Build a structured response
            response_parts = ["The School of Computer Science offers various programs and specializations:"]
            
            if specializations:
                response_parts.append(f"Specializations include: {', '.join(specializations)}.")
            
            if programs:
                response_parts.append("\nAvailable programs:")
                for program in sorted(set(programs)):  # Remove duplicates and sort
                    response_parts.append(f"- {program}")
            
            final_response = "\n".join(response_parts)
            logger.info(f"Final processed response: {final_response}")
            return final_response
        
        # Handle accommodation queries
        if any(word in query.lower() for word in ['accommodation', 'housing', 'residence', 'dorm', 'dormitory']):
            # Extract accommodation types
            has_on_campus = 'on-campus' in response.lower()
            has_off_campus = 'off-campus' in response.lower()
            has_postgrad = 'postgraduate' in response.lower()
            
            # Extract contact information
            contact = re.search(r'contact (?:the )?([^\.]+) for', response.lower())
            
            # Build a structured response
            response_parts = ["Yes, the University of Nottingham Malaysia provides accommodation options for students."]
            
            # Add accommodation types
            if has_on_campus and has_off_campus:
                response_parts.append("Both on-campus and off-campus accommodation are available.")
            elif has_on_campus:
                response_parts.append("On-campus accommodation is available.")
            elif has_off_campus:
                response_parts.append("Off-campus accommodation options are available near the university.")
            
            # Add specific details
            if has_postgrad:
                response_parts.append("There are specific accommodation options designed for postgraduate students.")
            
            # Add contact information
            if contact:
                response_parts.append(f"For more information and booking assistance, you can contact {contact.group(1)}.")
            
            final_response = "\n".join(response_parts)
            logger.info(f"Final processed response: {final_response}")
            return final_response
        
        # Handle international student support queries
        if any(word in query.lower() for word in ['international', 'support', 'student support']):
            # Extract contact information
            email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response)
            phone = re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{4}[-.\s]?\d{4}', response)
            
            # Extract support services
            services = []
            if 'documentation' in response.lower():
                services.append('documentation (visa letters, bank account letters)')
            if 'visa' in response.lower():
                services.append('visa assistance')
            if 'bank' in response.lower():
                services.append('bank account assistance')
            if 'living' in response.lower():
                services.append('living expenses advice')
            
            # Build a structured response
            response_parts = ["The University of Nottingham Malaysia provides comprehensive support for international students."]
            
            if services:
                response_parts.append(f"Services include: {', '.join(services)}.")
            
            if email or phone:
                response_parts.append("You can contact the International Student Support Office through:")
                if email:
                    response_parts.append(f"- Email: {email.group(0)}")
                if phone:
                    response_parts.append(f"- Phone: {phone.group(0)}")
            
            final_response = "\n".join(response_parts)
            logger.info(f"Final processed response: {final_response}")
            return final_response
        
        # Handle language requirement queries
        if any(word in query.lower() for word in ['language', 'english', 'ielts', 'toefl', 'pte']):
            # Extract all unique language tests from the response
            tests = re.findall(r'(?:IELTS|TOEFL|PTE|GCE A Level|SPM|GCSE O-Level|IGCSE|MUET|IB English|OSSD English|CBSE/CISCE Class XII|CBSE/CISCE Class X)', response)
            unique_tests = sorted(set(tests))  # Remove duplicates and sort
            
            if unique_tests:
                # Group similar tests together
                grouped_tests = []
                for test in unique_tests:
                    if test not in grouped_tests:
                        grouped_tests.append(test)
                
                # Format the response
                if len(grouped_tests) > 5:
                    # If many tests, group them by type
                    international_tests = [t for t in grouped_tests if t in ['IELTS', 'TOEFL', 'PTE']]
                    school_tests = [t for t in grouped_tests if t in ['GCE A Level', 'SPM', 'GCSE O-Level', 'IGCSE']]
                    other_tests = [t for t in grouped_tests if t not in international_tests + school_tests]
                    
                    response_parts = ["The University of Nottingham Malaysia accepts the following English language tests:"]
                    if international_tests:
                        response_parts.append(f"International tests: {', '.join(international_tests)}")
                    if school_tests:
                        response_parts.append(f"School qualifications: {', '.join(school_tests)}")
                    if other_tests:
                        response_parts.append(f"Other qualifications: {', '.join(other_tests)}")
                    
                    final_response = "\n".join(response_parts)
                    logger.info(f"Final processed response: {final_response}")
                    return final_response
                else:
                    final_response = f"The University of Nottingham Malaysia accepts the following English language tests: {', '.join(grouped_tests)}."
                    logger.info(f"Final processed response: {final_response}")
                    return final_response
        
        # Handle program-related queries
        if any(word in query.lower() for word in ['program', 'programs', 'course', 'courses', 'study', 'studies']):
            # Extract program names if they're in a list format
            program_match = re.search(r'programs listed include (.*?)(?:\.|$)', response.lower())
            if program_match:
                programs = program_match.group(1).strip()
                response = f"The University of Nottingham Malaysia offers the following programs: {programs}."
            else:
                # If no list found, try to extract program names from the response
                programs = re.findall(r'(?:Applied Psychology|Bioscience|Biomedical|Pharmacy|Business|Computer Sciences|Engineering|Economics|Education|Politics)', response)
                if programs:
                    response = f"The University of Nottingham Malaysia offers the following programs: {', '.join(programs)}."
        
        # Add location-specific formatting if needed
        elif query.lower().startswith('where'):
            # Ensure location information is properly formatted
            if 'located' not in response.lower():
                response = f"The University of Nottingham Malaysia campus is located in {response}"
        
        # Add facility-specific formatting if needed
        elif any(word in query.lower() for word in ['facility', 'facilities', 'amenities']):
            # Ensure facility information is properly formatted
            if 'facilities' not in response.lower():
                response = f"The University of Nottingham Malaysia campus offers the following facilities: {response}"
        
        logger.info("No special handling needed, returning cleaned response")
        return response

    def _similar_content(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two pieces of text are similar using character-level comparison"""
        if not text1 or not text2:
            return False
        
        # Use difflib's SequenceMatcher for similarity comparison
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity > threshold
    
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

    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate a unique cache key for the query and context."""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()

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