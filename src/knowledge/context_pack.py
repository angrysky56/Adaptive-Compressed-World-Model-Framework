"""
CompressedContextPack Module

This module provides the core functionality for compressing, storing, and expanding
knowledge representations in the Adaptive Compressed World Model Framework.
"""

import uuid
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import hashlib
import logging
import requests
from collections import Counter
import traceback
import sys

# Import ModelLoader if available, otherwise use fallback
try:
    from .model_loader import ModelLoader
except ImportError:
    ModelLoader = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_exception(e, context=""):
    """Log an exception with detailed traceback information."""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)
    logger.error(f"Exception in {context}: {str(e)}\n{tb_text}")


class CompressedContextPack:
    """
    A compressed, semantically rich representation of knowledge.
    
    The CompressedContextPack handles the compression and expansion of knowledge contexts,
    balancing information preservation with storage efficiency.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384,
                use_ollama: bool = True, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the CompressedContextPack with embedding model options.
        
        Args:
            model_name: The name of the model to use for embeddings
            embedding_dim: The dimension of the embedding vectors
            use_ollama: Whether to use Ollama for embeddings and summaries
            ollama_base_url: Base URL for Ollama API
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model_loader = None
        self.embedding_model = None
        self.summarization_model = None
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        
        # Track the critical entities that must be preserved
        self.critical_entities = []
        
        # Compression parameters
        self.compression_ratio = 0.25  # Target compression ratio (0.25 = 75% reduction)
        self.min_similarity_threshold = 0.7  # Minimum similarity for context linking
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize embedding and summarization models."""
        # If using Ollama
        if self.use_ollama:
            # Test Ollama connection
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    available_models = response.json().get("models", [])
                    model_names = [model.get("name") for model in available_models]
                    logger.info(f"Ollama connection successful. Available models: {', '.join(model_names)}")
                    
                    # Check if the embedding model is explicitly available
                    embedding_model_found = False
                    for model in model_names:
                        # Check for embedding models explicitly
                        if model.lower() == "nomic-embed-text:latest" or "all-minilm" in model.lower():
                            embedding_model_found = True
                            break
                    
                    if not embedding_model_found:
                        logger.warning("No dedicated embedding model found in Ollama. Falling back to general purpose models.")
                    
                    # Select appropriate embedding model - use the exact model name from the available models
                    if any("nomic-embed-text" in model.lower() for model in model_names):
                        for model in model_names:
                            if "nomic-embed-text" in model.lower():
                                self.model_name = model
                                break
                    elif any("all-minilm" in model.lower() for model in model_names):
                        for model in model_names:
                            if "all-minilm" in model.lower():
                                self.model_name = model
                                break
                    # Fallback to sentence-transformers if we're still using the input model name
                    elif self.model_name == "all-MiniLM-L6-v2":
                        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                        logger.info(f"No appropriate embedding model found in Ollama, using: {self.model_name}")
                        
                    # Select appropriate summarization model
                    for candidate in ["phi3", "phi2", "mistral", "gemma", "llama3"]:
                        if any(candidate in name.lower() for name in model_names):
                            self.summarization_model = next(name for name in model_names 
                                                          if candidate in name.lower())
                            break
                            
                    logger.info(f"Using Ollama with embedding model: {self.model_name}")
                    logger.info(f"Using Ollama with summarization model: {self.summarization_model}")
                else:
                    logger.warning(f"Ollama API returned status code {response.status_code}")
                    self.use_ollama = False
            except requests.exceptions.ConnectionError:
                logger.warning("Could not connect to Ollama API. Falling back to local embedding.")
                self.use_ollama = False
            except Exception as e:
                log_exception(e, "initializing Ollama connection")
                logger.warning("Error occurred while connecting to Ollama. Falling back to local embedding.")
                self.use_ollama = False
                
        # If not using Ollama (or Ollama failed), try to use ModelLoader
        if not self.use_ollama and ModelLoader is not None:
            try:
                self.model_loader = ModelLoader()
                success, self.embedding_model = self.model_loader.load_embedding_model(
                    model_name=f"sentence-transformers/{self.model_name}" 
                    if not self.model_name.startswith("sentence-transformers/") else self.model_name,
                    embedding_dim=self.embedding_dim
                )
                if not success:
                    logger.warning("Failed to load embedding model from ModelLoader. Using fallback.")
            except Exception as e:
                log_exception(e, "initializing ModelLoader")
                
    def compress(self, text: str, critical_entities: List[str] = None) -> Dict:
        """
        Compress text into a dense representation while preserving critical entities.
        
        Uses an embedding approach to generate vectors and extracts key information.
        
        Args:
            text: The raw text to compress
            critical_entities: Important entities that should be preserved in the compression
            
        Returns:
            A dictionary containing the compressed data and metadata
        """
        # Store critical entities
        if critical_entities:
            self.critical_entities = critical_entities
        
        # Create embedding for the entire text
        embedding = self._generate_embedding(text)
        
        # Create a summary
        summary = self._generate_summary(text)
        
        # Extract key phrases and entities
        key_phrases = self._extract_key_phrases(text)
        
        # Create the compressed context pack
        compressed_pack = {
            "id": str(uuid.uuid4()),
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "summary": summary,
            "key_phrases": key_phrases,
            "critical_entities": self.critical_entities,
            "original_length": len(text),
            "compressed_length": len(summary),
            "compression_ratio": len(summary) / max(1, len(text)),
            "creation_time": time.time(),
            "last_accessed": time.time(),
            "access_count": 0,
            "version": 1
        }
        
        return compressed_pack
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the text.
        
        Tries different methods in order of preference:
        1. Ollama API (if enabled)
        2. Loaded embedding model
        3. Fallback hashing-based method
        
        Args:
            text: The text to embed
            
        Returns:
            A numpy array containing the embedding vector
        """
        # Try Ollama API first if enabled
        if self.use_ollama:
            try:
                # Check if the model name needs to be adjusted for Ollama API
                model_for_api = self.model_name
                if "/" in model_for_api:  # It's likely a HuggingFace model path
                    # Extract just the model name from the path for Ollama
                    model_parts = model_for_api.split("/")
                    if len(model_parts) > 1:
                        # Try to find the model in the available models
                        response_tags = requests.get(f"{self.ollama_base_url}/api/tags")
                        if response_tags.status_code == 200:
                            available_models = response_tags.json().get("models", [])
                            model_names = [model.get("name") for model in available_models]
                            
                            # Look for a matching model name
                            for name in model_names:
                                if model_parts[-1].lower() in name.lower():
                                    model_for_api = name
                                    logger.info(f"Using Ollama model: {model_for_api} for embedding")
                                    break
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": model_for_api, "prompt": text}
                )
                
                if response.status_code == 200:
                    embedding = np.array(response.json().get("embedding", []))
                    if len(embedding) > 0:
                        return embedding
                else:
                    logger.warning(f"Ollama embedding API returned status code {response.status_code}")
                    # If we got a 404, it means the model doesn't exist or isn't properly loaded in Ollama
                    if response.status_code == 404:
                        logger.warning(f"Model '{model_for_api}' not found in Ollama. Checking for other available embedding models...")
                        
                        # Try to find any available embedding model
                        response_tags = requests.get(f"{self.ollama_base_url}/api/tags")
                        if response_tags.status_code == 200:
                            available_models = response_tags.json().get("models", [])
                            model_names = [model.get("name") for model in available_models]
                            
                            # Try with a different embedding model if available
                            for candidate in ["nomic-embed-text", "all-minilm", "e5", "bert"]:
                                matching_models = [m for m in model_names if candidate.lower() in m.lower()]
                                if matching_models:
                                    alternative_model = matching_models[0]
                                    logger.info(f"Trying alternative embedding model: {alternative_model}")
                                    
                                    alt_response = requests.post(
                                        f"{self.ollama_base_url}/api/embeddings",
                                        json={"model": alternative_model, "prompt": text}
                                    )
                                    
                                    if alt_response.status_code == 200:
                                        embedding = np.array(alt_response.json().get("embedding", []))
                                        if len(embedding) > 0:
                                            logger.info(f"Successfully used alternative model: {alternative_model}")
                                            return embedding
                    
                    logger.info("Using fallback hashing method for embedding generation")
            except Exception as e:
                log_exception(e, "using Ollama embedding API")
                
        # Try loaded embedding model if available
        if self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding with model: {e}")
                
        # Fallback to hashing method
        logger.info("Using fallback hashing method for embedding generation")
        return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate a fallback embedding using a simple hashing approach.
        
        Args:
            text: The text to embed
            
        Returns:
            A numpy array containing the embedding vector
        """
        try:
            # Validate input
            if not isinstance(text, str):
                logger.warning(f"Expected string input for embedding, got {type(text)}. Converting to string.")
                text = str(text)
                
            if not text:
                logger.warning("Empty text provided for embedding. Using zero vector.")
                return np.zeros(self.embedding_dim)
                
            # Create a fixed size embedding
            embedding = np.zeros(self.embedding_dim)
            
            # Use words for a more semantic-like embedding
            words = text.lower().split()
            
            # If no words, use characters
            if not words:
                words = list(text.lower())
                
            # Using word hash and position for a more stable embedding
            for i, word in enumerate(words):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self.embedding_dim
                embedding[idx] += 1.0 / (i + 1)  # Weight by position
                
            # Add bias to ensure non-zero values
            embedding += 0.01
                
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            log_exception(e, "generating fallback embedding")
            logger.warning("Error in fallback embedding generation. Returning zero vector.")
            return np.zeros(self.embedding_dim)
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a concise summary of the text.
        
        Tries different methods in order of preference:
        1. Ollama API (if enabled and summarization model is available)
        2. Extraction-based summarization
        
        Args:
            text: The text to summarize
            
        Returns:
            A concise summary of the text
        """
        # Try Ollama API for summarization if enabled and model available
        if self.use_ollama and self.summarization_model:
            try:
                prompt = f"""Please create a concise summary of the following text. Keep important information and entities. The summary should be about 25% of the original length.
                
Text to summarize:
{text}

Summary:"""
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.summarization_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code == 200:
                    summary = response.json().get("response", "").strip()
                    if summary:
                        # Ensure critical entities are represented
                        for entity in self.critical_entities:
                            if entity.lower() not in summary.lower() and entity.lower() in text.lower():
                                summary += f" [Critical: {entity}]"
                        return summary
                else:
                    logger.warning(f"Ollama summary API returned status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error using Ollama for summarization: {e}")
                
        # Fallback to extraction-based summarization
        logger.info("Using extraction-based method for summarization")
        return self._generate_extractive_summary(text)
    
    def _generate_extractive_summary(self, text: str) -> str:
        """
        Generate a summary by extracting key sentences from the text.
        
        Args:
            text: The text to summarize
            
        Returns:
            A summary created by extracting key sentences
        """
        # Split text into sentences
        sentences = text.replace('. ', '.|||').replace('! ', '!|||').replace('? ', '?|||').split('|||')
        
        # If the text is already short, return it directly
        if len(sentences) <= 3:
            return text
        
        # Score sentences based on presence of critical entities and keywords
        sentence_scores = []
        for sentence in sentences:
            score = 0
            # Increase score for sentences containing critical entities
            for entity in self.critical_entities:
                if entity.lower() in sentence.lower():
                    score += 2
            
            # Increase score for sentences near the beginning and end (often contain key info)
            if sentences.index(sentence) < 2:  # First two sentences
                score += 1
            elif sentences.index(sentence) > len(sentences) - 3:  # Last two sentences
                score += 1
                
            # Store the score
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score in descending order
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on target compression ratio
        target_sentences_count = max(1, int(len(sentences) * self.compression_ratio))
        top_sentences = [s[0] for s in sorted_sentences[:target_sentences_count]]
        
        # Reorder sentences to match original order
        ordered_summary = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_summary.append(sentence)
        
        # Join sentences back into a summary
        summary = ' '.join(ordered_summary)
        
        # Ensure critical entities are represented in the summary
        for entity in self.critical_entities:
            if entity.lower() not in summary.lower() and entity.lower() in text.lower():
                summary += f" [Critical: {entity}]"
                
        return summary
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from the text.
        
        Args:
            text: The text to extract key phrases from
            
        Returns:
            A list of key phrases
        """
        # Import entity extraction utilities
        try:
            from .entity_extraction import extract_entities, filter_extracted_entities
            
            # Try to use the improved entity extraction
            extracted_entities = extract_entities(text)
            filtered_entities = filter_extracted_entities(extracted_entities)
            
            # If we have critical entities, ensure they're included
            for entity in self.critical_entities:
                if entity not in filtered_entities:
                    filtered_entities.append(entity)
            
            # If we have a good number of entities, use them directly
            if len(filtered_entities) >= 5:
                return filtered_entities[:15]  # Return top 15
        except ImportError:
            logger.warning("Entity extraction module not found, using fallback approach")
            
        # Fallback: extract phrases containing critical entities
        words = text.lower().split()
        key_phrases = []
        
        # Add phrases containing critical entities
        for entity in self.critical_entities:
            entity_lower = entity.lower()
            # Find phrases containing the entity
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if entity_lower in phrase:
                    key_phrases.append(phrase)
        
        # If no critical entities or not enough phrases found, add some common words
        if len(key_phrases) < 5:
            # Count word frequencies
            word_counts = Counter(words)
            # Add phrases with common words
            for common_word, _ in word_counts.most_common(5):
                if len(common_word) > 3:  # Only consider meaningful words
                    for i in range(len(words) - 2):
                        if i < len(words) and words[i] == common_word:
                            if i + 2 < len(words):
                                phrase = ' '.join(words[i:i+3])
                                key_phrases.append(phrase)
        
        # Remove duplicates and limit the number of key phrases
        key_phrases = list(set(key_phrases))[:10]
        
        return key_phrases
    
    def expand(self, compressed_pack: Dict, include_related: bool = False) -> Dict:
        """
        Expand a compressed context pack into its detailed form.
        
        Args:
            compressed_pack: The compressed context pack
            include_related: Whether to include related contexts in the expansion
            
        Returns:
            An expanded version of the context pack
        """
        # Update access metadata
        compressed_pack["last_accessed"] = time.time()
        compressed_pack["access_count"] += 1
        
        expanded_content = compressed_pack["summary"]
        
        # Add key phrases if available
        if "key_phrases" in compressed_pack and compressed_pack["key_phrases"]:
            expanded_content += "\n\nKey phrases: " + ", ".join(compressed_pack["key_phrases"])
        
        # Add critical entities if available
        if "critical_entities" in compressed_pack and compressed_pack["critical_entities"]:
            expanded_content += "\n\nCritical entities: " + ", ".join(compressed_pack["critical_entities"])
        
        expansion = {
            "id": compressed_pack["id"],
            "expanded_content": expanded_content,
            "creation_time": compressed_pack.get("creation_time", "Unknown"),
            "last_accessed": compressed_pack["last_accessed"],
            "access_count": compressed_pack["access_count"],
            "version": compressed_pack.get("version", 1)
        }
        
        # In a full implementation, related_contexts would be populated with actual related contexts
        if include_related:
            expansion["related_contexts"] = []
        
        return expansion
    
    def calculate_similarity(self, pack1: Dict, pack2: Dict) -> float:
        """
        Calculate the similarity between two context packs based on their embeddings.
        
        Args:
            pack1: First context pack
            pack2: Second context pack
            
        Returns:
            Similarity score between 0 and 1
        """
        if "embedding" not in pack1 or "embedding" not in pack2:
            return 0.0
            
        embedding1 = np.array(pack1["embedding"])
        embedding2 = np.array(pack2["embedding"])
        
        # Check if embeddings have the same dimension
        if embedding1.shape != embedding2.shape:
            logger.warning(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")
            
            # If dimensions don't match, use a different similarity measure
            # like Jaccard similarity on entities
            entities1 = set(pack1.get("critical_entities", []))
            entities2 = set(pack2.get("critical_entities", []))
            
            if entities1 and entities2:
                # Calculate Jaccard similarity
                intersection = len(entities1.intersection(entities2))
                union = len(entities1.union(entities2))
                
                if union > 0:
                    return intersection / union
                    
            return 0.2  # Default similarity
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        return 0.0
    
    def detect_significant_change(self, original_pack: Dict, new_text: str) -> Tuple[float, bool]:
        """
        Detect if a new text represents a significant change from the original context pack.
        
        Args:
            original_pack: The original context pack
            new_text: The new text to compare against
            
        Returns:
            A tuple containing the change magnitude and whether it's significant
        """
        # Create a temporary compression of the new text
        critical_entities = original_pack.get("critical_entities", [])
        temp_pack = self.compress(new_text, critical_entities)
        
        # Calculate similarity between embeddings
        similarity = self.calculate_similarity(original_pack, temp_pack)
        
        # Convert to a change magnitude (1 - similarity)
        change_magnitude = 1.0 - similarity
        
        # Determine if the change is significant
        # The threshold could be adaptive based on the context
        is_significant = change_magnitude > 0.3  # Fixed threshold for now
        
        return change_magnitude, is_significant
    
    def save_to_json(self, compressed_pack: Dict, filepath: str) -> bool:
        """
        Save a compressed context pack to a JSON file.
        
        Args:
            compressed_pack: The context pack to save
            filepath: Path to the file to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(compressed_pack, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving context pack to {filepath}: {e}")
            return False
    
    def load_from_json(self, filepath: str) -> Optional[Dict]:
        """
        Load a compressed context pack from a JSON file.
        
        Args:
            filepath: Path to the file to load from
            
        Returns:
            The loaded context pack, or None if loading failed
        """
        try:
            with open(filepath, 'r') as f:
                compressed_pack = json.load(f)
            return compressed_pack
        except Exception as e:
            logger.error(f"Error loading context pack from {filepath}: {e}")
            return None
