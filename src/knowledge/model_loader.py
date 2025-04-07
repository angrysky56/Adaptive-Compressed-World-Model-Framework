"""
Model Loader Module

This module handles the loading and management of machine learning models
for the Adaptive Compressed World Model Framework.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    A utility class for loading and managing ML models.
    
    This class handles the loading of various ML models (transformers, embeddings, etc.)
    with support for quantization, caching, and fallback mechanisms.
    """
    
    def __init__(self, cache_dir: str = None, use_quantization: bool = True,
                device: str = None, model_config: Dict = None):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory for caching models
            use_quantization: Whether to use quantization (when available)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detection)
            model_config: Configuration for model loading
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.acwmf', 'models')
        self.use_quantization = use_quantization
        self.models = {}
        self.model_config = model_config or {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device: {self.device}")
        
    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                            embedding_dim: int = 384, force_reload: bool = False) -> Tuple[bool, Any]:
        """
        Load a sentence transformer model for generating embeddings.
        
        Args:
            model_name: Name of the model to load
            embedding_dim: Expected embedding dimension (for fallback)
            force_reload: Whether to force reloading even if already loaded
            
        Returns:
            (success, model): Tuple indicating success and the loaded model
        """
        # Check if already loaded
        if model_name in self.models and not force_reload:
            logger.info(f"Using already loaded model: {model_name}")
            return True, self.models[model_name]
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {model_name}")
            
            # Load the model with 4-bit quantization if requested
            load_kwargs = {
                "cache_folder": self.cache_dir
            }
            
            if self.use_quantization:
                try:
                    import bitsandbytes as bnb
                    load_kwargs["quantize_config"] = {"load_in_4bit": True}
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, quantization disabled")
                    
            # Load the model
            model = SentenceTransformer(model_name, device=self.device, **load_kwargs)
            
            # Store for future use
            self.models[model_name] = model
            
            return True, model
            
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            return False, self._create_fallback_embedding_model(embedding_dim)
            
    def _create_fallback_embedding_model(self, embedding_dim: int = 384) -> Any:
        """
        Create a simple fallback embedding model.
        
        This is used when the main model fails to load.
        
        Args:
            embedding_dim: Dimension of the embedding vector
            
        Returns:
            A minimal model object with an encode method
        """
        logger.warning(f"Using fallback embedding model with dimension {embedding_dim}")
        
        class FallbackEmbeddingModel:
            def __init__(self, dim: int):
                self.dim = dim
                
            def encode(self, text, **kwargs):
                if isinstance(text, list):
                    return np.array([self._hash_text(t) for t in text])
                return self._hash_text(text)
                
            def _hash_text(self, text: str) -> np.ndarray:
                # Simple hashing approach to generate stable embeddings
                import hashlib
                
                # Create a fixed size embedding
                embedding = np.zeros(self.dim)
                
                # Use words for a more semantic-like embedding
                words = text.lower().split()
                
                # Using word hash and position for a more stable embedding
                for i, word in enumerate(words):
                    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                    idx = h % self.dim
                    embedding[idx] += 1.0 / (i + 1)  # Weight by position
                    
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
                return embedding
                
        return FallbackEmbeddingModel(embedding_dim)
        
    def load_tokenizer(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[bool, Any]:
        """
        Load a tokenizer for a given model.
        
        Args:
            model_name: Name of the model to load tokenizer for
            
        Returns:
            (success, tokenizer): Tuple indicating success and the loaded tokenizer
        """
        tokenizer_key = f"{model_name}_tokenizer"
        
        if tokenizer_key in self.models:
            return True, self.models[tokenizer_key]
            
        try:
            from transformers import AutoTokenizer
            
            logger.info(f"Loading tokenizer for model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Store for future use
            self.models[tokenizer_key] = tokenizer
            
            return True, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_name}: {e}")
            return False, None
            
    def load_small_language_model(self, model_name: str = "microsoft/phi-2", 
                                max_length: int = 2048) -> Tuple[bool, Any]:
        """
        Load a small language model for text generation.
        
        Args:
            model_name: Name of the model to load
            max_length: Maximum sequence length
            
        Returns:
            (success, model): Tuple indicating success and the loaded model
        """
        if model_name in self.models:
            return True, self.models[model_name]
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            logger.info(f"Loading language model: {model_name}")
            
            # Load with quantization if available and requested
            load_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto"
            }
            
            if self.use_quantization:
                try:
                    load_kwargs["load_in_4bit"] = True
                    logger.info("Using 4-bit quantization for language model")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}")
                    
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Load model with quantization options
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            
            # Create a pipeline for easier use
            gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                device_map="auto"
            )
            
            # Store for future use
            self.models[model_name] = gen_pipeline
            
            return True, gen_pipeline
            
        except Exception as e:
            logger.error(f"Error loading language model {model_name}: {e}")
            return False, None
            
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free up memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            Whether the model was successfully unloaded
        """
        if model_name in self.models:
            try:
                # Remove reference to the model
                del self.models[model_name]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
                logger.info(f"Unloaded model: {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                
        return False
        
    def get_available_memory(self) -> Dict:
        """
        Get information about available memory.
        
        Returns:
            Dictionary with memory information
        """
        import psutil
        
        memory_info = {
            "system": {
                "total": psutil.virtual_memory().total / (1024 ** 3),  # GB
                "available": psutil.virtual_memory().available / (1024 ** 3),  # GB
                "percent_used": psutil.virtual_memory().percent
            }
        }
        
        # Add GPU memory info if available
        if self.device == "cuda" and torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            memory_info["cuda"] = {
                "total": torch.cuda.get_device_properties(cuda_device).total_memory / (1024 ** 3),  # GB
                "reserved": torch.cuda.memory_reserved(cuda_device) / (1024 ** 3),  # GB
                "allocated": torch.cuda.memory_allocated(cuda_device) / (1024 ** 3)  # GB
            }
            
        return memory_info
        
    def save_config(self, filepath: str) -> bool:
        """
        Save the current model configuration to a file.
        
        Args:
            filepath: Path to save the configuration
            
        Returns:
            Whether the config was saved successfully
        """
        config = {
            "cache_dir": self.cache_dir,
            "use_quantization": self.use_quantization,
            "device": self.device,
            "model_config": self.model_config,
            "loaded_models": list(self.models.keys())
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving model config to {filepath}: {e}")
            return False
            
    @classmethod
    def from_config(cls, filepath: str) -> 'ModelLoader':
        """
        Load a ModelLoader from a configuration file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            A new ModelLoader instance
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            return cls(
                cache_dir=config.get("cache_dir"),
                use_quantization=config.get("use_quantization", True),
                device=config.get("device"),
                model_config=config.get("model_config", {})
            )
            
        except Exception as e:
            logger.error(f"Error loading model config from {filepath}: {e}")
            return cls()  # Return default instance
