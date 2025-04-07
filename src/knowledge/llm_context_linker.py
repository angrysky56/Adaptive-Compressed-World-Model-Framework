"""
LLM-assisted Context Linker

This module leverages small language models to enhance the linking and relationship analysis
between knowledge contexts in the Adaptive Compressed World Model Framework.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

# Import model loader for LLM access
from .model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMContextLinker:
    """
    Uses language models to analyze and enhance context relationships.
    
    This class enriches the context linking process by using LLMs to:
    1. Identify nuanced relationship types
    2. Generate human-readable explanations of relationships
    3. Suggest missing connections
    4. Detect conceptual gaps in the knowledge graph
    5. Extract key entities and concepts from text
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2", use_ollama: bool = True,
                ollama_model: str = None, cache_dir: str = None):
        """
        Initialize the LLM Context Linker.
        
        Args:
            model_name: Name of the local language model to use
            use_ollama: Whether to use Ollama API instead of local models
            ollama_model: Ollama model name (if use_ollama is True)
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.ollama_base_url = "http://localhost:11434"
        self.model_loader = None
        self.llm = None
        
        # Set the Ollama model
        if ollama_model:
            self.ollama_model = ollama_model
        else:
            self.ollama_model = self._get_best_ollama_model()
        
        # Initialize model if not using Ollama
        if not use_ollama:
            self.model_loader = ModelLoader(cache_dir=cache_dir)
            success, self.llm = self.model_loader.load_small_language_model(model_name=model_name)
            
            if not success:
                logger.warning(f"Failed to load language model {model_name}. Will use simpler methods.")
        
    def _get_best_ollama_model(self) -> str:
        """
        Get the best available model from Ollama for context linking.
        
        Returns:
            Name of the best available model
        """
        import requests
        
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                model_names = [model.get("name") for model in available_models]
                
                # Look for suitable models in order of preference
                preferred_models = [
                    "gemma3", "gemma", "phi-3", "phi3", "phi-2", "phi2", "mistral", 
                    "llama3", "llama2", "deepseek", "qwen", "qwen2"
                ]
                
                for preferred in preferred_models:
                    matching_models = [m for m in model_names if preferred.lower() in m.lower()]
                    if matching_models:
                        return matching_models[0]  # Return the first matching model
                
                # If no preferred model found, return any available model
                if model_names:
                    return model_names[0]
                    
            return "gemma:7b"  # Default fallback
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {e}")
            return "gemma:7b"  # Default fallback
    
    def analyze_relationship(self, context1: Dict, context2: Dict) -> Dict:
        """
        Analyze the relationship between two contexts using LLM.
        
        Args:
            context1: First context pack
            context2: Second context pack
            
        Returns:
            Dictionary with relationship analysis
        """
        # Extract key information for analysis
        summary1 = context1.get("summary", "No summary available")
        summary2 = context2.get("summary", "No summary available")
        entities1 = context1.get("critical_entities", [])
        entities2 = context2.get("critical_entities", [])
        
        # Create the prompt for the LLM
        prompt = f"""Analyze the relationship between these two knowledge contexts:

CONTEXT 1:
Summary: {summary1}
Key entities: {', '.join(entities1)}

CONTEXT 2:
Summary: {summary2}
Key entities: {', '.join(entities2)}

Please determine:
1. The type of relationship between these contexts
2. The strength of the relationship on a scale of 0.0 to 1.0
3. Which entities or concepts serve as bridges between them
4. A brief explanation of the relationship

Format your response as a JSON object with the following keys:
- relationship_type: string (e.g., "is_parent_of", "is_example_of", "contradicts", "extends", "is_similar_to", etc.)
- strength: float between 0.0 and 1.0
- bridge_concepts: array of strings
- explanation: string description

Response:"""
        
        try:
            # Generate analysis using available LLM
            if self.use_ollama:
                return self._generate_with_ollama(prompt)
            elif self.llm is not None:
                return self._generate_with_local_model(prompt)
            else:
                # Fallback to simple analysis without LLM
                return self._fallback_analysis(context1, context2)
        except Exception as e:
            logger.error(f"Error in LLM relationship analysis: {e}")
            return self._fallback_analysis(context1, context2)
    
    def _generate_with_ollama(self, prompt: str) -> Dict:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Parsed JSON response as a dictionary
        """
        import requests
        
        try:
            # Set a timeout to prevent hanging
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=10  # Add a timeout to prevent hanging
            )
            
            if response.status_code == 200:
                llm_response = response.json().get("response", "").strip()
                
                # Extract JSON from response if needed
                if "{" in llm_response and "}" in llm_response:
                    json_text = llm_response[llm_response.find("{"):llm_response.rfind("}")+1]
                    try:
                        return json.loads(json_text)
                    except:
                        logger.warning("Failed to parse JSON from LLM response")
                        
                # If parsing failed, return a simple analysis
                return {
                    "relationship_type": "unknown",
                    "strength": 0.5,
                    "bridge_concepts": [],
                    "explanation": "Failed to analyze relationship with LLM"
                }
            else:
                logger.warning(f"Ollama API returned status code {response.status_code}")
                return self._create_simple_analysis()
                
        except Exception as e:
            logger.error(f"Error using Ollama API: {e}")
            return self._create_simple_analysis()
    
    def _generate_with_local_model(self, prompt: str) -> Dict:
        """
        Generate text using local language model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Parsed JSON response as a dictionary
        """
        if self.llm is None:
            return self._create_simple_analysis()
            
        try:
            # Generate text with the model
            result = self.llm(prompt, max_length=1024, temperature=0.1, top_p=0.9)
            
            if isinstance(result, list):
                # For pipeline outputs
                llm_response = result[0]["generated_text"]
                if prompt in llm_response:
                    # Remove the prompt from the output
                    llm_response = llm_response[len(prompt):].strip()
            else:
                llm_response = result.strip()
                
            # Extract JSON from response if needed
            if "{" in llm_response and "}" in llm_response:
                json_text = llm_response[llm_response.find("{"):llm_response.rfind("}")+1]
                try:
                    return json.loads(json_text)
                except:
                    logger.warning("Failed to parse JSON from LLM response")
                    
            # If parsing failed, return a simple analysis
            return self._create_simple_analysis()
            
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            return self._create_simple_analysis()
    
    def _fallback_analysis(self, context1: Dict, context2: Dict) -> Dict:
        """
        Perform a simple analysis without using LLM.
        
        Args:
            context1: First context pack
            context2: Second context pack
            
        Returns:
            Simple relationship analysis dictionary
        """
        # Extract entities for comparison
        entities1 = set(context1.get("critical_entities", []))
        entities2 = set(context2.get("critical_entities", []))
        
        # Find shared entities
        shared_entities = list(entities1.intersection(entities2))
        
        # Determine relationship type based on entity overlap
        relationship_type = "is_related_to"
        
        if entities1.issubset(entities2) and not entities1 == entities2:
            relationship_type = "is_subtopic_of"
        elif entities2.issubset(entities1) and not entities1 == entities2:
            relationship_type = "is_supertopic_of"
        elif len(shared_entities) == 0:
            relationship_type = "is_weakly_connected"
            
        # Calculate strength based on Jaccard similarity
        union = len(entities1.union(entities2))
        strength = len(shared_entities) / max(1, union)
        
        # Create explanation
        if shared_entities:
            explanation = f"Contexts share {len(shared_entities)} entities: {', '.join(shared_entities[:3])}"
            if len(shared_entities) > 3:
                explanation += f" and {len(shared_entities) - 3} more"
        else:
            explanation = "Contexts have no shared entities but may be semantically related"
            
        return {
            "relationship_type": relationship_type,
            "strength": strength,
            "bridge_concepts": shared_entities,
            "explanation": explanation
        }
    
    def _create_simple_analysis(self) -> Dict:
        """Create a default simple analysis when LLM generation fails."""
        return {
            "relationship_type": "unknown",
            "strength": 0.5,
            "bridge_concepts": [],
            "explanation": "Unable to analyze relationship with LLM"
        }
        
    def suggest_missing_links(self, context_graph: Any, min_similarity: float = 0.7) -> List[Dict]:
        """
        Use LLM to suggest potential missing links in the context graph.
        
        Args:
            context_graph: The DynamicContextGraph object
            min_similarity: Minimum similarity threshold for suggestions
            
        Returns:
            List of dictionaries with suggested links
        """
        # Extract nodes from the graph
        nodes = list(context_graph.graph.nodes(data=True))
        if len(nodes) < 2:
            return []
            
        suggested_links = []
        
        # Check a sample of potential links to avoid excessive LLM calls
        max_checks = min(100, len(nodes) * (len(nodes) - 1) // 2)
        checked = 0
        
        # Sort nodes by degree to prioritize connecting isolated nodes
        node_degrees = {node: context_graph.graph.degree(node) for node, _ in nodes}
        sorted_nodes = sorted(nodes, key=lambda x: node_degrees[x[0]])
        
        # Check for potential missing links, prioritizing isolated nodes
        for i, (node_id1, node_data1) in enumerate(sorted_nodes):
            for j, (node_id2, node_data2) in enumerate(sorted_nodes):
                if j <= i:  # Skip self-links and duplicates
                    continue
                    
                # Skip if link already exists
                if context_graph.graph.has_edge(node_id1, node_id2):
                    continue
                    
                # Calculate basic similarity to filter obvious non-matches
                embedding_sim = 0.0
                
                if "embedding" in node_data1 and "embedding" in node_data2:
                    embedding1 = np.array(node_data1["embedding"])
                    embedding2 = np.array(node_data2["embedding"])
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(embedding1, embedding2)
                    norm1 = np.linalg.norm(embedding1)
                    norm2 = np.linalg.norm(embedding2)
                    
                    if norm1 > 0 and norm2 > 0:
                        embedding_sim = dot_product / (norm1 * norm2)
                
                # Only use LLM for pairs with decent basic similarity
                if embedding_sim >= min_similarity:
                    # Use LLM to analyze potential relationship
                    analysis = self.analyze_relationship(node_data1, node_data2)
                    
                    # If LLM confirms a strong relationship, suggest the link
                    if analysis.get("strength", 0) >= min_similarity:
                        suggested_links.append({
                            "source_id": node_id1,
                            "target_id": node_id2,
                            "relationship_type": analysis.get("relationship_type", "is_related_to"),
                            "strength": analysis.get("strength", 0),
                            "bridge_concepts": analysis.get("bridge_concepts", []),
                            "explanation": analysis.get("explanation", "")
                        })
                
                checked += 1
                if checked >= max_checks:
                    break
                    
            if checked >= max_checks:
                break
                
        return suggested_links
        
    def explain_context_cluster(self, context_graph: Any, context_ids: List[str]) -> Dict:
        """
        Generate a comprehensive explanation of a cluster of related contexts.
        
        Args:
            context_graph: The DynamicContextGraph object
            context_ids: List of context IDs in the cluster
            
        Returns:
            Dictionary with explanation of the cluster
        """
        if len(context_ids) == 0:
            return {
                "theme": "No contexts provided",
                "summary": "No contexts to analyze",
                "key_concepts": []
            }
            
        # Gather context data
        contexts = []
        all_entities = set()
        
        for context_id in context_ids:
            if context_graph.graph.has_node(context_id):
                node_data = context_graph.graph.nodes[context_id]
                summary = node_data.get("summary", "No summary available")
                entities = node_data.get("critical_entities", [])
                
                contexts.append({
                    "id": context_id,
                    "summary": summary,
                    "entities": entities
                })
                
                all_entities.update(entities)
        
        # If no contexts found, return empty analysis
        if not contexts:
            return {
                "theme": "No valid contexts found",
                "summary": "None of the provided context IDs were found in the graph",
                "key_concepts": []
            }
            
        # Create the prompt for the LLM
        prompt = f"""Analyze this cluster of related knowledge contexts:

{json.dumps(contexts, indent=2)}

Please provide:
1. A theme that unifies these contexts
2. A summary describing how these contexts relate to each other
3. The most important concepts that appear across multiple contexts

Format your response as a JSON object with the following keys:
- theme: string (a concise label for the cluster)
- summary: string (a paragraph explaining the relationships)
- key_concepts: array of strings (important shared concepts)

Response:"""
        
        try:
            # Generate analysis using available LLM
            if self.use_ollama:
                result = self._generate_with_ollama(prompt)
            elif self.llm is not None:
                result = self._generate_with_local_model(prompt)
            else:
                # Fallback to simple analysis without LLM
                result = {
                    "theme": self._extract_theme(contexts),
                    "summary": f"This cluster contains {len(contexts)} related contexts.",
                    "key_concepts": list(all_entities)[:10]
                }
                
            return result
                
        except Exception as e:
            logger.error(f"Error in LLM cluster analysis: {e}")
            return {
                "theme": self._extract_theme(contexts),
                "summary": f"This cluster contains {len(contexts)} related contexts.",
                "key_concepts": list(all_entities)[:10]
            }
    
    async def extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """
        Extract key concepts and entities from text using LLM.
        
        Args:
            text: Text to analyze
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of extracted concepts
        """
        # Create the prompt for the LLM
        prompt = f"""Extract the most important entities, concepts, and technical terms from this text.
Focus on proper nouns, domain-specific terminology, and key concepts that are central to understanding the content.
Exclude common words like 'this', 'that', 'also', etc.

Text:
{text[:3000]}... # Truncate if too long

Return ONLY a JSON array of strings with no explanation or additional text:
"""
        
        try:
            # Generate concepts using available LLM
            if self.use_ollama:
                result = self._generate_with_ollama(prompt)
                if isinstance(result, list):
                    return result[:max_concepts]
                elif isinstance(result, dict) and "concepts" in result:
                    return result["concepts"][:max_concepts]
                else:
                    return self._fallback_extract_concepts(text)[:max_concepts]
                    
            elif self.llm is not None:
                result = self._generate_with_local_model(prompt)
                if isinstance(result, list):
                    return result[:max_concepts]
                elif isinstance(result, dict) and "concepts" in result:
                    return result["concepts"][:max_concepts]
                else:
                    return self._fallback_extract_concepts(text)[:max_concepts]
            else:
                # Fallback to simple extraction without LLM
                return self._fallback_extract_concepts(text)[:max_concepts]
        except Exception as e:
            logger.error(f"Error in LLM concept extraction: {e}")
            return self._fallback_extract_concepts(text)[:max_concepts]
            
    def _fallback_extract_concepts(self, text: str) -> List[str]:
        """Extract concepts using simple heuristics when LLM is unavailable."""
        # Import here to avoid circular imports
        from .entity_extraction import extract_entities
        
        return extract_entities(text)
    
    def _extract_theme(self, contexts: List[Dict]) -> str:
        """Extract a simple theme from a list of contexts without using LLM."""
        # Count entity occurrences
        entity_counts = {}
        for context in contexts:
            for entity in context.get("entities", []):
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
                
        # Find most common entities
        if entity_counts:
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            return " & ".join([entity for entity, _ in top_entities])
        else:
            return f"Cluster of {len(contexts)} contexts"
