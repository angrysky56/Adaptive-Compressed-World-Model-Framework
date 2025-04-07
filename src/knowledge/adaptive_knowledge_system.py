"""
Adaptive Knowledge System

This module integrates the CompressedContextPack, EventTriggerSystem, and DynamicContextGraph
components to provide a complete adaptive, compressed knowledge representation system.
"""

import asyncio
import time
import uuid
import os
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import numpy as np

# Import components
from .context_pack import CompressedContextPack
from .event_trigger import EventTriggerSystem
from .context_graph import DynamicContextGraph


class StorageHierarchy:
    """
    Manages the hierarchical storage of context packs.
    
    Implements a caching system for frequently accessed contexts and manages
    long-term storage for less frequently used data.
    """
    
    def __init__(self, cache_size: int = 100, cache_expiry: int = 3600,
                 storage_dir: str = None):
        """
        Initialize the storage hierarchy.
        
        Args:
            cache_size: Maximum number of items to keep in the cache
            cache_expiry: Time in seconds before cached items expire
            storage_dir: Directory for long-term storage (None for in-memory only)
        """
        self.cache = {}  # In-memory cache
        self.cache_size = cache_size
        self.cache_expiry = cache_expiry
        self.storage_dir = storage_dir
        self.long_term_storage = {}  # In-memory backup for long-term storage
        
        # Create storage directory if specified and doesn't exist
        if storage_dir and not os.path.exists(storage_dir):
            try:
                os.makedirs(storage_dir)
            except Exception as e:
                print(f"Error creating storage directory: {e}")
                self.storage_dir = None
    
    async def store(self, context_pack: Dict) -> None:
        """
        Store a context pack in the appropriate storage tier.
        
        Args:
            context_pack: The context pack to store
        """
        context_id = context_pack["id"]
        
        # Store in cache
        self.cache[context_id] = {
            "data": context_pack,
            "expiry": time.time() + self.cache_expiry
        }
        
        # Prune cache if it exceeds the size limit
        if len(self.cache) > self.cache_size:
            self._prune_cache()
        
        # Store in long-term storage
        self.long_term_storage[context_id] = context_pack
        
        # Write to file storage if configured
        if self.storage_dir:
            asyncio.create_task(self._write_to_file(context_id, context_pack))
    
    async def retrieve(self, context_id: str) -> Optional[Dict]:
        """
        Retrieve a context pack, preferring cache if available.
        
        Args:
            context_id: ID of the context pack to retrieve
            
        Returns:
            The context pack, or None if not found
        """
        # Try to get from cache first
        if context_id in self.cache:
            cache_entry = self.cache[context_id]
            
            # Check if the entry has expired
            if time.time() < cache_entry["expiry"]:
                # Update expiry time
                self.cache[context_id]["expiry"] = time.time() + self.cache_expiry
                
                # Update access metadata
                context_pack = cache_entry["data"]
                context_pack["last_accessed"] = time.time()
                context_pack["access_count"] = context_pack.get("access_count", 0) + 1
                
                return context_pack
            
            # Entry has expired, remove from cache
            del self.cache[context_id]
        
        # Try in-memory long-term storage
        if context_id in self.long_term_storage:
            context_pack = self.long_term_storage[context_id]
            
            # Update access metadata
            context_pack["last_accessed"] = time.time()
            context_pack["access_count"] = context_pack.get("access_count", 0) + 1
            
            # Add to cache for future quick access
            self.cache[context_id] = {
                "data": context_pack,
                "expiry": time.time() + self.cache_expiry
            }
            
            return context_pack
        
        # Try file storage if configured
        if self.storage_dir:
            context_pack = await self._read_from_file(context_id)
            if context_pack:
                # Update in-memory storage
                self.long_term_storage[context_id] = context_pack
                
                # Add to cache
                self.cache[context_id] = {
                    "data": context_pack,
                    "expiry": time.time() + self.cache_expiry
                }
                
                return context_pack
        
        # Not found
        return None
    
    def _prune_cache(self) -> None:
        """Prune the cache by removing expired or least recently used items."""
        current_time = time.time()
        
        # First, remove expired entries
        expired_keys = [k for k, v in self.cache.items() if current_time > v["expiry"]]
        for key in expired_keys:
            del self.cache[key]
            
        # If still over capacity, remove oldest entries
        if len(self.cache) > self.cache_size:
            # Sort by expiry time (oldest first)
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1]["expiry"])
            
            # Remove oldest entries
            for key, _ in sorted_entries[:len(self.cache) - self.cache_size]:
                del self.cache[key]
    
    async def prune_storage(self, max_age_days: int = 30, min_access_count: int = 5) -> List[str]:
        """
        Remove old or rarely accessed context packs from storage.
        
        Args:
            max_age_days: Maximum age in days to keep unused contexts
            min_access_count: Minimum access count to keep old contexts
            
        Returns:
            List of context IDs that were removed
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        contexts_to_remove = []
        
        # Identify contexts to remove
        for context_id, context_pack in self.long_term_storage.items():
            # Calculate age
            creation_time = context_pack.get("creation_time", 0)
            age = current_time - creation_time
            
            # Get access count
            access_count = context_pack.get("access_count", 0)
            
            # Check if it should be removed
            if age > max_age_seconds and access_count < min_access_count:
                contexts_to_remove.append(context_id)
        
        # Remove identified contexts
        for context_id in contexts_to_remove:
            # Remove from long-term storage
            del self.long_term_storage[context_id]
            
            # Remove from cache if present
            if context_id in self.cache:
                del self.cache[context_id]
                
            # Remove from file storage
            if self.storage_dir:
                file_path = os.path.join(self.storage_dir, f"{context_id}.json")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
        
        return contexts_to_remove
    
    async def _write_to_file(self, context_id: str, context_pack: Dict) -> bool:
        """
        Write a context pack to file storage.
        
        Args:
            context_id: ID of the context pack
            context_pack: The context pack to write
            
        Returns:
            True if successful, False otherwise
        """
        if not self.storage_dir:
            return False
            
        file_path = os.path.join(self.storage_dir, f"{context_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(context_pack, f, indent=2)
            return True
        except Exception as e:
            print(f"Error writing context pack to {file_path}: {e}")
            return False
    
    async def _read_from_file(self, context_id: str) -> Optional[Dict]:
        """
        Read a context pack from file storage.
        
        Args:
            context_id: ID of the context pack
            
        Returns:
            The context pack, or None if not found or error
        """
        if not self.storage_dir:
            return None
            
        file_path = os.path.join(self.storage_dir, f"{context_id}.json")
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                context_pack = json.load(f)
            return context_pack
        except Exception as e:
            print(f"Error reading context pack from {file_path}: {e}")
            return None


class AdaptiveKnowledgeSystem:
    """
    Main class integrating all components of the knowledge representation system.
    
    The AdaptiveKnowledgeSystem provides a complete solution for managing compressed
    knowledge representations, with event-triggered updates and dynamic context linking.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 storage_dir: str = None, use_llm: bool = False, ollama_model: str = None):
        """
        Initialize the complete knowledge system.
        
        Args:
            model_name: The name of the sentence transformer model to use
            storage_dir: Directory for long-term storage (None for in-memory only)
            use_llm: Whether to use LLM for enhanced relationship analysis
            ollama_model: Specific Ollama model to use for LLM analysis
        """
        self.compressor = CompressedContextPack(model_name)
        self.context_graph = DynamicContextGraph(use_llm=use_llm, ollama_model=ollama_model)
        self.event_trigger = EventTriggerSystem()
        self.storage = StorageHierarchy(storage_dir=storage_dir)
        self.use_llm = use_llm
        
    async def add_knowledge(self, text: str, critical_entities: List[str] = None, metadata: Dict = None) -> str:
        """
        Add new knowledge to the system.
        
        Args:
            text: The knowledge text to add
            critical_entities: Important entities that should be preserved
            metadata: Additional metadata about the context (e.g., source, file type)
            
        Returns:
            The ID of the added context
        """
        # Compress the knowledge
        context_pack = self.compressor.compress(text, critical_entities)
        
        # Add metadata if provided
        if metadata:
            context_pack.update({
                "metadata": metadata,
                "source": metadata.get("source", "unknown")
            })
        
        # Add to the context graph
        self.context_graph.add_context(context_pack)
        
        # Calculate relevance and link to related contexts
        for node in list(self.context_graph.graph.nodes()):
            if node != context_pack["id"]:
                relevance = self.context_graph.calculate_relevance(node, context_pack["id"])
                if relevance > 0.5:  # Only link if sufficiently relevant
                    self.context_graph.link_contexts(node, context_pack["id"], relevance)
        
        # Store the context pack
        await self.storage.store(context_pack)
        
        return context_pack["id"]
    
    async def query_knowledge(self, query_text: str, max_results: int = 5, 
                            include_explanations: bool = False) -> List[Dict]:
        """
        Query the knowledge system based on a text query.
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            include_explanations: Whether to include LLM-generated explanations
            
        Returns:
            List of context packs matching the query
        """
        # Create a temporary context pack for the query
        query_pack = self.compressor.compress(query_text)
        
        # Calculate relevance to all contexts in the graph
        relevance_scores = {}
        
        for node in self.context_graph.graph.nodes():
            node_data = self.context_graph.graph.nodes[node]
            
            # Skip if missing required data
            if "embedding" not in node_data:
                continue
                
            # Calculate similarity
            embedding = np.array(node_data["embedding"])
            query_embedding = np.array(query_pack["embedding"])
            
            # Normalize embeddings
            embedding_norm = np.linalg.norm(embedding)
            query_norm = np.linalg.norm(query_embedding)
            
            if embedding_norm > 0 and query_norm > 0:
                similarity = np.dot(embedding, query_embedding) / (embedding_norm * query_norm)
                relevance_scores[node] = float(similarity)
        
        # Sort by relevance
        sorted_contexts = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
        
        # Retrieve the context packs
        results = []
        for context_id, score in sorted_contexts:
            context_pack = await self.storage.retrieve(context_id)
            if context_pack:
                # Add the relevance score
                context_pack = context_pack.copy()
                context_pack["relevance_score"] = score
                results.append(context_pack)
        
        return results
    
    async def expand_knowledge(self, context_id: str) -> Dict:
        """
        Expand a compressed context into its detailed form.
        
        Args:
            context_id: ID of the context to expand
            
        Returns:
            Expanded knowledge with related contexts
        """
        # Retrieve the context pack
        context_pack = await self.storage.retrieve(context_id)
        
        if not context_pack:
            return {"error": "Context not found"}
            
        # Expand the context
        expanded = self.compressor.expand(context_pack)
        
        # Get related contexts
        related_context_ids = self.context_graph.get_related_contexts(context_id)
        related_contexts = []
        
        for related_id in related_context_ids:
            related_pack = await self.storage.retrieve(related_id)
            if related_pack:
                # Get the edge data for relevance score
                if self.context_graph.graph.has_edge(context_id, related_id):
                    relevance = self.context_graph.graph[context_id][related_id]["weight"]
                else:
                    relevance = self.context_graph.calculate_relevance(context_id, related_id)
                    
                related_contexts.append({
                    "id": related_id,
                    "summary": related_pack.get("summary", "No summary available"),
                    "relevance": relevance
                })
        
        # Add related contexts to the expanded result
        expanded["related_contexts"] = related_contexts
        
        return expanded
    
    async def update_knowledge(self, context_id: str, new_text: str) -> bool:
        """
        Update existing knowledge when a significant change is detected.
        
        Args:
            context_id: ID of the context to update
            new_text: New text to update with
            
        Returns:
            True if the update was performed, False otherwise
        """
        # Retrieve the existing context
        existing_context = await self.storage.retrieve(context_id)
        
        if not existing_context:
            return False
            
        # Get critical entities from existing context
        critical_entities = existing_context.get("critical_entities", [])
        
        # Detect if the change is significant
        change_magnitude, is_significant = self.compressor.detect_significant_change(
            existing_context, new_text
        )
        
        # Record the event for threshold adaptation
        self.event_trigger.record_event(
            context_id,
            change_magnitude,
            is_significant
        )
        
        # Check if update should be triggered
        should_update = self.event_trigger.should_trigger_update(context_id, change_magnitude)
        
        if should_update:
            # Create a new compressed pack with the updated text
            new_context = self.compressor.compress(new_text, critical_entities)
            
            # Preserve the original ID and metadata
            new_context["id"] = context_id
            new_context["creation_time"] = existing_context.get("creation_time", time.time())
            new_context["access_count"] = existing_context.get("access_count", 0)
            new_context["version"] = existing_context.get("version", 0) + 1
            
            # Update the graph and storage
            self.context_graph.add_context(new_context)
            await self.storage.store(new_context)
            
            # Update relationships
            for node in self.context_graph.graph.nodes():
                if node != context_id:
                    relevance = self.context_graph.calculate_relevance(node, context_id)
                    if relevance > 0.5:  # Only link if sufficiently relevant
                        self.context_graph.link_contexts(node, context_id, relevance)
                        
            return True
        
        return False
    
    async def remove_knowledge(self, context_id: str) -> bool:
        """
        Remove knowledge from the system.
        
        Args:
            context_id: ID of the context to remove
            
        Returns:
            True if the knowledge was removed, False otherwise
        """
        # Remove from context graph
        removed_from_graph = self.context_graph.remove_context(context_id)
        
        # Remove from long-term storage
        contexts_removed = await self.storage.prune_storage(
            max_age_days=0,
            min_access_count=float('inf')  # Force removal regardless of access count
        )
        
        return removed_from_graph and context_id in contexts_removed
    
    async def search_knowledge(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """
        Search for knowledge using more advanced criteria than simple queries.
        
        This method provides more options for filtering and sorting results.
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            
        Returns:
            List of matching contexts with relevance scores
        """
        # For now, this is similar to query_knowledge but could be extended with more features
        results = await self.query_knowledge(query_text, max_results)
        return results
    
    async def get_knowledge_history(self, context_id: str) -> List[Dict]:
        """
        Get the update history for a specific context.
        
        This would retrieve previous versions of the context if they are stored.
        In a full implementation, this would access version history from storage.
        
        Args:
            context_id: ID of the context to get history for
            
        Returns:
            List of context versions in chronological order
        """
        # In a real implementation, this would retrieve version history
        # For now, just return the current version
        context = await self.storage.retrieve(context_id)
        if not context:
            return []
            
        return [context]
    
    async def generate_knowledge_summary(self, context_ids: List[str] = None) -> Dict:
        """
        Generate a summary of the knowledge in the system.
        
        Args:
            context_ids: Optional list of context IDs to summarize, or None for all
            
        Returns:
            Summary information about the knowledge system
        """
        # If no context IDs provided, use all contexts
        if context_ids is None:
            context_ids = list(self.context_graph.graph.nodes())
            
        # Count total contexts
        total_contexts = len(context_ids)
        
        # Get graph statistics
        graph_stats = self.context_graph.get_graph_stats()
        
        # Calculate average context metrics
        total_size = 0
        total_compression_ratio = 0
        total_access_count = 0
        creation_times = []
        
        for context_id in context_ids:
            context = await self.storage.retrieve(context_id)
            if context:
                total_size += context.get("compressed_length", 0)
                total_compression_ratio += context.get("compression_ratio", 0)
                total_access_count += context.get("access_count", 0)
                creation_times.append(context.get("creation_time", 0))
                
        # Calculate averages
        avg_size = total_size / max(1, total_contexts)
        avg_compression_ratio = total_compression_ratio / max(1, total_contexts)
        avg_access_count = total_access_count / max(1, total_contexts)
        
        # Get oldest and newest contexts
        oldest_time = min(creation_times) if creation_times else 0
        newest_time = max(creation_times) if creation_times else 0
        
        # Compile the summary
        summary = {
            "total_contexts": total_contexts,
            "graph_stats": graph_stats,
            "avg_context_size": avg_size,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_access_count": avg_access_count,
            "oldest_context_time": oldest_time,
            "newest_context_time": newest_time
        }
        
        return summary
    
    async def export_knowledge_graph(self, filepath: str) -> bool:
        """
        Export the knowledge graph to a file.
        
        Args:
            filepath: Path to the file to export to
            
        Returns:
            True if successful, False otherwise
        """
        return self.context_graph.save_to_json(filepath)
    
    async def import_knowledge_graph(self, filepath: str) -> bool:
        """
        Import a knowledge graph from a file.
        
        Args:
            filepath: Path to the file to import from
            
        Returns:
            True if successful, False otherwise
        """
        return self.context_graph.load_from_json(filepath)
    
    async def enhance_knowledge_links(self, min_similarity: float = 0.6, max_suggestions: int = 10) -> List[Dict]:
        """
        Enhance knowledge connections using LLM analysis.
        
        This method uses language models to suggest potential connections between
        contexts that might not be obvious through simple similarity measures.
        
        Args:
            min_similarity: Minimum similarity threshold for suggested connections
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries with information about suggested connections
        """
        if not self.use_llm:
            return [{
                "success": False,
                "reason": "LLM analysis is not enabled"
            }]
            
        # Use the DynamicContextGraph to enhance connections
        suggested_links = self.context_graph.enhance_connections(
            min_similarity=min_similarity,
            max_suggestions=max_suggestions
        )
        
        return suggested_links
    
    async def maintain_storage(self) -> Dict:
        """
        Perform maintenance on the storage system.
        
        This includes pruning old or rarely accessed contexts and optimizing storage.
        
        Returns:
            Dictionary with maintenance results
        """
        # Prune storage
        removed_contexts = await self.storage.prune_storage()
        
        # Get current storage stats
        total_contexts = len(self.context_graph.graph.nodes())
        cache_size = len(self.storage.cache)
        long_term_size = len(self.storage.long_term_storage)
        
        # Compile results
        results = {
            "removed_contexts": len(removed_contexts),
            "total_contexts_remaining": total_contexts,
            "cache_size": cache_size,
            "long_term_storage_size": long_term_size
        }
        
        return results
    
    async def visualize_knowledge_graph(self, highlight_contexts: List[str] = None,
                                       output_file: str = None) -> Optional[Any]:
        """
        Visualize the knowledge graph.
        
        Args:
            highlight_contexts: Optional list of context IDs to highlight
            output_file: Optional file to save the visualization to
            
        Returns:
            Visualization object or None
        """
        # Use the DynamicContextGraph's visualize method
        fig = self.context_graph.visualize(
            highlight_nodes=highlight_contexts,
            title="Knowledge Graph Visualization",
            show=output_file is None
        )
        
        # Save to file if requested
        if fig and output_file:
            try:
                fig.savefig(output_file)
                return True
            except Exception as e:
                print(f"Error saving visualization to {output_file}: {e}")
                return False
                
        return fig
