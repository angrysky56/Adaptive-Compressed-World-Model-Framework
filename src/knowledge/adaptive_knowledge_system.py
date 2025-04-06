"""
Adaptive Compressed Knowledge Representation System

This system implements the core components of an adaptive, compressed world model
framework focused on efficient knowledge representation and retrieval.

Key components:
1. CompressedContextPack - Handles efficient storage of knowledge chunks
2. DynamicContextGraph - Manages relationships between context packs
3. EventTriggerSystem - Controls when updates and expansions occur
4. StorageHierarchy - Manages caching and long-term storage
"""

import networkx as nx
import redis
import asyncio
import numpy as np
import uuid
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Any, Optional, Tuple
import json
import time

class CompressedContextPack:
    """Handles the compression and expansion of knowledge contexts"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model for embedding generation"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.critical_entities = []
        
    def compress(self, text: str, critical_entities: List[str] = None) -> Dict:
        """
        Compress text into a dense representation
        
        Args:
            text: The raw text to compress
            critical_entities: Important entities that should be preserved
            
        Returns:
            A dictionary containing compressed data and metadata
        """
        # Store critical entities
        if critical_entities:
            self.critical_entities = critical_entities
            
        # Create embedding for the entire text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling to get a fixed-size representation
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Create a summary using chain of density approach
        summary = self._generate_summary(text)
        
        # Create the compressed context pack
        compressed_pack = {
            "id": str(uuid.uuid4()),
            "embedding": embedding.tolist(),
            "summary": summary,
            "critical_entities": self.critical_entities,
            "creation_time": time.time(),
            "last_accessed": time.time(),
            "access_count": 0
        }
        
        return compressed_pack
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a concise summary of the text
        Implementation would use a summarization model or chain of density technique
        
        For now, we'll use a simple placeholder implementation
        """
        # In a real implementation, we would use a transformer model for summarization
        # or implement Chain of Density technique
        words = text.split()
        if len(words) > 100:
            summary = " ".join(words[:100]) + "..."
        else:
            summary = text
            
        # Ensure critical entities are represented in the summary
        for entity in self.critical_entities:
            if entity not in summary and entity in text:
                summary += f" [Critical: {entity}]"
                
        return summary
    
    def expand(self, compressed_pack: Dict) -> str:
        """
        Expand a compressed context pack into its detailed form
        In a real system, this might retrieve the full content from storage
        
        For now, this is a placeholder implementation
        """
        # In a real implementation, we would rebuild or retrieve the original content
        # For now, we'll just return the summary
        return compressed_pack["summary"]


class DynamicContextGraph:
    """Manages the relationships and links between context packs"""
    
    def __init__(self):
        """Initialize the graph structure for context relationships"""
        self.graph = nx.Graph()
        
    def add_context(self, context_pack: Dict) -> None:
        """Add a new context pack to the graph"""
        self.graph.add_node(context_pack["id"], **context_pack)
        
    def link_contexts(self, context_id1: str, context_id2: str, relevance_score: float) -> None:
        """Link two context packs with a relevance score"""
        if not self.graph.has_edge(context_id1, context_id2):
            self.graph.add_edge(context_id1, context_id2, weight=relevance_score)
        else:
            # Update existing edge with new relevance score
            self.graph[context_id1][context_id2]["weight"] = relevance_score
    
    def calculate_relevance(self, context_id1: str, context_id2: str) -> float:
        """Calculate relevance between two contexts based on embedding similarity"""
        if not self.graph.has_node(context_id1) or not self.graph.has_node(context_id2):
            return 0.0
            
        embedding1 = np.array(self.graph.nodes[context_id1]["embedding"])
        embedding2 = np.array(self.graph.nodes[context_id2]["embedding"])
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def get_related_contexts(self, context_id: str, threshold: float = 0.7) -> List[str]:
        """Get related contexts based on graph connections and threshold"""
        if not self.graph.has_node(context_id):
            return []
            
        related_contexts = []
        
        # First, get direct neighbors
        neighbors = list(self.graph.neighbors(context_id))
        for neighbor in neighbors:
            if self.graph[context_id][neighbor]["weight"] >= threshold:
                related_contexts.append(neighbor)
                
        return related_contexts
    
    def expand_subgraph(self, context_ids: List[str], max_depth: int = 2) -> nx.Graph:
        """Return a subgraph containing the specified contexts and their neighbors up to max_depth"""
        if not context_ids:
            return nx.Graph()
            
        # Create a set to track the nodes to include
        nodes_to_include = set(context_ids)
        
        # Add neighbors up to max_depth
        current_nodes = set(context_ids)
        for _ in range(max_depth):
            new_nodes = set()
            for node in current_nodes:
                if self.graph.has_node(node):
                    neighbors = set(self.graph.neighbors(node))
                    new_nodes.update(neighbors)
            nodes_to_include.update(new_nodes)
            current_nodes = new_nodes
            
        # Create the subgraph
        return self.graph.subgraph(nodes_to_include)


class EventTriggerSystem:
    """Controls when updates and expansions of the knowledge model occur"""
    
    def __init__(self, initial_threshold: float = 0.5, adaptation_rate: float = 0.05):
        """Initialize the event trigger system with threshold parameters"""
        self.thresholds = {}  # Context-specific thresholds
        self.default_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.event_history = []
        
    def set_threshold(self, context_id: str, threshold: float) -> None:
        """Set a specific threshold for a context"""
        self.thresholds[context_id] = threshold
        
    def get_threshold(self, context_id: str) -> float:
        """Get the threshold for a specific context or return the default"""
        return self.thresholds.get(context_id, self.default_threshold)
    
    def should_trigger_update(self, context_id: str, change_magnitude: float) -> bool:
        """Determine if an update should be triggered based on threshold"""
        threshold = self.get_threshold(context_id)
        return change_magnitude > threshold
    
    def record_event(self, context_id: str, change_magnitude: float, was_triggered: bool) -> None:
        """Record an event to adapt thresholds over time"""
        self.event_history.append({
            "context_id": context_id,
            "change_magnitude": change_magnitude,
            "was_triggered": was_triggered,
            "timestamp": time.time()
        })
        
        # Adapt threshold based on recent history
        self._adapt_threshold(context_id)
    
    def _adapt_threshold(self, context_id: str) -> None:
        """Adapt the threshold based on recent event history"""
        # Get recent events for this context
        recent_events = [e for e in self.event_history[-50:] if e["context_id"] == context_id]
        
        if len(recent_events) < 10:
            return  # Not enough data to adapt
            
        # Calculate the average change magnitude
        avg_magnitude = sum(e["change_magnitude"] for e in recent_events) / len(recent_events)
        
        # Adjust threshold to be slightly below the average magnitude
        # to trigger on significant changes but avoid constant triggering
        current_threshold = self.get_threshold(context_id)
        new_threshold = current_threshold + self.adaptation_rate * (avg_magnitude * 0.8 - current_threshold)
        
        # Update the threshold
        self.set_threshold(context_id, new_threshold)


class StorageHierarchy:
    """Manages the hierarchical storage of context packs"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, db: int = 0):
        """Initialize the storage hierarchy with Redis for caching"""
        self.cache = redis.Redis(host=redis_host, port=redis_port, db=db)
        self.long_term_storage = {}  # Placeholder for actual long-term storage
        
    def store(self, context_pack: Dict) -> None:
        """Store a context pack in the appropriate storage tier"""
        context_id = context_pack["id"]
        
        # Store in cache for quick access
        self.cache.set(f"context:{context_id}", json.dumps(context_pack), ex=3600)  # 1 hour expiry
        
        # Store in long-term storage
        self.long_term_storage[context_id] = context_pack
    
    async def retrieve(self, context_id: str) -> Optional[Dict]:
        """Retrieve a context pack, preferring cache if available"""
        # Try to get from cache first
        cached_data = self.cache.get(f"context:{context_id}")
        
        if cached_data:
            context_pack = json.loads(cached_data)
            # Update access metadata
            context_pack["last_accessed"] = time.time()
            context_pack["access_count"] += 1
            # Refresh cache
            self.cache.set(f"context:{context_id}", json.dumps(context_pack), ex=3600)
            return context_pack
            
        # If not in cache, try long-term storage
        if context_id in self.long_term_storage:
            context_pack = self.long_term_storage[context_id]
            # Update access metadata
            context_pack["last_accessed"] = time.time()
            context_pack["access_count"] += 1
            # Add to cache for future quick access
            self.cache.set(f"context:{context_id}", json.dumps(context_pack), ex=3600)
            return context_pack
            
        return None
    
    def prune_storage(self, max_age_days: int = 30, min_access_count: int = 5) -> None:
        """Remove old or rarely accessed context packs from long-term storage"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        contexts_to_remove = []
        
        for context_id, context_pack in self.long_term_storage.items():
            age = current_time - context_pack["creation_time"]
            access_count = context_pack["access_count"]
            
            if age > max_age_seconds and access_count < min_access_count:
                contexts_to_remove.append(context_id)
                
        # Remove the identified contexts
        for context_id in contexts_to_remove:
            del self.long_term_storage[context_id]


class AdaptiveKnowledgeSystem:
    """Main class integrating all components of the knowledge representation system"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the complete knowledge system"""
        self.compressor = CompressedContextPack(model_name)
        self.context_graph = DynamicContextGraph()
        self.event_trigger = EventTriggerSystem()
        self.storage = StorageHierarchy()
        
    async def add_knowledge(self, text: str, critical_entities: List[str] = None) -> str:
        """Add new knowledge to the system, returning the context ID"""
        # Compress the knowledge
        context_pack = self.compressor.compress(text, critical_entities)
        
        # Add to the context graph
        self.context_graph.add_context(context_pack)
        
        # Calculate relevance and link to related contexts
        for node in self.context_graph.graph.nodes:
            if node != context_pack["id"]:
                relevance = self.context_graph.calculate_relevance(node, context_pack["id"])
                if relevance > 0.5:  # Only link if sufficiently relevant
                    self.context_graph.link_contexts(node, context_pack["id"], relevance)
        
        # Store the context pack
        self.storage.store(context_pack)
        
        return context_pack["id"]
    
    async def query_knowledge(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """Query the knowledge system based on a text query"""
        # Create an embedding for the query
        query_pack = self.compressor.compress(query_text)
        
        # Find relevant contexts
        relevance_scores = {}
        
        for node in self.context_graph.graph.nodes:
            context_data = self.context_graph.graph.nodes[node]
            if "embedding" in context_data:
                embedding = np.array(context_data["embedding"])
                query_embedding = np.array(query_pack["embedding"])
                
                # Calculate similarity
                similarity = np.dot(embedding, query_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(query_embedding))
                relevance_scores[node] = float(similarity)
        
        # Sort by relevance
        sorted_contexts = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
        
        # Retrieve the context packs
        results = []
        for context_id, score in sorted_contexts:
            context_pack = await self.storage.retrieve(context_id)
            if context_pack:
                # Add the relevance score
                context_pack["relevance_score"] = score
                results.append(context_pack)
        
        return results
    
    async def expand_knowledge(self, context_id: str) -> Dict:
        """Expand a compressed context into its detailed form"""
        # Retrieve the context pack
        context_pack = await self.storage.retrieve(context_id)
        
        if not context_pack:
            return {"error": "Context not found"}
            
        # Expand the context
        expanded_text = self.compressor.expand(context_pack)
        
        # Get related contexts
        related_context_ids = self.context_graph.get_related_contexts(context_id)
        related_contexts = []
        
        for related_id in related_context_ids:
            related_pack = await self.storage.retrieve(related_id)
            if related_pack:
                related_contexts.append({
                    "id": related_id,
                    "summary": related_pack["summary"],
                    "relevance": self.context_graph.graph[context_id][related_id]["weight"]
                })
        
        # Return the expanded knowledge with related contexts
        return {
            "id": context_id,
            "expanded_content": expanded_text,
            "related_contexts": related_contexts
        }
    
    async def update_knowledge(self, context_id: str, new_text: str) -> bool:
        """Update existing knowledge when a significant change is detected"""
        # Retrieve the existing context
        existing_context = await self.storage.retrieve(context_id)
        
        if not existing_context:
            return False
            
        # Create a temporary compression of the new text
        new_context = self.compressor.compress(new_text, existing_context.get("critical_entities", []))
        
        # Calculate the change magnitude (using embedding distance)
        existing_embedding = np.array(existing_context["embedding"])
        new_embedding = np.array(new_context["embedding"])
        
        change_magnitude = np.linalg.norm(existing_embedding - new_embedding)
        
        # Check if update should be triggered
        should_update = self.event_trigger.should_trigger_update(context_id, change_magnitude)
        
        # Record the event for threshold adaptation
        self.event_trigger.record_event(context_id, change_magnitude, should_update)
        
        if should_update:
            # Preserve the original ID and metadata
            new_context["id"] = context_id
            new_context["creation_time"] = existing_context["creation_time"]
            new_context["access_count"] = existing_context["access_count"]
            
            # Update the graph and storage
            self.context_graph.graph.nodes[context_id].update(new_context)
            self.storage.store(new_context)
            
            # Update relationships
            for node in self.context_graph.graph.nodes:
                if node != context_id:
                    relevance = self.context_graph.calculate_relevance(node, context_id)
                    self.context_graph.link_contexts(node, context_id, relevance)
                    
            return True
        
        return False

# Usage example
async def main():
    # Initialize the system
    knowledge_system = AdaptiveKnowledgeSystem()
    
    # Add some knowledge
    context_id1 = await knowledge_system.add_knowledge(
        "Artificial intelligence is the simulation of human intelligence by machines. "
        "It includes machine learning, natural language processing, and computer vision.",
        ["artificial intelligence", "machine learning", "natural language processing"]
    )
    
    context_id2 = await knowledge_system.add_knowledge(
        "Machine learning is a subset of AI that involves training algorithms on data. "
        "Common techniques include neural networks, decision trees, and support vector machines.",
        ["machine learning", "neural networks", "algorithms"]
    )
    
    # Query the knowledge
    results = await knowledge_system.query_knowledge("How does AI relate to neural networks?")
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (Relevance: {result['relevance_score']:.2f}):")
        print(f"Summary: {result['summary']}\n")
    
    # Expand a context
    expanded = await knowledge_system.expand_knowledge(context_id1)
    print(f"Expanded content: {expanded['expanded_content']}")
    print("Related contexts:")
    for related in expanded['related_contexts']:
        print(f"- {related['summary']} (Relevance: {related['relevance']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
