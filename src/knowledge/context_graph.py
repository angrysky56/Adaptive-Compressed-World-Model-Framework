"""
Dynamic Context Graph

This module provides the functionality for managing relationships between context packs
using a graph-based approach that enables efficient retrieval and expansion of related contexts.
"""

import networkx as nx
import numpy as np
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import the LLM context linker
try:
    from .llm_context_linker import LLMContextLinker
except ImportError:
    LLMContextLinker = None

class DynamicContextGraph:
    """
    Manages the relationships and links between context packs.
    
    The DynamicContextGraph maintains a graph structure where nodes represent context packs
    and edges represent relationships between them. It enables efficient retrieval and
    traversal of related contexts.
    """
    
    def __init__(self, use_llm: bool = False, ollama_model: str = None):
        """
        Initialize the graph structure for context relationships.
        
        Args:
            use_llm: Whether to use LLM for enhanced relationship analysis
            ollama_model: Specific Ollama model to use for LLM analysis
        """
        self.graph = nx.Graph()
        
        # Track metadata about context relationships
        self.relationship_types = defaultdict(int)
        self.last_update_time = time.time()
        
        # Initialize LLM context linker if requested
        self.use_llm = use_llm
        self.llm_linker = None
        
        if use_llm and LLMContextLinker is not None:
            try:
                self.llm_linker = LLMContextLinker(
                    use_ollama=True,
                    ollama_model=ollama_model
                )
                logger.info("LLM Context Linker initialized for enhanced relationship analysis")
            except Exception as e:
                logger.error(f"Failed to initialize LLM Context Linker: {e}")
        
    def add_context(self, context_pack: Dict) -> None:
        """
        Add a new context pack to the graph.
        
        Args:
            context_pack: The context pack to add
        """
        context_id = context_pack["id"]
        
        # Add node with all properties from the context pack
        self.graph.add_node(context_id, **context_pack)
        
        # Track last update time
        self.last_update_time = time.time()
        
    def remove_context(self, context_id: str) -> bool:
        """
        Remove a context pack from the graph.
        
        Args:
            context_id: ID of the context pack to remove
            
        Returns:
            True if the context was removed, False otherwise
        """
        if context_id in self.graph:
            self.graph.remove_node(context_id)
            self.last_update_time = time.time()
            return True
        return False
        
    def link_contexts(self, context_id1: str, context_id2: str, relevance_score: float, 
                     relationship_type: str = None, bidirectional: bool = True,
                     shared_entities: List[str] = None, use_llm_analysis: bool = None) -> Dict:
        """
        Link two context packs with a relevance score and relationship type.
        
        Args:
            context_id1: ID of the first context pack
            context_id2: ID of the second context pack
            relevance_score: Strength of the relationship (0-1)
            relationship_type: Type of relationship between the contexts. If None, it will
                              be automatically determined based on content analysis.
            bidirectional: Whether the relationship is bidirectional (symmetric)
            shared_entities: Optional list of shared entities between the contexts
            use_llm_analysis: Whether to use LLM for relationship analysis (overrides default)
            
        Returns:
            Dictionary with information about the created/updated relationship
        """
        if not self.graph.has_node(context_id1) or not self.graph.has_node(context_id2):
            raise ValueError(f"Cannot link contexts: one or both context IDs not found in graph")
            
        # Don't link a context to itself
        if context_id1 == context_id2:
            return {"success": False, "reason": "Cannot link a context to itself"}
        
        # Get node data for relationship analysis
        node1 = self.graph.nodes[context_id1]
        node2 = self.graph.nodes[context_id2]
        
        # Determine whether to use LLM analysis
        use_llm = self.use_llm if use_llm_analysis is None else use_llm_analysis
        
        # Perform relationship analysis
        relationship_info = {}
        
        if use_llm and self.llm_linker is not None:
            # Use LLM for enhanced relationship analysis
            analysis = self.llm_linker.analyze_relationship(node1, node2)
            
            # Extract information from the analysis
            if relationship_type is None:
                relationship_type = analysis.get("relationship_type", "is_related_to")
                
            if shared_entities is None:
                shared_entities = analysis.get("bridge_concepts", [])
                
            # Store the explanation
            relationship_info["explanation"] = analysis.get("explanation", "")
            
            # Consider using the LLM's relevance score
            llm_relevance = analysis.get("strength")
            if llm_relevance is not None and relevance_score is None:
                relevance_score = llm_relevance
                
        else:
            # Use standard relationship determination
            if relationship_type is None:
                relationship_type = self._determine_relationship_type(node1, node2)
            
            # Calculate shared entities if not provided
            if shared_entities is None:
                entities1 = set(node1.get("critical_entities", []))
                entities2 = set(node2.get("critical_entities", []))
                shared_entities = list(entities1.intersection(entities2))
        
        # Update or create the edge
        edge_attrs = {
            "weight": relevance_score,
            "type": relationship_type,
            "shared_entities": shared_entities,
            "update_time": time.time()
        }
        
        if relationship_info:
            edge_attrs["relationship_info"] = relationship_info
        
        if not self.graph.has_edge(context_id1, context_id2):
            edge_attrs["creation_time"] = time.time()
            self.graph.add_edge(context_id1, context_id2, **edge_attrs)
            relationship_info["action"] = "created"
        else:
            # Update existing edge while preserving creation time
            creation_time = self.graph[context_id1][context_id2].get("creation_time", time.time())
            edge_attrs["creation_time"] = creation_time
            self.graph[context_id1][context_id2].update(edge_attrs)
            relationship_info["action"] = "updated"
            
        # Track relationship type
        self.relationship_types[relationship_type] += 1
        
        # Update last update time
        self.last_update_time = time.time()
        
        # Return information about the relationship
        return {
            "success": True,
            "context1_id": context_id1,
            "context2_id": context_id2,
            "relationship_type": relationship_type,
            "relevance_score": relevance_score,
            "shared_entities": shared_entities,
            "action": relationship_info.get("action", "created")
        }
    
    def _determine_relationship_type(self, node1: Dict, node2: Dict) -> str:
        """
        Determine the relationship type between two contexts based on their content.
        
        Args:
            node1: Data for the first context node
            node2: Data for the second context node
            
        Returns:
            String describing the relationship type
        """
        # Extract key information from both nodes
        entities1 = set(node1.get("critical_entities", []))
        entities2 = set(node2.get("critical_entities", []))
        
        # Get summaries and key phrases
        summary1 = node1.get("summary", "")
        summary2 = node2.get("summary", "")
        phrases1 = set(node1.get("key_phrases", []))
        phrases2 = set(node2.get("key_phrases", []))
        
        # Calculate entity overlap
        shared_entities = entities1.intersection(entities2)
        total_entities = entities1.union(entities2)
        entity_overlap_ratio = len(shared_entities) / max(1, len(total_entities))
        
        # Check for hierarchical relationships
        is_subset = False
        is_superset = False
        
        if len(entities1) > 0 and len(entities2) > 0:
            is_subset = entities1.issubset(entities2)
            is_superset = entities1.issuperset(entities2)
        
        # Look for specific types of relationships
        if is_subset and not is_superset:
            return "is_subtopic_of"
        elif is_superset and not is_subset:
            return "is_supertopic_of"
        elif entity_overlap_ratio > 0.7:
            return "is_strongly_related"
        elif entity_overlap_ratio > 0.4:
            return "is_related"
        elif len(shared_entities) > 0:
            return "shares_concepts"
        else:
            # Check for semantic similarity without entity overlap
            embedding1 = node1.get("embedding")
            embedding2 = node2.get("embedding")
            
            if embedding1 and embedding2:
                # Use embedding similarity to determine relationship type
                embedding1 = np.array(embedding1)
                embedding2 = np.array(embedding2)
                similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                
                if similarity > 0.8:
                    return "is_semantically_similar"
                elif similarity > 0.6:
                    return "is_contextually_related"
            
            return "is_weakly_connected"
    
    def calculate_relevance(self, context_id1: str, context_id2: str) -> float:
        """
        Calculate relevance between two contexts based on their properties.
        
        This implementation considers multiple factors:
        - Embedding similarity (semantic relevance)
        - Shared entities (concept overlap)
        - Key phrase similarity (topic relevance)
        - Temporal relationship (recency and creation order)
        
        The final relevance score is a weighted combination of these factors.
        
        Args:
            context_id1: ID of the first context
            context_id2: ID of the second context
            
        Returns:
            Relevance score between 0 and 1
        """
        if not self.graph.has_node(context_id1) or not self.graph.has_node(context_id2):
            return 0.0
            
        # Get node data
        node1 = self.graph.nodes[context_id1]
        node2 = self.graph.nodes[context_id2]
        
        # Initialize score components
        embedding_similarity = 0.0
        entity_similarity = 0.0
        phrase_similarity = 0.0
        temporal_factor = 0.0
        
        # 1. Calculate embedding similarity (semantic relevance)
        if "embedding" in node1 and "embedding" in node2:
            embedding1 = np.array(node1["embedding"])
            embedding2 = np.array(node2["embedding"])
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 > 0 and norm2 > 0:
                embedding_similarity = dot_product / (norm1 * norm2)
        
        # 2. Calculate entity similarity (concept overlap)
        entities1 = set(node1.get("critical_entities", []))
        entities2 = set(node2.get("critical_entities", []))
        
        if entities1 and entities2:
            # Calculate Jaccard similarity
            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))
            
            if union > 0:
                entity_similarity = intersection / union
                
                # Boost score for complete subset/superset relationships
                if entities1.issubset(entities2) or entities2.issubset(entities1):
                    entity_similarity += 0.1  # Boost for hierarchical relationships
        
        # 3. Calculate key phrase similarity
        phrases1 = set(node1.get("key_phrases", []))
        phrases2 = set(node2.get("key_phrases", []))
        
        if phrases1 and phrases2:
            # Calculate similarity between key phrases
            phrase_intersection = len(phrases1.intersection(phrases2))
            phrase_union = len(phrases1.union(phrases2))
            
            if phrase_union > 0:
                phrase_similarity = phrase_intersection / phrase_union
        
        # 4. Calculate temporal factor
        creation_time1 = node1.get("creation_time", 0)
        creation_time2 = node2.get("creation_time", 0)
        
        if creation_time1 > 0 and creation_time2 > 0:
            # Calculate temporal proximity (normalized)
            time_diff = abs(creation_time1 - creation_time2)
            max_time_diff = 30 * 24 * 60 * 60  # 30 days in seconds
            temporal_factor = max(0, 1 - (time_diff / max_time_diff))
        
        # Combine factors with weights
        weights = {
            "embedding": 0.6,
            "entity": 0.25,
            "phrase": 0.1,
            "temporal": 0.05
        }
        
        relevance = (
            weights["embedding"] * embedding_similarity +
            weights["entity"] * entity_similarity +
            weights["phrase"] * phrase_similarity +
            weights["temporal"] * temporal_factor
        )
        
        # Ensure the result is between 0 and 1
        return max(0.0, min(1.0, float(relevance)))
    
    def get_related_contexts(self, context_id: str, threshold: float = 0.5, max_results: int = 10, 
                           relationship_types: List[str] = None, include_indirect: bool = False, 
                           max_depth: int = 2, include_explanations: bool = False) -> List[Dict]:
        """
        Get related contexts based on graph connections and criteria.
        
        This enhanced implementation supports:
        - Filtering by relationship types
        - Including indirect relationships (multi-hop)
        - Returning rich relationship information
        - LLM-generated explanations of relationships
        
        Args:
            context_id: ID of the context to find related contexts for
            threshold: Minimum relevance score to include
            max_results: Maximum number of results to return
            relationship_types: Optional filter for specific relationship types
            include_indirect: Whether to include indirect (multi-hop) relationships
            max_depth: Maximum path length for indirect relationships
            include_explanations: Whether to include LLM-generated explanations
            
        Returns:
            List of dictionaries with related context information
        """
        if not self.graph.has_node(context_id):
            return []
            
        related_contexts = []
        
        # First, get direct neighbors
        for neighbor, data in self.graph[context_id].items():
            # Check if it meets the threshold
            if data["weight"] < threshold:
                continue
                
            # Check relationship type filter
            if relationship_types and data.get("type") not in relationship_types:
                continue
                
            # Create relationship info
            relation_info = {
                "id": neighbor,
                "relevance": data["weight"],
                "relationship_type": data.get("type", "similarity"),
                "shared_entities": data.get("shared_entities", []),
                "path": [context_id, neighbor],
                "is_direct": True
            }
            
            # Add explanation if requested
            if include_explanations:
                if self.use_llm and self.llm_linker is not None:
                    # Check if explanation already exists in edge data
                    if "relationship_info" in data and "explanation" in data["relationship_info"]:
                        relation_info["explanation"] = data["relationship_info"]["explanation"]
                    else:
                        # Generate explanation using relationship_explanation method
                        explanation_result = self.get_relationship_explanation(context_id, neighbor)
                        relation_info["explanation"] = explanation_result.get("explanation", "")
                else:
                    # Use simple rule-based explanation
                    explanation_result = self.get_relationship_explanation(context_id, neighbor)
                    relation_info["explanation"] = explanation_result.get("explanation", "")
            
            # Add to related contexts
            related_contexts.append(relation_info)
                
        # If requested, include indirect relationships
        if include_indirect:
            # Use breadth-first search to find multi-hop relationships
            explored = set([context_id] + [rc["id"] for rc in related_contexts])
            frontier = [(neighbor, 2) for neighbor in self.graph.neighbors(context_id)]
            
            while frontier and max_depth > 1:
                current, depth = frontier.pop(0)
                
                if current in explored:
                    continue
                    
                explored.add(current)
                
                # Calculate indirect relevance through path analysis
                paths = list(nx.all_simple_paths(self.graph, context_id, current, cutoff=max_depth))
                
                if not paths:
                    continue
                    
                # Calculate the total relevance through the best path
                best_path = None
                best_path_score = 0
                
                for path in paths:
                    path_score = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.graph[path[i]][path[i+1]]
                        path_score *= edge_data.get("weight", 0)
                        
                    if path_score > best_path_score:
                        best_path_score = path_score
                        best_path = path
                
                # Only add if it meets the threshold
                if best_path_score >= threshold:
                    # Gather shared entities along the path
                    all_shared_entities = set()
                    for i in range(len(best_path) - 1):
                        edge_shared = self.graph[best_path[i]][best_path[i+1]].get("shared_entities", [])
                        all_shared_entities.update(edge_shared)
                    
                    # Determine relationship type for the indirect connection
                    relationship_type = "indirect_connection"
                    
                    # Create relation info
                    relation_info = {
                        "id": current,
                        "relevance": best_path_score,
                        "relationship_type": relationship_type,
                        "shared_entities": list(all_shared_entities),
                        "path": best_path,
                        "is_direct": False
                    }
                    
                    # Add explanation if requested
                    if include_explanations:
                        # Generate explanation for indirect relationship
                        if self.use_llm and self.llm_linker is not None:
                            # Get path node summaries
                            path_summaries = []
                            for path_node in best_path:
                                if self.graph.has_node(path_node):
                                    node_data = self.graph.nodes[path_node]
                                    summary = node_data.get("summary", f"Context {path_node}")
                                    path_summaries.append(summary)
                                    
                            # Create a simplified explanation
                            if len(path_summaries) > 2:
                                via_nodes = path_summaries[1:-1]
                                if len(via_nodes) == 1:
                                    via_text = f"via '{via_nodes[0]}'"
                                else:
                                    via_text = f"via {len(via_nodes)} intermediate contexts"
                                    
                                explanation = f"'{path_summaries[0]}' is indirectly connected to '{path_summaries[-1]}' {via_text} with overall relevance {best_path_score:.2f}."
                                relation_info["explanation"] = explanation
                        else:
                            # Use simple rule-based explanation
                            path_length = len(best_path) - 1
                            explanation = f"Connected through {path_length} intermediate contexts with overall relevance {best_path_score:.2f}."
                            relation_info["explanation"] = explanation
                    
                    # Add to related contexts
                    related_contexts.append(relation_info)
                
                # Add neighbors to frontier if we're not at max depth
                if depth < max_depth:
                    frontier.extend([(nbr, depth + 1) for nbr in self.graph.neighbors(current) 
                                     if nbr not in explored])
        
        # Sort by relevance score
        related_contexts.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Limit to max_results
        return related_contexts[:max_results]
    
    def expand_subgraph(self, context_ids: List[str], max_depth: int = 2) -> nx.Graph:
        """
        Return a subgraph containing the specified contexts and their neighbors up to max_depth.
        
        Args:
            context_ids: List of context IDs to include
            max_depth: Maximum depth of neighbors to include
            
        Returns:
            NetworkX subgraph
        """
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
    
    def find_path(self, source_id: str, target_id: str, max_path_length: int = 4) -> List[str]:
        """
        Find the shortest path between two contexts.
        
        Args:
            source_id: ID of the source context
            target_id: ID of the target context
            max_path_length: Maximum length of path to consider
            
        Returns:
            List of context IDs forming the path, or empty list if no path exists
        """
        if not (self.graph.has_node(source_id) and self.graph.has_node(target_id)):
            return []
            
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source=source_id, target=target_id, weight='weight')
            
            # Check if path is within max length
            if len(path) <= max_path_length + 1:  # +1 because path includes start and end nodes
                return path
            else:
                return []
        except nx.NetworkXNoPath:
            return []
    
    def get_relationship_explanation(self, context_id1: str, context_id2: str) -> Dict:
        """
        Generate a human-readable explanation of the relationship between two contexts.
        
        Args:
            context_id1: ID of the first context
            context_id2: ID of the second context
            
        Returns:
            Dictionary with relationship explanation
        """
        if not (self.graph.has_node(context_id1) and self.graph.has_node(context_id2)):
            return {"explanation": "No relationship exists - one or both contexts do not exist"}
        
        # Check if there's a direct relationship
        if self.graph.has_edge(context_id1, context_id2):
            edge_data = self.graph[context_id1][context_id2]
            
            # Get node data
            node1 = self.graph.nodes[context_id1]
            node2 = self.graph.nodes[context_id2]
            
            # Get summaries
            summary1 = node1.get("summary", "Context 1")
            summary2 = node2.get("summary", "Context 2")
            
            # Create a natural language explanation
            relationship_type = edge_data.get("type", "similarity")
            relationship_strength = edge_data.get("weight", 0)
            shared_entities = edge_data.get("shared_entities", [])
            
            # Map relationship types to human-readable descriptions
            relationship_descriptions = {
                "similarity": "is similar to",
                "is_subtopic_of": "is a subtopic of",
                "is_supertopic_of": "is a supertopic of",
                "is_strongly_related": "is strongly related to",
                "is_related": "is related to",
                "shares_concepts": "shares concepts with",
                "is_semantically_similar": "is semantically similar to",
                "is_contextually_related": "is contextually related to",
                "is_weakly_connected": "is weakly connected to"
            }
            
            relation_desc = relationship_descriptions.get(relationship_type, "is connected to")
            strength_desc = ""
            
            if relationship_strength > 0.8:
                strength_desc = "very strongly"
            elif relationship_strength > 0.6:
                strength_desc = "strongly"
            elif relationship_strength > 0.4:
                strength_desc = "moderately"
            elif relationship_strength > 0.2:
                strength_desc = "weakly"
            else:
                strength_desc = "very weakly"
            
            entity_desc = ""
            if shared_entities:
                if len(shared_entities) == 1:
                    entity_desc = f" through the shared concept of '{shared_entities[0]}'"
                else:
                    entity_list = "', '".join(shared_entities[:3])
                    if len(shared_entities) > 3:
                        entity_list += "', and others"
                    entity_desc = f" through shared concepts including '{entity_list}"
            
            explanation = f"Context 1 {strength_desc} {relation_desc} Context 2{entity_desc}."
            
            return {
                "explanation": explanation,
                "context1_summary": summary1,
                "context2_summary": summary2,
                "relationship_type": relationship_type,
                "relationship_strength": relationship_strength,
                "shared_entities": shared_entities
            }
        
        # If no direct relationship, check for paths
        try:
            path = nx.shortest_path(self.graph, source=context_id1, target=context_id2)
            
            if path and len(path) > 2:
                # Describe the indirect relationship through the path
                node1 = self.graph.nodes[context_id1]
                node2 = self.graph.nodes[context_id2]
                summary1 = node1.get("summary", "Context 1")
                summary2 = node2.get("summary", "Context 2")
                
                # Describe intermediate nodes
                intermediates = []
                for i in range(1, len(path) - 1):
                    inter_node = self.graph.nodes[path[i]]
                    inter_summary = inter_node.get("summary", f"Context {path[i]}")
                    intermediates.append(inter_summary)
                
                # Create explanation of indirect relationship
                if len(intermediates) == 1:
                    via_text = f"via {intermediates[0]}"
                else:
                    via_text = f"via {', '.join(intermediates[:-1])} and {intermediates[-1]}"
                
                explanation = f"Context 1 is indirectly connected to Context 2 {via_text}."
                
                return {
                    "explanation": explanation,
                    "context1_summary": summary1,
                    "context2_summary": summary2,
                    "relationship_type": "indirect",
                    "path": path,
                    "path_length": len(path) - 1
                }
            
        except nx.NetworkXNoPath:
            pass
        
        # No relationship found
        return {"explanation": "No relationship exists between these contexts"}
    
    def find_communities(self) -> Dict[int, List[str]]:
        """
        Identify communities (clusters) of related contexts.
        
        Returns:
            Dictionary mapping community IDs to lists of context IDs
        """
        # Use Louvain method for community detection
        try:
            from community import best_partition
            partition = best_partition(self.graph)
            
            # Group contexts by community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
                
            return dict(communities)
        except ImportError:
            # Fallback to a simpler approach if python-louvain is not available
            connected_components = list(nx.connected_components(self.graph))
            communities = {i: list(component) for i, component in enumerate(connected_components)}
            return communities
    
    def calculate_centrality(self, centrality_type: str = "eigenvector") -> Dict[str, float]:
        """
        Calculate centrality scores for contexts in the graph.
        
        Args:
            centrality_type: Type of centrality to calculate ('eigenvector', 'degree', 'betweenness')
            
        Returns:
            Dictionary mapping context IDs to centrality scores
        """
        if centrality_type == "eigenvector":
            return nx.eigenvector_centrality_numpy(self.graph, weight='weight')
        elif centrality_type == "degree":
            return nx.degree_centrality(self.graph)
        elif centrality_type == "betweenness":
            return nx.betweenness_centrality(self.graph, weight='weight')
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
    
    def get_graph_stats(self) -> Dict:
        """
        Calculate statistics about the graph structure.
        
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "relationship_types": dict(self.relationship_types),
            "last_update_time": self.last_update_time
        }
        
        # Calculate additional stats if graph is not empty
        if self.graph.number_of_nodes() > 0:
            stats["average_degree"] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            
            # Calculate clustering coefficient if graph has at least 3 nodes
            if self.graph.number_of_nodes() >= 3:
                try:
                    stats["clustering_coefficient"] = nx.average_clustering(self.graph, weight='weight')
                except:
                    stats["clustering_coefficient"] = 0.0
                
            # Calculate diameter if graph is connected and has at least 2 nodes
            if self.graph.number_of_nodes() >= 2:
                try:
                    if nx.is_connected(self.graph):
                        stats["diameter"] = nx.diameter(self.graph)
                except:
                    # Graph might not be connected
                    stats["diameter"] = -1
                    
        return stats
    
    def find_knowledge_gaps(self) -> List[Dict]:
        """
        Identify potential gaps in the knowledge graph.
        
        Returns:
            List of dictionaries describing identified gaps
        """
        gaps = []
        
        # If the graph is very small, it's inherently sparse
        if self.graph.number_of_nodes() < 3:
            return [{
                "type": "sparse_graph",
                "description": "The knowledge graph is very small, consider adding more contexts."
            }]
        
        # 1. Find disconnected components
        components = list(nx.connected_components(self.graph))
        if len(components) > 1:
            gaps.append({
                "type": "disconnected_components",
                "description": f"The knowledge graph has {len(components)} disconnected components.",
                "components": [list(c) for c in components]
            })
        
        # 2. Identify potential bridge contexts (removing them would disconnect the graph)
        if self.graph.number_of_nodes() > 2:
            try:
                articulation_points = list(nx.articulation_points(self.graph))
                if articulation_points:
                    gaps.append({
                        "type": "bridge_contexts",
                        "description": "These contexts act as bridges between different parts of the knowledge graph.",
                        "contexts": articulation_points
                    })
            except:
                pass
        
        # 3. Identify sparse areas (nodes with few connections)
        degree_dict = dict(self.graph.degree())
        avg_degree = sum(degree_dict.values()) / max(1, len(degree_dict))
        sparse_nodes = [node for node, degree in degree_dict.items() if degree < avg_degree / 2]
        
        if sparse_nodes:
            gaps.append({
                "type": "sparse_areas",
                "description": "These contexts have fewer connections than expected.",
                "contexts": sparse_nodes
            })
        
        # 4. Check for balance in relationship types
        relationship_counts = dict(self.relationship_types)
        if len(relationship_counts) > 0:
            total_relationships = sum(relationship_counts.values())
            dominant_type = max(relationship_counts.items(), key=lambda x: x[1])
            
            # If one type accounts for more than 80% of relationships
            if total_relationships > 0 and dominant_type[1] / total_relationships > 0.8:
                gaps.append({
                    "type": "relationship_imbalance",
                    "description": f"Relationship type '{dominant_type[0]}' accounts for {dominant_type[1] / total_relationships:.1%} of all relationships.",
                    "dominant_type": dominant_type[0],
                    "counts": relationship_counts
                })
        
        # 5. Check for concept islands (critical entities that appear in only one context)
        entity_appearances = {}
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            for entity in node_data.get("critical_entities", []):
                if entity not in entity_appearances:
                    entity_appearances[entity] = []
                entity_appearances[entity].append(node)
        
        isolated_entities = {entity: nodes for entity, nodes in entity_appearances.items() if len(nodes) == 1}
        
        if isolated_entities:
            gaps.append({
                "type": "isolated_concepts",
                "description": f"Found {len(isolated_entities)} concepts that appear in only one context.",
                "entities": isolated_entities
            })
        
        return gaps
    
    def visualize(self, highlight_nodes: List[str] = None, title: str = "Context Graph", 
                 figsize: Tuple[int, int] = (12, 10), show: bool = True) -> Optional[plt.Figure]:
        """
        Create a visualization of the context graph.
        
        Args:
            highlight_nodes: List of node IDs to highlight
            title: Title for the visualization
            figsize: Figure size as (width, height)
            show: Whether to display the plot or return the figure
            
        Returns:
            Matplotlib figure if show=False, otherwise None
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, nothing to visualize")
            return None
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Get edge weights for line thickness
        edge_weights = [self.graph[u][v].get("weight", 0.1) * 5 for u, v in self.graph.edges()]
        
        # Prepare node colors
        node_colors = ["skyblue" for _ in self.graph.nodes()]
        
        # Highlight specified nodes
        if highlight_nodes:
            for i, node in enumerate(self.graph.nodes()):
                if node in highlight_nodes:
                    node_colors[i] = "orange"
                    
        # Draw the graph
        nx.draw_networkx(
            self.graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            width=edge_weights,
            alpha=0.7,
            font_size=8,
            node_size=700,
            ax=ax
        )
        
        # Set title and remove axis
        plt.title(title)
        plt.axis("off")
        
        if show:
            plt.tight_layout()
            plt.show()
            return None
        else:
            return fig
    
    def enhance_connections(self, min_similarity: float = 0.6, max_suggestions: int = 10) -> List[Dict]:
        """
        Enhance the context graph by suggesting and adding missing connections using LLM analysis.
        
        This method uses the LLM Context Linker to identify potentially related contexts
        that are not directly connected in the graph.
        
        Args:
            min_similarity: Minimum similarity threshold for suggested connections
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries with information about suggested connections
        """
        if not self.use_llm or self.llm_linker is None:
            return [{
                "success": False,
                "reason": "LLM Context Linker is not enabled or available"
            }]
            
        try:
            # Get suggestions from LLM Context Linker with a timeout
            suggested_links = self.llm_linker.suggest_missing_links(self, min_similarity=min_similarity)
        except Exception as e:
            logger.error(f"Error getting link suggestions from LLM: {e}")
            return [{
                "success": False,
                "reason": f"Error in LLM processing: {str(e)}"
            }]
        
        # Limit the number of suggestions
        suggestions = suggested_links[:max_suggestions]
        
        # Process each suggestion
        results = []
        for suggestion in suggestions:
            source_id = suggestion["source_id"]
            target_id = suggestion["target_id"]
            
            # Skip if the link already exists
            if self.graph.has_edge(source_id, target_id):
                results.append({
                    "success": False,
                    "reason": "Link already exists",
                    "source_id": source_id,
                    "target_id": target_id
                })
                continue
                
            # Create the link
            relationship_type = suggestion.get("relationship_type", "suggested_by_llm")
            strength = suggestion.get("strength", 0.0)
            bridge_concepts = suggestion.get("bridge_concepts", [])
            explanation = suggestion.get("explanation", "")
            
            try:
                # Link the contexts
                link_result = self.link_contexts(
                    source_id,
                    target_id,
                    relevance_score=strength,
                    relationship_type=relationship_type,
                    shared_entities=bridge_concepts,
                    use_llm_analysis=False  # Already analyzed with LLM
                )
                
                # Update edge with explanation
                if explanation and self.graph.has_edge(source_id, target_id):
                    if "relationship_info" not in self.graph[source_id][target_id]:
                        self.graph[source_id][target_id]["relationship_info"] = {}
                    self.graph[source_id][target_id]["relationship_info"]["explanation"] = explanation
                
                # Add to results
                link_result["was_suggested"] = True
                link_result["explanation"] = explanation
                results.append(link_result)
                
            except Exception as e:
                results.append({
                    "success": False,
                    "reason": f"Error creating link: {str(e)}",
                    "source_id": source_id,
                    "target_id": target_id
                })
        
        return results
    
    def save_to_json(self, filepath: str) -> bool:
        """
        Save the graph to a JSON file.
        
        Args:
            filepath: Path to the file to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert graph to data structure that can be serialized to JSON
            data = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "relationship_types": dict(self.relationship_types),
                    "last_update_time": self.last_update_time
                }
            }
            
            # Add nodes with their attributes
            for node, attrs in self.graph.nodes(data=True):
                # Convert numpy arrays to lists for JSON serialization
                node_data = {}
                for key, value in attrs.items():
                    if isinstance(value, np.ndarray):
                        node_data[key] = value.tolist()
                    else:
                        node_data[key] = value
                        
                data["nodes"].append({
                    "id": node,
                    "attributes": node_data
                })
                
            # Add edges with their attributes
            for u, v, attrs in self.graph.edges(data=True):
                data["edges"].append({
                    "source": u,
                    "target": v,
                    "attributes": attrs
                })
                
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving graph to {filepath}: {e}")
            return False
    
    def load_from_json(self, filepath: str) -> bool:
        """
        Load the graph from a JSON file.
        
        Args:
            filepath: Path to the file to load from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Create a new graph
            new_graph = nx.Graph()
            
            # Add nodes with their attributes
            for node_data in data["nodes"]:
                new_graph.add_node(node_data["id"], **node_data["attributes"])
                
            # Add edges with their attributes
            for edge_data in data["edges"]:
                new_graph.add_edge(
                    edge_data["source"],
                    edge_data["target"],
                    **edge_data["attributes"]
                )
                
            # Update the graph and metadata
            self.graph = new_graph
            self.relationship_types = defaultdict(int)
            for rt, count in data["metadata"]["relationship_types"].items():
                self.relationship_types[rt] = count
                
            self.last_update_time = data["metadata"].get("last_update_time", time.time())
            
            return True
        except Exception as e:
            print(f"Error loading graph from {filepath}: {e}")
            return False
