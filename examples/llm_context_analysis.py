#!/usr/bin/env python3
"""
LLM-Enhanced Context Linking Demo

This script demonstrates the use of language models to enhance context linking
in the Adaptive Knowledge System, providing more meaningful relationships and explanations.
"""

import asyncio
import os
import sys
import time

# Add the parent directory to the Python path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge import AdaptiveKnowledgeSystem


async def main():
    """Run the LLM-enhanced context linking demo."""
    print("Initializing Adaptive Knowledge System with LLM enhancement...")
    
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_llm")
    os.makedirs(data_dir, exist_ok=True)
    
    # Flag to track whether LLM enhancement is working
    use_llm = True

    try:
        # Try to initialize with LLM enhancement
        knowledge_system = AdaptiveKnowledgeSystem(
            storage_dir=data_dir,
            use_llm=True,  # Enable LLM enhancement
            ollama_model="mistral-nemo:latest"  # Specify a known model to avoid auto-detection issues
        )
        
        # Check if LLM is actually available
        if not (hasattr(knowledge_system.context_graph, 'llm_linker') and 
                knowledge_system.context_graph.llm_linker is not None):
            print("LLM enhancement not available. Falling back to standard mode.")
            use_llm = False
    except Exception as e:
        print(f"Error initializing with LLM: {e}")
        print("Falling back to standard mode without LLM enhancement.")
        knowledge_system = AdaptiveKnowledgeSystem(
            storage_dir=data_dir,
            use_llm=False  # Disable LLM enhancement
        )
        use_llm = False
    
    print("\n1. Adding Diverse Knowledge to the System")
    print("-----------------------------------------")
    
    # Add knowledge about various domains with nuanced relationships
    print("Adding knowledge about neural networks...")
    nn_context_id = await knowledge_system.add_knowledge(
        "Neural Networks are a set of algorithms, modeled loosely after the human brain, that "
        "are designed to recognize patterns. They interpret sensory data through a kind of machine "
        "perception, labeling or clustering raw input. The neural network contains layers: "
        "the input layer, hidden layers, and the output layer.",
        ["neural networks", "algorithms", "patterns", "layers", "machine learning"]
    )
    print(f"Added with context ID: {nn_context_id}")
    
    print("\nAdding knowledge about backpropagation...")
    bp_context_id = await knowledge_system.add_knowledge(
        "Backpropagation is a method used to train neural networks by adjusting the weights "
        "based on the error rate obtained in the previous iteration. It is an important "
        "mathematical tool for calculating the gradient of the loss function with respect "
        "to the weights. Backpropagation works by calculating the gradient of the loss function "
        "with respect to each weight, allowing the weights to be updated to minimize the loss.",
        ["backpropagation", "weights", "neural networks", "training", "gradient descent"]
    )
    print(f"Added with context ID: {bp_context_id}")
    
    print("\nAdding knowledge about gradient descent...")
    gd_context_id = await knowledge_system.add_knowledge(
        "Gradient Descent is an optimization algorithm used to minimize a function by iteratively "
        "moving in the direction of steepest descent as defined by the negative of the gradient. "
        "It's commonly used in machine learning to find the values of a function's parameters that "
        "minimize a cost function. The algorithm updates the parameters in the opposite direction "
        "of the gradient of the objective function.",
        ["gradient descent", "optimization", "machine learning", "parameters", "cost function"]
    )
    print(f"Added with context ID: {gd_context_id}")
    
    print("\nAdding knowledge about semiconductor manufacturing...")
    sm_context_id = await knowledge_system.add_knowledge(
        "Semiconductor manufacturing is the process used to create chips and other integrated "
        "circuits that are used in electronics. It involves a sequence of photographic and chemical "
        "processing steps during which electronic circuits are gradually created on a wafer of "
        "semiconductor material. Silicon is the most commonly used semiconductor material, although "
        "gallium arsenide, germanium, and others are also used.",
        ["semiconductors", "manufacturing", "integrated circuits", "silicon", "electronics"]
    )
    
    print(f"Added with context ID: {sm_context_id}")
    
    # Allow time for graph relationships to be computed
    print("\nBuilding knowledge graph relationships...")
    await asyncio.sleep(1)
    
    print("\n2. Exploring Relationships with LLM Analysis")
    print("------------------------------------------")
    
    print("Analyzing relationship between neural networks and backpropagation...")
    
    # Get related contexts with LLM explanations
    nn_related = knowledge_system.context_graph.get_related_contexts(
        nn_context_id,
        include_explanations=True
    )
    
    # Display relationships
    for relation in nn_related:
        print(f"\nRelationship to: {relation['id']}")
        print(f"Relationship type: {relation['relationship_type']}")
        print(f"Relevance score: {relation['relevance']:.2f}")
        print(f"Shared entities: {', '.join(relation['shared_entities'])}")
        
        if "explanation" in relation:
            print(f"Explanation: {relation['explanation']}")
    
    print("\n3. Enhancing Context Links with LLM")
    print("----------------------------------")
    
    print("Using LLM to suggest potential missing links...")
    
    # Use LLM to enhance connections
    enhancement_results = await knowledge_system.enhance_knowledge_links(
        min_similarity=0.4,
        max_suggestions=3
    )
    
    # Display results
    for i, result in enumerate(enhancement_results, 1):
        print(f"\nSuggestion {i}:")
        if result.get("success", False):
            print(f"Created link between: {result['context1_id']} and {result['context2_id']}")
            print(f"Relationship type: {result['relationship_type']}")
            print(f"Strength: {result['relevance_score']:.2f}")
            if "explanation" in result:
                print(f"Explanation: {result['explanation']}")
        else:
            print(f"Failed: {result.get('reason', 'Unknown reason')}")
    
    print("\n4. Querying with Enhanced Context Links")
    print("--------------------------------------")
    
    # Query for knowledge with explanations
    print("Querying for information about 'optimization algorithms'...")
    query_results = await knowledge_system.query_knowledge(
        "optimization algorithms",
        include_explanations=True
    )
    
    # Display the results
    for i, result in enumerate(query_results, 1):
        print(f"\nResult {i} (Relevance: {result.get('relevance_score', 0):.2f})")
        print(f"Summary: {result.get('summary', 'No summary')}")
        if "explanation" in result:
            print(f"Explanation of relevance: {result['explanation']}")
    
    print("\n5. Testing Context Clustering with LLM Analysis")
    print("---------------------------------------------")
    
    # Find communities in the knowledge graph
    communities = knowledge_system.context_graph.find_communities()
    
    # Use LLM to explain the first community
    if communities and 0 in communities:
        community_contexts = communities[0]
        
        print(f"Analyzing cluster with {len(community_contexts)} contexts...")
        
        if hasattr(knowledge_system.context_graph, 'llm_linker') and knowledge_system.context_graph.llm_linker:
            cluster_analysis = knowledge_system.context_graph.llm_linker.explain_context_cluster(
                knowledge_system.context_graph,
                community_contexts
            )
            
            print(f"\nCluster theme: {cluster_analysis.get('theme', 'No theme identified')}")
            print(f"Cluster summary: {cluster_analysis.get('summary', 'No summary available')}")
            print(f"Key concepts: {', '.join(cluster_analysis.get('key_concepts', []))}")
    
    print("\n6. Visualizing the Enhanced Knowledge Graph")
    print("-----------------------------------------")
    
    # Visualize the knowledge graph
    output_file = os.path.join(data_dir, "knowledge_graph_llm.png")
    viz_result = await knowledge_system.visualize_knowledge_graph(
        output_file=output_file
    )
    
    if viz_result:
        print(f"Knowledge graph visualization saved to {output_file}")
    
    print("\nDemo completed successfully!")
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
