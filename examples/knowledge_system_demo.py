#!/usr/bin/env python3
"""
Knowledge System Demo

This script demonstrates the basic functionality of the Adaptive Knowledge System
with compression, dynamic context linking, and event-triggered updates.
"""

import asyncio
import os
import sys
import time

# Add the parent directory to the Python path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge import AdaptiveKnowledgeSystem


async def main():
    """Run the knowledge system demo."""
    print("Initializing Adaptive Knowledge System...")
    
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize the knowledge system with storage
    knowledge_system = AdaptiveKnowledgeSystem(storage_dir=data_dir)
    
    print("\n1. Adding Knowledge to the System")
    print("---------------------------------")
    
    # Add some knowledge about AI concepts
    print("Adding knowledge about machine learning...")
    ml_context_id = await knowledge_system.add_knowledge(
        "Machine Learning is a subset of artificial intelligence that involves training algorithms "
        "on data to make predictions or decisions without being explicitly programmed to do so. "
        "Common techniques include neural networks, decision trees, and support vector machines.",
        ["machine learning", "algorithms", "neural networks", "decision trees"]
    )
    print(f"Added with context ID: {ml_context_id}")
    
    print("\nAdding knowledge about neural networks...")
    nn_context_id = await knowledge_system.add_knowledge(
        "Neural Networks are a set of algorithms, modeled loosely after the human brain, that "
        "are designed to recognize patterns. They interpret sensory data through a kind of machine "
        "perception, labeling or clustering raw input. The neural network contains three types of layers: "
        "the input layer, hidden layers, and the output layer.",
        ["neural networks", "algorithms", "patterns", "layers"]
    )
    print(f"Added with context ID: {nn_context_id}")
    
    print("\nAdding knowledge about deep learning...")
    dl_context_id = await knowledge_system.add_knowledge(
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers "
        "(deep neural networks) to analyze various factors of data. Deep learning is behind many recent "
        "advances in AI, including computer vision and natural language processing.",
        ["deep learning", "neural networks", "computer vision", "natural language processing"]
    )
    print(f"Added with context ID: {dl_context_id}")
    
    # Allow time for graph relationships to be computed
    print("\nBuilding knowledge graph relationships...")
    await asyncio.sleep(1)
    
    print("\n2. Querying the Knowledge System")
    print("-------------------------------")
    
    # Query for knowledge about neural networks
    print("Querying for information about 'neural networks and deep learning'...")
    query_results = await knowledge_system.query_knowledge("neural networks and deep learning")
    
    # Display the results
    for i, result in enumerate(query_results, 1):
        print(f"\nResult {i} (Relevance: {result['relevance_score']:.2f})")
        print(f"Summary: {result['summary']}")
    
    print("\n3. Expanding Knowledge")
    print("---------------------")
    
    # Expand a specific context
    print(f"Expanding the neural networks context ({nn_context_id})...")
    nn_expanded = await knowledge_system.expand_knowledge(nn_context_id)
    
    print("\nExpanded content:")
    print(nn_expanded["expanded_content"])
    
    print("\nRelated contexts:")
    for related in nn_expanded["related_contexts"]:
        print(f"- {related['summary']} (Relevance: {related['relevance']:.2f})")
    
    print("\n4. Testing Event-Triggered Updates")
    print("---------------------------------")
    
    # Update with a small change (might not trigger an update)
    print("\nMaking a small update to deep learning context...")
    small_update = await knowledge_system.update_knowledge(
        dl_context_id,
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers "
        "to analyze various factors of data. Deep learning is behind many recent "
        "advances in AI, including computer vision and natural language processing."
    )
    print(f"Update triggered: {small_update}")
    
    # Update with a more significant change (should trigger an update)
    print("\nMaking a significant update to deep learning context...")
    large_update = await knowledge_system.update_knowledge(
        dl_context_id,
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers "
        "(deep neural networks) to analyze various factors of data. Recent advances in deep learning "
        "include transformer models like BERT and GPT, which have revolutionized natural language processing. "
        "Deep learning is also used in computer vision, reinforcement learning, and generative models."
    )
    print(f"Update triggered: {large_update}")
    
    # Check the updated content
    if large_update:
        print("\nRetrieving updated deep learning context...")
        dl_expanded = await knowledge_system.expand_knowledge(dl_context_id)
        print("\nUpdated content:")
        print(dl_expanded["expanded_content"])
    
    print("\n5. Knowledge Graph Analysis")
    print("-------------------------")
    
    # Generate a summary of the knowledge system
    print("Generating knowledge summary...")
    summary = await knowledge_system.generate_knowledge_summary()
    
    print(f"\nTotal contexts: {summary['total_contexts']}")
    print(f"Average compression ratio: {summary['avg_compression_ratio']:.2f}")
    print(f"Graph statistics:")
    for key, value in summary['graph_stats'].items():
        print(f"  - {key}: {value}")
    
    # Visualize the knowledge graph
    print("\nVisualizing knowledge graph...")
    output_file = os.path.join(data_dir, "knowledge_graph.png")
    viz_result = await knowledge_system.visualize_knowledge_graph(
        highlight_contexts=[nn_context_id],
        output_file=output_file
    )
    
    if viz_result:
        print(f"Knowledge graph visualization saved to {output_file}")
    
    print("\n6. Testing Storage Hierarchy")
    print("--------------------------")
    
    # Retrieve a context that should be in cache now
    print(f"Retrieving neural networks context from cache ({nn_context_id})...")
    start_time = time.time()
    nn_context = await knowledge_system.storage.retrieve(nn_context_id)
    cache_retrieval_time = time.time() - start_time
    
    if nn_context:
        print(f"Retrieved from cache in {cache_retrieval_time:.6f} seconds")
        print(f"Access count: {nn_context.get('access_count', 0)}")
    
    # Simulate long-term storage retrieval
    # First clear the cache entry
    if nn_context_id in knowledge_system.storage.cache:
        del knowledge_system.storage.cache[nn_context_id]
    
    print(f"\nRetrieving neural networks context from long-term storage ({nn_context_id})...")
    start_time = time.time()
    nn_context = await knowledge_system.storage.retrieve(nn_context_id)
    storage_retrieval_time = time.time() - start_time
    
    if nn_context:
        print(f"Retrieved from long-term storage in {storage_retrieval_time:.6f} seconds")
        print(f"Cache retrieval was {storage_retrieval_time / cache_retrieval_time:.2f}x slower")
    
    print("\n7. Maintenance and Cleanup")
    print("------------------------")
    
    # Perform maintenance on the storage system
    print("Performing storage maintenance...")
    maintenance_results = await knowledge_system.maintain_storage()
    
    print("\nMaintenance results:")
    for key, value in maintenance_results.items():
        print(f"  - {key}: {value}")
    
    # Export the knowledge graph
    graph_file = os.path.join(data_dir, "knowledge_graph.json")
    print(f"\nExporting knowledge graph to {graph_file}...")
    export_result = await knowledge_system.export_knowledge_graph(graph_file)
    
    if export_result:
        print("Knowledge graph exported successfully")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
