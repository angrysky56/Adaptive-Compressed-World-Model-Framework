"""
Visualization utilities for the Adaptive Compressed World Model Framework.

This module provides helper functions for visualizing various aspects of the framework,
including agent positions, knowledge graphs, and convergence metrics.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def plot_agent_positions(agent_positions: Dict[str, Tuple[float, float]], 
                        perception_radii: Dict[str, float] = None,
                        resource_positions: Dict[str, Tuple[float, float]] = None,
                        env_size: Tuple[int, int] = (100, 100),
                        title: str = "Agent Positions"):
    """
    Plot agent positions in the environment.
    
    Args:
        agent_positions: Dictionary mapping agent IDs to (x, y) positions
        perception_radii: Optional dictionary mapping agent IDs to perception radii
        resource_positions: Optional dictionary mapping resource IDs to (x, y) positions
        env_size: Size of the environment as (width, height)
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set limits
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    
    # Plot agents
    for agent_id, position in agent_positions.items():
        ax.plot(position[0], position[1], 'bo', markersize=10, label=f"Agent {agent_id}")
        
        # Draw perception radius if provided
        if perception_radii and agent_id in perception_radii:
            perception_circle = plt.Circle(
                position, 
                perception_radii[agent_id], 
                fill=False, 
                linestyle='--', 
                color='blue', 
                alpha=0.5
            )
            ax.add_patch(perception_circle)
    
    # Plot resources if provided
    if resource_positions:
        for resource_id, position in resource_positions.items():
            ax.plot(position[0], position[1], 'go', markersize=6)
    
    # Handle legend (only show unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_knowledge_graph(knowledge_graph: nx.Graph, title: str = "Knowledge Graph"):
    """
    Visualize a knowledge graph.
    
    Args:
        knowledge_graph: NetworkX graph object
        title: Title for the plot
    """
    if not knowledge_graph.nodes:
        print("Knowledge graph is empty")
        return
        
    # Create the plot
    plt.figure(figsize=(14, 12))
    
    # Get node types and set colors
    node_colors = []
    for node in knowledge_graph.nodes:
        node_type = knowledge_graph.nodes[node].get("type", "unknown")
        if node_type == "agent":
            node_colors.append("skyblue")
        elif node_type == "behavior":
            node_colors.append("lightgreen")
        elif node_type == "knowledge":
            node_colors.append("lightyellow")
        else:
            node_colors.append("lightgray")
    
    # Get edge types and set colors
    edge_colors = []
    for _, _, data in knowledge_graph.edges(data=True):
        edge_type = data.get("type", "unknown")
        if edge_type == "world_model_similarity":
            edge_colors.append("red")
        elif edge_type == "exhibits_behavior":
            edge_colors.append("green")
        elif edge_type == "contains_knowledge":
            edge_colors.append("blue")
        else:
            edge_colors.append("gray")
    
    # Use spring layout for positioning
    pos = nx.spring_layout(knowledge_graph, seed=42)
    
    # Draw the graph
    nx.draw_networkx(
        knowledge_graph,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=600,
        font_size=8,
        alpha=0.7
    )
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_convergence_heatmap(similarity_matrix: np.ndarray, 
                             agent_ids: List[str],
                             title: str = "Knowledge Similarity Between Agents"):
    """
    Plot a heatmap of the similarity between agent knowledge models.
    
    Args:
        similarity_matrix: 2D numpy array of similarity values
        agent_ids: List of agent IDs corresponding to matrix indices
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(agent_ids)))
    ax.set_yticks(np.arange(len(agent_ids)))
    ax.set_xticklabels(agent_ids)
    ax.set_yticklabels(agent_ids)
    
    # Rotate the x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Similarity')
    
    # Add similarity values in each cell
    for i in range(len(agent_ids)):
        for j in range(len(agent_ids)):
            text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.7 else "white")
    
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_convergence_over_time(time_steps: List[int], 
                              convergence_values: List[float],
                              title: str = "Knowledge Convergence Over Time"):
    """
    Plot the convergence of agent knowledge over time.
    
    Args:
        time_steps: List of time step values
        convergence_values: List of convergence metric values
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, convergence_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Simulation Time Step')
    plt.ylabel('Average Convergence (0-1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
